import copy
import os
import os.path as osp
import pickle
from functools import partial

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from losses import (
    ensemble_prediction_loss,
    ensemble_prediction_loss_with_specialization,
    expected_expert_loss,
    load_balance_penalty,
    moe_nll,
)
from losses import (
    entropy_penalty as entropy_penalty_loss,
)
from model_utils import deserialize_standard_scaler, serialize_standard_scaler
from moe_pytorch_utils import (
    IndexSubset,
    _as_list,
    make_deterministic,
    train_lit_model,
)
from torch.utils.data import DataLoader

try:
    from sparsemax import Sparsemax
except ImportError:
    pass

try:
    import torchsort
except ImportError:
    torchsort = None


##################################################################
# Define modules needed for defining the Mixture-of-Experts model:


# Define the gating model:


class GateModel(nn.Module):
    """
    A generic implementation for the gating model. This function can accommodate
    linear + non-linear gating models.
    """

    def __init__(
        self,
        n_covar,
        n_experts,
        gate_add_intercept=True,
        hidden_layers=None,
        add_batch_norm=False,
        add_layer_norm=False,
        activation=nn.ReLU,
        final_activation="softmax",
    ):
        super(GateModel, self).__init__()

        if n_covar <= 0:
            raise ValueError("GateModel requires n_covar > 0.")

        self.n_covar = n_covar
        self.n_experts = n_experts
        self.gate_add_intercept = bool(gate_add_intercept)

        input_dim = n_covar
        layers = []

        is_mlp = hidden_layers is not None and len(hidden_layers) > 0
        if is_mlp:
            for layer_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, layer_dim))

                if add_batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dim))
                elif add_layer_norm:
                    layers.append(nn.LayerNorm(layer_dim))

                layers.append(activation())
                input_dim = layer_dim

        # Final linear layer:
        # - Linear gate (no hidden layers): intercept comes from Lit_MoEPRS input expansion,
        #   so keep bias=False to avoid duplicate intercept parameters.
        # - MLP gate: gate_add_intercept controls the final layer bias.
        final_bias = self.gate_add_intercept if is_mlp else False
        layers.append(nn.Linear(input_dim, n_experts, bias=final_bias))

        self.gate = nn.Sequential(*layers)

        # ------------------------------------------------------------
        # Shared output activation
        # ------------------------------------------------------------
        if final_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        elif final_activation == "sparsemax":
            self.output_activation = Sparsemax(dim=1)
        else:
            self.output_activation = nn.Identity()

    def forward(self, covar):
        logits = self.gate(covar)
        return self.output_activation(logits)

    def predict_proba(self, covar):
        return self.forward(covar)


class LinearScaler(nn.Module):
    """
    A linear model for scaling the predictions of the experts.
    """

    def __init__(self, n_covar=0, bias=False, family="gaussian"):
        super(LinearScaler, self).__init__()

        assert family in ("gaussian", "binomial")

        # The linear model takes as inputs 1 (for the PRS) + the number of covariates.
        # If there are no covariates, it just takes the PRS itself.
        self.linear_model = nn.Linear(n_covar + 1, 1, bias=bias)
        self.family = family

    def forward(self, prs, covar=None, return_logits=False):
        if len(prs.shape) < 2:
            prs = prs.reshape(-1, 1)

        if covar is None:
            pred = self.linear_model(prs)
        else:
            pred = self.linear_model(torch.cat([prs, covar], dim=1))

        if self.family == "gaussian":
            return pred

        if return_logits:
            return pred
        return torch.sigmoid(pred)


class GroupedLinearScaler(nn.Module):
    """
    Batched linear scaling for all experts at once:
      output = experts * scale + covar @ covar_coef.T
    """

    def __init__(self, n_experts, n_covar=0, add_intercept=False):
        super(GroupedLinearScaler, self).__init__()

        self.n_experts = int(n_experts)
        self.n_covar = int(n_covar)
        self.add_intercept = bool(add_intercept)

        # (K,) scale per expert for expert predictions
        self.scale = nn.Parameter(torch.ones(self.n_experts))
        # (K,C) covariate coefficients per expert
        self.covar_coef = (
            nn.Parameter(torch.zeros(self.n_experts, self.n_covar))
            if self.n_covar > 0
            else None
        )
        self.intercept = (
            nn.Parameter(torch.zeros(self.n_experts)) if self.add_intercept else None
        )

    def forward(self, experts, covar=None):
        if experts.ndim != 2:
            raise ValueError(f"`experts` must be shape (N,K), got {experts.shape}")
        if experts.shape[1] != self.n_experts:
            raise ValueError(
                f"Expected experts with K={self.n_experts}, got K={experts.shape[1]}"
            )

        out = experts * self.scale.view(1, -1)

        if self.covar_coef is not None:
            if covar is None:
                raise ValueError(
                    "expert_covariates are required when n_expert_covariates > 0"
                )
            out = out + covar.matmul(self.covar_coef.t())

        if self.intercept is not None:
            out = out + self.intercept.view(1, -1)

        return out


##################################################################
# Define a PyTorch Lightning module to streamline training


class Lit_MoEPRS(pl.LightningModule):
    @classmethod
    def default_training_kwargs(cls, seed=8):
        return {
            "loss": "nll",
            "optimizer": "Adam",
            "family": "gaussian",
            "gate_model_spec": None,
            "learning_rate": 1e-3,
            "gate_penalty": 0.0,
            "expert_penalty": 0.0,
            "entropy_penalty": 0.0,
            "entropy_schedule_params": {
                "coef_start": 0.0,
                "coef_end": 0.0,
                "warm_epochs": 0,
                "decay_epochs": 0,
            },
            "temperature_schedule_params": {
                "tau_start": 2.0,
                "tau_end": 1.0,
                "warm_epochs": 10,
                "decay_epochs": 90,
            },
            "binomial_mixing_level": "probability",
            "max_epochs": 500,
            "batch_size": 4096,
            "seed": seed,
            "topk_k": None,
            "hard_ste": False,
            "lb_coef": 0.0,
            "ancestry_balance_lambda": None,
            "expert_add_intercept": False,
        }

    def __init__(
        self,
        group_getitem_cols,
        gate_add_intercept=True,
        expert_add_intercept=False,
        gate_model_spec=None,
        loss="nll",
        optimizer="Adam",
        family="gaussian",
        learning_rate=1e-3,
        gate_penalty=0.0,
        entropy_penalty=0.0,
        expert_penalty=0.0,
        topk_k=None,  # None = disable top-k
        entropy_schedule_params=None,
        temperature_schedule_params=None,
        hard_ste=True,  # straight-through estimator
        lb_coef=0.0,  # 0.0 = disable load-balancing aux loss
        specialization_weight=0.8,
        eps=1e-8,
        binomial_mixing_level="probability",
    ):
        """
        A PyTorch Lightning module for training a mixture of experts model.

        :param group_getitem_cols: A dictionary mapping categories of data to the relevant keys from the
         pandas dataframe. This is useful for iterative data fetching (e.g. data loaders).
            These are used to define what columns/groups of columns are fetched in the __getitem__ method.
        :param loss: The loss function to use.
          Options are: ('nll', 'ensemble_loss', 'likelihood_mixture', 'hybrid_prediction_loss')
        :param optimizer: The optimizer to use. Options are: ('Adam', 'LBFGS', 'SGD')
        :param family: The family of the likelihood. Options are: ('gaussian', 'binomial')
        :param learning_rate: The learning rate for the optimizer.
        """

        super().__init__()

        self.training_scaler = None

        # -------------------------------------------------------
        # Sanity checks for the inputs:
        assert loss in (
            "nll",
            "ensemble_loss",
            "likelihood_mixture",
            "hybrid_prediction_loss",
        )
        assert optimizer in ("Adam", "LBFGS", "SGD")
        assert family in ("gaussian", "binomial")
        assert binomial_mixing_level in ("probability", "logit")

        assert "phenotype" in group_getitem_cols
        assert "gate_input" in group_getitem_cols
        assert "expert_predictions" in group_getitem_cols

        # -------------------------------------------------------
        # Define / initialize the model components:

        self.group_getitem_cols = group_getitem_cols
        self.gate_add_intercept = bool(gate_add_intercept)

        gate_model_spec = gate_model_spec or {}
        self.gate_model_spec = gate_model_spec

        # The gating model:
        self.gate_model = GateModel(
            self.gate_input_dim,
            self.n_experts,
            gate_add_intercept=self.gate_add_intercept,
            **gate_model_spec,
        )

        self.expert_add_intercept = bool(expert_add_intercept)

        # Grouped expert linear scaler for efficient batched compute:
        # scaled_{ik} = gamma_k * expert_{ik} + covar_i @ beta_k
        self.expert_scaler = GroupedLinearScaler(
            n_experts=self.n_experts,
            n_covar=self.n_expert_covariates,
            add_intercept=self.expert_add_intercept,
        )

        self.global_head = None

        if "global_covariates" in group_getitem_cols:
            self.global_head = nn.Linear(
                len(group_getitem_cols["global_covariates"]), 1, bias=True
            )

        # ----------------------------------------------------------------
        # Losses and penalties:

        self.loss = loss
        self.family = family
        self.binomial_mixing_level = binomial_mixing_level

        self.metrics = {
            "nll": moe_nll,
            "ensemble_loss": ensemble_prediction_loss,
            "likelihood_mixture": expected_expert_loss,
            "hybrid_prediction_loss": partial(
                ensemble_prediction_loss_with_specialization,
                specialization_weight=specialization_weight,
            ),
        }

        self.gate_penalty = float(gate_penalty)
        self.expert_penalty = float(expert_penalty)
        self.lb_coef = float(lb_coef)  # load balancing coefficient

        # ----------------------------------------------------------------
        # Optimizer options:

        self.optimizer = optimizer
        self.lr = learning_rate

        self.topk_k = topk_k  # top_k routing

        # -----------------------------------------------
        # Entropy penalty coefficient scheduling:

        self.entropy_schedule_params = {
            "coef_start": float(entropy_penalty),
            "coef_end": float(entropy_penalty),
            "warm_epochs": 0,
            "decay_epochs": 0,
        }

        if entropy_schedule_params is not None:
            assert isinstance(entropy_schedule_params, dict)
            self.entropy_schedule_params.update(entropy_schedule_params)

        if self.entropy_schedule_params["coef_end"] is None:
            self.entropy_schedule_params["coef_end"] = self.entropy_schedule_params[
                "coef_start"
            ]

        # -----------------------------------------------
        # temperature scheduling

        self.temperature_schedule_params = {
            "tau_start": 1.0,  # 1.0 = no temperature
            "tau_end": 1.0,  # 1.0 = no temperature
            "warm_epochs": 0,
            "decay_epochs": 0,
        }

        if temperature_schedule_params is not None:
            assert isinstance(temperature_schedule_params, dict)
            self.temperature_schedule_params.update(temperature_schedule_params)

        # -----------------------------------------------

        self.hard_ste = bool(hard_ste)
        self.eps = float(eps)

        self.log_sigma2 = None

        # For guassian MoE with per-expert residual variance
        if family == "gaussian" and loss != "ensemble_loss":
            # estimated residual variance per expert
            self.log_sigma2 = nn.Parameter(torch.zeros(self.n_experts))  # init sigma2=1

    @property
    def n_experts(self):
        return len(self.group_getitem_cols["expert_predictions"])

    @property
    def gate_input_dim(self):
        return len(self.group_getitem_cols["gate_input"]) + int(self.gate_add_intercept)

    @property
    def n_expert_covariates(self):
        if "expert_covariates" in self.group_getitem_cols:
            return len(self.group_getitem_cols["expert_covariates"])
        else:
            return 0

    @property
    def sigma2(self):
        if self.log_sigma2 is not None:
            return torch.exp(self.log_sigma2)

    @property
    def current_entropy_penalty(self):
        return float(self._current_ent_coef())

    def _compute_penalty(self, batch):
        zero = batch["gate_input"].new_zeros(())
        total = zero

        penalties = {
            "gate_penalty": None,
            "expert_penalty": None,
            "load_balance": None,
            "entropy": None,
            "entropy_coef": self.current_entropy_penalty,
        }

        if self.gate_penalty > 0.0:
            gate_l2 = zero
            for name, p in self.gate_model.named_parameters():
                if name.endswith("bias"):
                    continue
                gate_l2 = gate_l2 + (p**2).sum()
            penalties["gate_penalty"] = gate_l2
            total = total + self.gate_penalty * gate_l2

        if self.expert_penalty > 0.0:
            expert_l2 = zero
            for name, p in self.expert_scaler.named_parameters():
                if name.endswith("bias") or "intercept" in name:
                    continue
                expert_l2 = expert_l2 + (p**2).sum()
            penalties["expert_penalty"] = expert_l2
            total = total + self.expert_penalty * expert_l2

        if self.lb_coef > 0.0:
            p_dense = self.gate_model.forward(self._gate_model_input(batch))
            lb = load_balance_penalty(p_dense, eps=self.eps)
            penalties["load_balance"] = lb
            total = total + self.lb_coef * lb

        ent_coef = penalties["entropy_coef"]
        if ent_coef > 0.0:
            p = self.gate_forward(batch)
            ent = entropy_penalty_loss(p, eps=self.eps)
            penalties["entropy"] = ent
            total = total + ent_coef * ent

        return total, penalties

    def batch_step(self, batch, batch_idx):
        losses = {}
        for m, loss_fn in self.metrics.items():
            losses[m] = loss_fn(batch, self)

        return losses

    def training_step(self, batch, batch_idx):

        losses = self.batch_step(batch, batch_idx)
        total = losses[self.loss]
        penalty_total, penalties = self._compute_penalty(batch)
        total = total + penalty_total

        # --------------------------------------------------
        # Logging:

        for pen_name, pen_val in penalties.items():
            if pen_val is None:
                continue
            self.log(
                pen_name,
                pen_val,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        self.log("train_loss", total, prog_bar=True)

        for m, loss in losses.items():
            if m != self.loss:
                self.log(m, loss, prog_bar=True)

        # --------------------------------------------------

        return total

    def validation_step(self, batch, batch_idx):
        losses = self.batch_step(batch, batch_idx)
        total = losses[self.loss]

        # Log final val loss
        self.log("val_loss", total, prog_bar=True)

        return total

    def _expert_linear_predictor(self, batch):
        expert_covariates = batch.get("expert_covariates", None)
        return self.expert_scaler.forward(
            batch["expert_predictions"], covar=expert_covariates
        )

    def _global_linear_predictor(self, batch):
        if self.global_head is None:
            return None

        global_in = batch.get("global_covariates")
        return self.global_head(global_in).squeeze(-1)

    def _combined_linear_predictor(self, batch):
        expert_pred = self._expert_linear_predictor(batch)  # (N,K)
        global_pred = self._global_linear_predictor(batch)  # (N,) or None
        if global_pred is None:
            return expert_pred
        return expert_pred + global_pred.unsqueeze(1)

    def _current_tau(self):
        p = self.temperature_schedule_params
        tau_start = float(p.get("tau_start", 1.0))
        tau_end = float(p.get("tau_end", tau_start))
        warm = int(p.get("warm_epochs", 0))
        decay = int(p.get("decay_epochs", 0))

        # if no trainer, default to final value
        try:
            _ = self.trainer
        except Exception:
            return tau_end

        e = int(getattr(self, "current_epoch", 0))

        # No schedule => constant
        if warm <= 0 and decay <= 0:
            return tau_start

        # Warmup: keep tau_start
        if e < warm:
            return tau_start

        # Linear decay
        if e < warm + decay:
            t = (e - warm) / max(1, decay)
            return tau_start + (tau_end - tau_start) * t

        # After schedule
        return tau_end

    def _current_ent_coef(self):
        p = self.entropy_schedule_params
        c_start = float(p.get("coef_start", 0.0))
        c_end = p.get("coef_end", c_start)
        c_end = c_start if c_end is None else float(c_end)
        warm = int(p.get("warm_epochs", 0))
        decay = int(p.get("decay_epochs", 0))

        # if no trainer, default to end value
        try:
            _ = self.trainer
        except Exception:
            return c_end

        e = int(getattr(self, "current_epoch", 0))

        # No schedule => constant
        if warm <= 0 and decay <= 0:
            return c_start

        # Warmup
        if e < warm:
            return c_start

        # Linear decay
        if e < warm + decay:
            t = (e - warm) / max(1, decay)
            return c_start + (c_end - c_start) * t

        # Floor
        return c_end

    def _apply_temperature(self, p):
        # p: (N,K) already softmaxed; sharpen via p^(1/tau) and renormalize
        tau = self._current_tau()
        if abs(tau - 1.0) < 1e-8:
            return p
        p_tau = (p.clamp_min(self.eps)) ** (1.0 / tau)
        p_tau = p_tau / (p_tau.sum(dim=1, keepdim=True) + self.eps)
        return p_tau

    def _apply_topk(self, p):
        # p: (N,K) after temperature
        if self.topk_k is None or self.topk_k >= self.n_experts:
            return p

        k = int(self.topk_k)

        # ---- smooth / differentiable approximation ----
        if torchsort is None:
            # fallback: hard only (no smooth surrogate)
            vals, idx = torch.topk(p, k, dim=1)
            hard_mask = torch.zeros_like(p).scatter_(1, idx, 1.0)
            p_hard = p * hard_mask
            return p_hard / (p_hard.sum(dim=1, keepdim=True) + self.eps)

        ranks = torchsort.soft_rank(-p, regularization_strength=1.0)  # (N, K)
        sharpness = 10.0
        threshold = float(k) + 0.5
        mask = torch.sigmoid((threshold - ranks) * sharpness)  # (N, K)

        p_soft = p * mask
        p_soft = p_soft / (p_soft.sum(dim=1, keepdim=True) + self.eps)

        # If you want fully smooth behavior, stop here:
        if not self.hard_ste:
            return p_soft

        # ---- hard top-k forward pass (EXACTLY k active) ----
        vals, idx = torch.topk(p, k, dim=1)
        hard_mask = torch.zeros_like(p).scatter_(1, idx, 1.0)
        p_hard = p * hard_mask
        p_hard = p_hard / (p_hard.sum(dim=1, keepdim=True) + self.eps)

        # STE: forward uses p_hard, backward uses p_soft
        return p_hard + (p_soft - p_soft.detach())

    def _gate_model_input(self, batch):
        x = batch["gate_input"]
        if self.gate_add_intercept:
            ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
            x = torch.cat([ones, x], dim=1)
        return x

    def gate_forward(self, batch, return_dense=False):
        # base dense softmax from GateModel: (N,K)
        p_soft = self.gate_model.forward(self._gate_model_input(batch))
        # temperature sharpening
        p_tau = self._apply_temperature(p_soft)
        # optional top-k sparsification
        p_used = self._apply_topk(p_tau)

        if return_dense:
            return p_used, p_soft
        return p_used

    def forward(self, batch):
        w = self.gate_forward(batch)  # (N,K)
        pred_lin = self._combined_linear_predictor(batch)  # (N,K)

        if self.family == "gaussian":
            return (w * pred_lin).sum(dim=1)

        else:
            if self.binomial_mixing_level == "logit":
                return torch.sigmoid((w * pred_lin).sum(dim=1))
            else:
                return (w * torch.sigmoid(pred_lin)).sum(dim=1)

    def predict(self, batch):
        if isinstance(batch, dict):
            return self.forward(batch)
        else:
            return self.predict_from_dataset(batch)

    def predict_from_dataset(self, prs_dataset):
        # Sanity checks:
        assert "expert_predictions" in prs_dataset.group_getitem_cols
        assert "gate_input" in prs_dataset.group_getitem_cols
        assert (
            self.group_getitem_cols["expert_predictions"]
            == prs_dataset.group_getitem_cols["expert_predictions"]
        )
        assert (
            self.group_getitem_cols["gate_input"]
            == prs_dataset.group_getitem_cols["gate_input"]
        )

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.forward(next(iter(dat))).detach().numpy()

    def predict_proba(self, batch):
        if isinstance(batch, dict):
            with torch.no_grad():
                return self.gate_forward(batch).detach().cpu().numpy()
            # return self.gate_model(batch)
        else:
            return self.predict_proba_from_dataset(batch)

    def predict_proba_from_dataset(self, prs_dataset):
        assert "gate_input" in prs_dataset.group_getitem_cols
        assert (
            self.group_getitem_cols["gate_input"]
            == prs_dataset.group_getitem_cols["gate_input"]
        )

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.gate_forward(next(iter(dat))).detach().numpy()

    def export_config(self):
        return {
            "loss": self.loss,
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
            "family": self.family,
            "gate_model_spec": self.gate_model_spec,
            "gate_penalty": self.gate_penalty,
            "expert_penalty": self.expert_penalty,
            "topk_k": getattr(self, "topk_k", None),
            "temperature_schedule_params": self.temperature_schedule_params,
            "hard_ste": getattr(self, "hard_ste", True),
            "lb_coef": getattr(self, "lb_coef", 0.0),
            "eps": getattr(self, "eps", 1e-12),
            "entropy_schedule_params": self.entropy_schedule_params,
            "binomial_mixing_level": self.binomial_mixing_level,
            "gate_add_intercept": bool(self.gate_add_intercept),
            "expert_add_intercept": bool(getattr(self, "expert_add_intercept", False)),
        }

    def get_gate_parameters(self, return_dataframe=False):

        if not hasattr(self, "gate_model") or self.gate_model is None:
            return None

        cov_names = list(self.group_getitem_cols.get("gate_input", []))
        expert_names = list(self.group_getitem_cols.get("expert_predictions", []))

        K = len(expert_names)

        if self.gate_add_intercept:
            cov_names = ["Intercept"] + cov_names

        layers = list(self.gate_model.gate)
        linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        if len(linear_layers) != 1:
            return None

        linear = linear_layers[0]
        if linear.in_features != len(cov_names) or linear.out_features != K:
            return None

        W = linear.weight.detach().cpu().numpy().T

        if return_dataframe:
            return pd.DataFrame(W, index=cov_names, columns=expert_names)
        else:
            return W

    def get_expert_scaler_parameters(self, return_dataframe=False):
        if not hasattr(self, "expert_scaler") or self.expert_scaler is None:
            return None

        scale = self.expert_scaler.scale.detach().cpu().numpy()
        covar_coef = (
            None
            if self.expert_scaler.covar_coef is None
            else self.expert_scaler.covar_coef.detach().cpu().numpy()
        )
        intercept = (
            None
            if self.expert_scaler.intercept is None
            else self.expert_scaler.intercept.detach().cpu().numpy()
        )

        if not return_dataframe:
            return {
                "scale": scale,
                "covariate_coefficients": covar_coef,
                "intercept": intercept,
            }

        expert_names = list(self.group_getitem_cols.get("expert_predictions", []))
        cov_names = list(self.group_getitem_cols.get("expert_covariates", []))

        rows = []
        if intercept is not None:
            rows.append(
                pd.DataFrame(intercept.reshape(1, -1), index=["Intercept"], columns=expert_names)
            )

        rows.append(pd.DataFrame(scale.reshape(1, -1), index=["PRS"], columns=expert_names))

        if covar_coef is not None:
            rows.append(pd.DataFrame(covar_coef.T, index=cov_names, columns=expert_names))

        return pd.concat(rows, axis=0)

    def get_global_parameters(self, return_dataframe=False):
        if not hasattr(self, "global_head") or self.global_head is None:
            return None

        weight = self.global_head.weight.detach().cpu().numpy().reshape(-1)
        bias = (
            None
            if self.global_head.bias is None
            else self.global_head.bias.detach().cpu().numpy()
        )

        if not return_dataframe:
            return {"weight": weight, "bias": bias}

        cov_names = list(self.group_getitem_cols.get("global_covariates", []))
        values = weight
        idx = cov_names

        if bias is not None:
            values = np.concatenate([bias.reshape(1), weight])
            idx = ["Intercept"] + cov_names

        return pd.DataFrame(values.reshape(-1, 1), index=idx, columns=["Coefficient"])

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer in ("LBFGS", "L-BFGS", "L-BFGS-B"):
            optimizer = torch.optim.LBFGS(self.parameters())
        else:
            raise KeyError(f"Optimizer {self.optimizer} is not recognized.")

        return optimizer


##################################################################
# Define the user-interface for the MoEPRS model implemented in PyTorch:


class TorchMoEPRS:
    """
    MoEPRS-like wrapper around the PyTorch-Lightning MoE implementation.
    """

    default_loader_params = {
        "batch_size": 4096,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "device": None,
    }

    @classmethod
    def default_training_kwargs(cls, seed=8):
        return Lit_MoEPRS.default_training_kwargs(seed=seed)

    def __init__(
        self,
        prs_dataset=None,
        expert_cols=None,
        gate_input_cols=None,
        expert_covariates_cols=None,
        global_covariates_cols=None,
        gate_add_intercept=True,
        expert_add_intercept=False,
        standardize_data=True,
        loss="nll",
        gate_model_spec=None,
        optimizer="Adam",
        learning_rate=1e-3,
        gate_penalty=0.0,
        expert_penalty=0.0,
        entropy_penalty=0.0,
        entropy_schedule_params=None,
        topk_k=None,
        temperature_schedule_params=None,
        hard_ste=True,
        lb_coef=0.0,
        specialization_weight=0.8,
        binomial_mixing_level="probability",
        device="cpu",
    ):
        # Persistent model/data-definition state:
        self.prs_dataset = prs_dataset
        self.expert_cols = _as_list(expert_cols)
        self.gate_input_cols = _as_list(gate_input_cols)
        self.expert_covariates_cols = _as_list(expert_covariates_cols)
        self.global_covariates_cols = _as_list(global_covariates_cols)
        self.standardize_data = bool(standardize_data)
        self.loss = loss

        # Persistent architecture/semantics:
        self.gate_model_spec = gate_model_spec or {}
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.gate_penalty = float(gate_penalty)
        self.expert_penalty = float(expert_penalty)
        self.entropy_penalty = float(entropy_penalty)
        self.entropy_schedule_params = (
            None if entropy_schedule_params is None else dict(entropy_schedule_params)
        )
        self.topk_k = topk_k
        self.temperature_schedule_params = (
            None
            if temperature_schedule_params is None
            else dict(temperature_schedule_params)
        )
        self.hard_ste = bool(hard_ste)
        self.lb_coef = float(lb_coef)
        self.specialization_weight = specialization_weight
        self.gate_add_intercept = bool(gate_add_intercept)
        self.expert_add_intercept = bool(expert_add_intercept)
        self.binomial_mixing_level = str(binomial_mixing_level)

        # Default compute device (can be overridden per call):
        self.device = torch.device(device)
        self.loader_params = dict(self.default_loader_params)

        # Fitted artifacts:
        self.lit_model = None
        self.trainer = None
        self.history = None
        self.data_scaler = None
        self.group_getitem_cols = None
        self.model_dir = None

        if self.prs_dataset is not None:
            self.group_getitem_cols = self._build_group_getitem(self.prs_dataset)
            self.prs_dataset.set_group_getitem_cols(self.group_getitem_cols)

    @property
    def N(self):
        if self.prs_dataset is None:
            return None
        return self.prs_dataset.N

    @property
    def K(self):
        return 0 if self.expert_cols is None else len(self.expert_cols)

    def _build_group_getitem(self, dataset):
        if dataset is None:
            raise ValueError("dataset is required to build group_getitem_cols.")

        missing = []
        if self.expert_cols is None:
            missing.append("expert_cols")
        if self.global_covariates_cols is None:
            missing.append("global_covariates_cols")
        if missing:
            raise ValueError(
                "Missing required column groups: "
                + ", ".join(missing)
                + ". Define them in TorchMoEPRS.__init__."
            )
        gate_input_cols = list(self.gate_input_cols or [])
        if len(gate_input_cols) == 0 and not self.gate_add_intercept:
            raise ValueError(
                "No covariates provided for gating model. Set `gate_add_intercept=True` or provide `gate_input_cols`."
            )

        group_getitem_cols = {
            "phenotype": [dataset.phenotype_col],
            "expert_predictions": list(self.expert_cols),
            "gate_input": gate_input_cols,
            "global_covariates": list(self.global_covariates_cols),
        }
        if self.expert_covariates_cols is not None:
            group_getitem_cols["expert_covariates"] = list(self.expert_covariates_cols)
        return group_getitem_cols

    def _make_lit_model(self, family):
        gate_model_spec = {"final_activation": "softmax"}
        gate_model_spec.update(self.gate_model_spec)

        entropy_schedule_params = (
            dict(self.entropy_schedule_params)
            if self.entropy_schedule_params is not None
            else {
                "coef_start": float(self.entropy_penalty),
                "coef_end": float(self.entropy_penalty),
                "warm_epochs": 0,
                "decay_epochs": 0,
            }
        )
        temperature_schedule_params = (
            dict(self.temperature_schedule_params)
            if self.temperature_schedule_params is not None
            else None
        )

        return Lit_MoEPRS(
            group_getitem_cols=self.group_getitem_cols,
            gate_model_spec=gate_model_spec,
            loss=self.loss,
            optimizer=self.optimizer,
            family=family,
            learning_rate=self.learning_rate,
            gate_penalty=self.gate_penalty,
            entropy_penalty=self.entropy_penalty,
            expert_penalty=self.expert_penalty,
            topk_k=self.topk_k,
            entropy_schedule_params=entropy_schedule_params,
            temperature_schedule_params=temperature_schedule_params,
            hard_ste=self.hard_ste,
            lb_coef=self.lb_coef,
            specialization_weight=self.specialization_weight,
            binomial_mixing_level=self.binomial_mixing_level,
            gate_add_intercept=self.gate_add_intercept,
            expert_add_intercept=self.expert_add_intercept,
        )

    def fit(
        self,
        min_epochs=50,
        max_epochs=500,
        prop_validation=0.1,
        min_validation=1000,
        batch_size=4096,
        weigh_samples=False,
        seed=8,
        ancestry_balance_lambda=0.3,
    ):
        """
        Train the wrapped PyTorch MoE model.
        """
        if self.prs_dataset is None:
            raise ValueError("No dataset provided. Pass prs_dataset to constructor.")

        d = self.prs_dataset
        d.set_backend("torch")

        self.lit_model = self._make_lit_model(family=d.phenotype_likelihood).to(
            self.device
        )

        run_seed = int(seed)
        make_deterministic(run_seed)

        trainer, lit = train_lit_model(
            self.lit_model,
            d,
            min_epochs=int(min_epochs),
            max_epochs=int(max_epochs),
            prop_validation=float(prop_validation),
            min_validation=int(min_validation),
            batch_size=batch_size,
            weigh_samples=bool(weigh_samples),
            seed=run_seed,
            ancestry_balance_lambda=ancestry_balance_lambda,
            standardize_data=self.standardize_data,
        )

        self.trainer = trainer
        self.lit_model = lit.to(self.device).eval()
        self.data_scaler = copy.deepcopy(
            getattr(self.lit_model, "training_scaler", None)
        )
        self.history = None

        return self

    def _check_fitted(self):
        if self.lit_model is None:
            raise ValueError("Model has not been fitted yet. Call `.fit()` first.")

    def _prepare_dataset_for_inference(self, prs_dataset):
        self._check_fitted()
        # Copy only when we need to apply scaler-based transforms.
        d = copy.deepcopy(prs_dataset) if self.standardize_data else prs_dataset

        if self.standardize_data:
            if self.data_scaler is not None:
                d.standardize_data(scaler=self.data_scaler, refit=False)

        expected = self.group_getitem_cols or self.lit_model.group_getitem_cols
        d.set_group_getitem_cols(expected)
        d.set_backend("torch")
        d._data_matrix = None
        d.cache_data_matrix()

        return d

    def _batch_loader(self, d, batch_size=None):
        subset = IndexSubset(d, np.arange(d.N))
        cfg = self.loader_params
        bs = min(int(cfg["batch_size"] if batch_size is None else batch_size), d.N)
        n_workers = int(cfg["num_workers"])

        kwargs = dict(
            batch_size=bs,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=d.get_batch,
            pin_memory=bool(cfg["pin_memory"]),
            persistent_workers=bool(cfg["persistent_workers"]) and (n_workers > 0),
        )
        if n_workers > 0:
            kwargs["prefetch_factor"] = int(cfg["prefetch_factor"])

        return DataLoader(subset, **kwargs)

    def set_loader_params(self, **kwargs):
        self.loader_params.update(kwargs)

    def _send_batch_to_device(self, batch, device=None):
        dev = self._resolve_runtime_device(device)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(dev, non_blocking=True)
        return batch

    def _resolve_runtime_device(self, device=None):
        dev = self.device if device is None else torch.device(device)
        self.device = dev
        if self.lit_model is not None:
            self.lit_model = self.lit_model.to(dev).eval()
        return dev

    def _expert_linear_predictor(
        self,
        prs_dataset=None,
        device=None,
    ):
        if prs_dataset is None:
            if self.prs_dataset is None:
                raise ValueError("No dataset provided.")
            prs_dataset = self.prs_dataset
        d = self._prepare_dataset_for_inference(prs_dataset)
        outs = []
        dev = self._resolve_runtime_device(device)
        with torch.inference_mode():
            for batch in self._batch_loader(d):
                batch = self._send_batch_to_device(batch, device=dev)
                outs.append(
                    self.lit_model._expert_linear_predictor(batch)
                    .detach()
                    .cpu()
                    .numpy()
                )
        return np.vstack(outs)

    def _global_linear_predictor(
        self,
        prs_dataset=None,
        device=None,
    ):
        if prs_dataset is None:
            if self.prs_dataset is None:
                raise ValueError("No dataset provided.")
            prs_dataset = self.prs_dataset
        d = self._prepare_dataset_for_inference(prs_dataset)
        outs = []
        dev = self._resolve_runtime_device(device)
        with torch.inference_mode():
            for batch in self._batch_loader(d):
                batch = self._send_batch_to_device(batch, device=dev)
                g = self.lit_model._global_linear_predictor(batch)
                if g is None:
                    g = torch.zeros(batch["expert_predictions"].shape[0], device=dev)
                outs.append(g.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def _combined_linear_predictor(
        self,
        prs_dataset=None,
        device=None,
    ):
        if prs_dataset is None:
            if self.prs_dataset is None:
                raise ValueError("No dataset provided.")
            prs_dataset = self.prs_dataset
        d = self._prepare_dataset_for_inference(prs_dataset)
        outs = []
        dev = self._resolve_runtime_device(device)
        with torch.inference_mode():
            for batch in self._batch_loader(d):
                batch = self._send_batch_to_device(batch, device=dev)
                pred = self.lit_model._combined_linear_predictor(batch)
                if self.lit_model.family == "binomial":
                    pred = torch.sigmoid(pred)
                outs.append(pred.detach().cpu().numpy())
        return np.vstack(outs)

    def predict(
        self,
        prs_dataset=None,
        device=None,
    ):
        if prs_dataset is None:
            if self.prs_dataset is None:
                raise ValueError("No dataset provided.")
            prs_dataset = self.prs_dataset

        d = self._prepare_dataset_for_inference(prs_dataset)

        outs = []
        dev = self._resolve_runtime_device(device)
        with torch.inference_mode():
            for batch in self._batch_loader(d):
                batch = self._send_batch_to_device(batch, device=dev)
                yhat = self.lit_model.forward(batch)
                outs.append(yhat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    def predict_prs(
        self,
        prs_dataset=None,
        device=None,
    ):
        """
        Prediction excluding global covariate head contribution.
        """
        if prs_dataset is None:
            if self.prs_dataset is None:
                raise ValueError("No dataset provided.")
            prs_dataset = self.prs_dataset

        d = self._prepare_dataset_for_inference(prs_dataset)

        outs = []
        dev = self._resolve_runtime_device(device)
        with torch.inference_mode():
            for batch in self._batch_loader(d):
                batch = self._send_batch_to_device(batch, device=dev)
                w = self.lit_model.gate_forward(batch)
                scaled = self.lit_model._expert_linear_predictor(batch)

                if self.lit_model.family == "gaussian":
                    yhat = (w * scaled).sum(dim=1)
                else:
                    if self.lit_model.binomial_mixing_level == "logit":
                        yhat = torch.sigmoid((w * scaled).sum(dim=1))
                    else:
                        yhat = (w * torch.sigmoid(scaled)).sum(dim=1)

                outs.append(yhat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    def predict_proba(
        self,
        prs_dataset=None,
        log=False,
        batch_size=None,
        device=None,
    ):
        if prs_dataset is None:
            if self.prs_dataset is None:
                raise ValueError("No dataset provided.")
            prs_dataset = self.prs_dataset

        d = self._prepare_dataset_for_inference(prs_dataset)
        outs = []
        dev = self._resolve_runtime_device(device)
        with torch.inference_mode():
            for batch in self._batch_loader(d, batch_size=batch_size):
                batch = self._send_batch_to_device(batch, device=dev)
                p = self.lit_model.gate_forward(batch)
                if log:
                    p = torch.log(p.clamp_min(1e-12))
                outs.append(p.detach().cpu().numpy())

        return np.vstack(outs)

    def get_model_parameters(self, return_dataframe=False):
        self._check_fitted()

        params = dict()

        params["gate_params"] = self.lit_model.get_gate_parameters(
            return_dataframe=return_dataframe
        )
        params["expert_scaler_params"] = self.lit_model.get_expert_scaler_parameters(
            return_dataframe=return_dataframe
        )
        params["global_params"] = self.lit_model.get_global_parameters(
            return_dataframe=return_dataframe
        )

        return params

    def save(self, output_file):
        self._check_fitted()

        output_dir = osp.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        config = self.lit_model.export_config()

        wrapper_meta = {
            "expert_cols": self.expert_cols,
            "gate_input_cols": self.gate_input_cols,
            "gate_add_intercept": self.gate_add_intercept,
            "expert_covariates_cols": self.expert_covariates_cols,
            "global_covariates_cols": self.global_covariates_cols,
            "standardize_data": self.standardize_data,
            "expert_add_intercept": self.expert_add_intercept,
            "group_getitem_cols": self.group_getitem_cols
            or self.lit_model.group_getitem_cols,
        }

        checkpoint = {
            "state_dict": self.lit_model.state_dict(),
            "config": config,
            "wrapper_meta": wrapper_meta,
            "history": self.history,
        }

        scaler = self.data_scaler or getattr(self.lit_model, "training_scaler", None)
        if scaler is not None:
            checkpoint["scaler_data"] = serialize_standard_scaler(scaler)

        with open(output_file, "wb") as outf:
            pickle.dump(checkpoint, outf)

    @classmethod
    def from_saved_model(
        cls,
        param_file,
        map_location="cpu",
        strict=True,
        device=None,
    ):
        try:
            with open(param_file, "rb") as pf:
                checkpoint = pickle.load(pf)
        except Exception:
            checkpoint = torch.load(
                param_file, map_location=map_location, weights_only=False
            )
        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError(f"Malformed checkpoint: {param_file}")

        cfg = checkpoint["config"]
        state = checkpoint["state_dict"]
        wrapper_meta = checkpoint.get("wrapper_meta", {})

        group_getitem_cols = wrapper_meta.get("group_getitem_cols", None)
        if group_getitem_cols is None:
            raise ValueError(
                "Checkpoint is missing `wrapper_meta.group_getitem_cols`; cannot reconstruct model."
            )

        # Normalize wrapper-level column lists from checkpoint metadata.
        expert_cols = wrapper_meta.get(
            "expert_cols", group_getitem_cols.get("expert_predictions")
        )
        gate_input_cols = wrapper_meta.get("gate_input_cols", None)
        if gate_input_cols is None:
            gate_input_cols = group_getitem_cols.get("gate_input")
        expert_covariates_cols = wrapper_meta.get(
            "expert_covariates_cols", group_getitem_cols.get("expert_covariates")
        )
        global_covariates_cols = wrapper_meta.get(
            "global_covariates_cols", group_getitem_cols.get("global_covariates")
        )

        model = cls(
            expert_cols=expert_cols,
            gate_input_cols=gate_input_cols,
            expert_covariates_cols=expert_covariates_cols,
            global_covariates_cols=global_covariates_cols,
            standardize_data=wrapper_meta.get("standardize_data", True),
            loss=cfg.get("loss", "nll"),
            gate_model_spec=cfg.get("gate_model_spec", None),
            optimizer=cfg.get("optimizer", "Adam"),
            learning_rate=cfg.get("learning_rate", 1e-3),
            gate_penalty=cfg.get("gate_penalty", 0.0),
            expert_penalty=cfg.get("expert_penalty", 0.0),
            entropy_penalty=float(
                cfg.get("entropy_schedule_params", {}).get("coef_start", 0.0)
            ),
            entropy_schedule_params=cfg.get("entropy_schedule_params", None),
            topk_k=cfg.get("topk_k", None),
            temperature_schedule_params=cfg.get("temperature_schedule_params", None),
            hard_ste=cfg.get("hard_ste", True),
            lb_coef=cfg.get("lb_coef", 0.0),
            gate_add_intercept=wrapper_meta.get(
                "gate_add_intercept", cfg.get("gate_add_intercept", True)
            ),
            expert_add_intercept=wrapper_meta.get(
                "expert_add_intercept", cfg.get("expert_add_intercept", False)
            ),
            binomial_mixing_level=cfg.get("binomial_mixing_level", "probability"),
            device=(device or map_location),
        )

        # No dataset is needed for reconstruction; use persisted batch schema.
        model.group_getitem_cols = group_getitem_cols
        lit = model._make_lit_model(family=cfg.get("family", "gaussian"))
        lit.load_state_dict(state, strict=strict)
        model.lit_model = lit.to(model.device).eval()
        model.model_dir = osp.dirname(param_file)
        model.history = checkpoint.get("history", None)

        if "scaler_data" in checkpoint:
            model.data_scaler = deserialize_standard_scaler(checkpoint["scaler_data"])
        elif "scaler" in checkpoint:
            model.data_scaler = copy.deepcopy(checkpoint["scaler"])

        return model
