import copy
import json
import os
import os.path as osp
import pickle
from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl

try:
    from sparsemax import Sparsemax
except ImportError:
    pass

try:
    import torchsort
except ImportError:
    torchsort = None

import torch.nn.functional as F

def configure_cpu_threads(cpus_per_task: int, num_workers: int, interop_threads: int = 1):
    """
    Give the main training process most of the CPU threads, leave some CPU capacity
    for DataLoader workers.
    """
    compute_threads = max(1, cpus_per_task - num_workers)

    # PyTorch CPU threading (main process)
    torch.set_num_threads(compute_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    # BLAS/threaded libs (main process)
    os.environ["OMP_NUM_THREADS"] = str(compute_threads)
    os.environ["MKL_NUM_THREADS"] = str(compute_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(compute_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(compute_threads)

    return compute_threads

def dataloader_worker_init_fn(worker_id: int):
    """
    Prevent each DataLoader worker from spawning its own CPU threadpool.
    This avoids massive oversubscription when num_workers > 0.
    """
    import os
    import torch

    torch.set_num_threads(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

class ConvergenceCheck(pl.callbacks.Callback):

    def __init__(self, rtol=1e-05, atol=1e-05):

        super().__init__()

        self.rtol = rtol
        self.atol = atol

        self.best_loss = np.inf
        self.best_params = None

    def on_train_start(self, trainer, pl_module):

        self.best_loss = np.inf
        self.best_params = None

    def on_train_end(self, trainer, pl_module):

        current_loss_t = trainer.callback_metrics['train_loss']
        current_loss = float(current_loss_t.detach().cpu().item()) if isinstance(current_loss_t, torch.Tensor) else float(current_loss_t)
        
        current_params = [p.detach().numpy().copy() for p in pl_module.parameters()]

        if self.best_params is not None:
            if np.allclose(current_loss, self.best_loss, rtol=self.rtol, atol=self.atol):
                print("> Convergence achieved (negligible change in objective)")
                trainer.should_stop = True
            elif all([np.allclose(p1, p2, rtol=self.rtol, atol=self.atol)
                      for p1, p2 in zip(current_params, self.best_params)]):
                print("> Convergence achieved (negligible change in parameters)")
                trainer.should_stop = True

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_params = current_params
        else:
            self.best_loss = current_loss
            self.best_params = current_params

class DelayedEarlyStopping(pl.callbacks.Callback):
    def __init__(self, monitor="val_loss", min_epoch=100, patience=10, mode="min", min_delta=0.0):
        super().__init__()
        self.monitor = monitor
        self.min_epoch = int(min_epoch)
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None
        self.num_bad = 0

    def on_fit_start(self, trainer, pl_module):
        self.best = None
        self.num_bad = 0

    def on_validation_end(self, trainer, pl_module):
        # trainer.current_epoch is 0-indexed; compare using 1-indexed epoch count
        epoch = int(trainer.current_epoch) + 1
        if epoch < self.min_epoch:
            return

        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor, None)
        if current is None:
            return

        if isinstance(current, torch.Tensor):
            current = float(current.detach().cpu().item())
        else:
            current = float(current)

        if self.best is None:
            self.best = current
            self.num_bad = 0
            return

        if self.mode == "min":
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)

        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                trainer.should_stop = True


class IndexSubset(Dataset):
    """
    Like torch.utils.data.Subset, but returns the ORIGINAL row index (int),
    so collate_fn can fetch the whole batch via PRSDataset.get_batch().
    Keeps .dataset and .indices so your sampler helpers still work.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])

#########################################################


def likelihood_mixture_loss(expert_weights, expert_predictions, phenotype, family="gaussian"):

    assert family in ("gaussian", "binomial")

    N = expert_weights.shape[0]

    if family == "gaussian":
        losses = (expert_predictions - phenotype)**2
    else:
        expert_predictions = torch.clamp(expert_predictions, 1e-6, 1.-1e-6)
        losses = -(phenotype*torch.log(expert_predictions) + (1.-phenotype)*torch.log(1.-expert_predictions))

    return (1./N)*(expert_weights*losses).sum()


def likelihood_mixture_loss2(expert_weights, expert_predictions, phenotype, family="gaussian"):
    """
    An alternative loss that matches the likelihood mixture loss outlined in Equation (1.3)
    in Jacobs et al. 1991.
    """

    assert family in ("gaussian", "binomial")

    if family == "gaussian":
        lik = torch.exp(-.5*(expert_predictions - phenotype)**2)
    else:
        expert_predictions = torch.clamp(expert_predictions, 1e-6, 1.-1e-6)
        lik = torch.exp(phenotype*torch.log(expert_predictions) + (1.-phenotype)*torch.log(1.-expert_predictions))

    return -torch.log((expert_weights*lik).sum(axis=1)).mean()


def ensemble_mixture_loss(expert_weights, expert_predictions, phenotype, family="gaussian"):

    assert family in ("gaussian", "binomial")

    pred = (expert_weights*expert_predictions).sum(axis=1)

    if family == "gaussian":
        return ((pred - phenotype)**2).mean()
    else:
        pred = torch.clamp(pred, 1e-6, 1.-1e-6)
        return -(phenotype*torch.log(pred) + (1.-phenotype)*torch.log(1.-pred)).mean()


def ensemble_mixture_loss_simple(phenotype, pred, family="gaussian"):

    assert family in ("gaussian", "binomial")

    if family == "gaussian":
        return ((pred - phenotype)**2).mean()
    else:
        pred = torch.clamp(pred, 1e-6, 1.-1e-6)
        return -(phenotype*torch.log(pred) + (1.-phenotype)*torch.log(1.-pred)).mean()

# Negative log-likelihood for Gaussian MoE with per-expert variances
def gaussian_moe_nll(expert_weights, expert_predictions, phenotype, sigma2, eps=1e-12):
    y = phenotype.view(-1, 1)
    sigma2 = sigma2.view(1, -1).clamp_min(eps)                      # (1,K)
    resid2 = (expert_predictions - y) ** 2                          # (N,K)

    loglik = -0.5 * (torch.log(2.0 * np.pi * sigma2) + resid2 / sigma2)  # (N,K)
    logw   = torch.log(expert_weights.clamp_min(eps))               # (N,K)

    return -(torch.logsumexp(logw + loglik, dim=1)).mean()

#########################################################
# Define a PyTorch Lightning module to streamline training
class Lit_MoEPRS(pl.LightningModule):
    save_ext = ".pt"

    @classmethod
    def default_training_kwargs(cls, seed=8):
        return {
            "loss": "likelihood_mixture2",
            "fix_sigma2": False,
            "optimizer": "Adam",
            "gate_model_layers": None,
            "gate_add_batch_norm": False,
            "gate_add_layer_norm": True,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 500,
            "batch_size": 2048,
            "seed": seed,
            "topk_k": None,
            "tau_start": 2.0,
            "tau_end": 1.0,
            "tau_warm_epochs": 10,
            "tau_decay_epochs": 90,
            "hard_ste": False,
            "lb_coef": 0.0,
            "ent_coef": 0.5,
            "ent_coef_end": 0.0,
            "ent_warm_epochs": 10,
            "ent_decay_epochs": 90,
            "ancestry_balance_lambda": None,
            "use_per_expert_bias": False,
            "add_covariates_to_experts": False,
            "use_global_head": True,
            "global_head_bias": True,
            "binomial_logit_level": True,
        }

    def __init__(self,
                 group_getitem_cols,
                 gate_model_layers=None,
                 gate_add_batch_norm=True,
                 gate_add_layer_norm=False,
                 loss="likelihood_mixture",
                 optimizer="Adam",
                 family="gaussian",
                 learning_rate=1e-3,
                 weight_decay=0.,
                 l2_pen = 0,
                 ent_coef = 0,
                 ent_coef_end=None,       
                 ent_warm_epochs=0,       
                 ent_decay_epochs=0,      
                 topk_k=None,        # None = disable top-k
                 tau_start=1.0,      # 1.0 = no temperature
                 tau_end=1.0,        # 1.0 = no temperature
                 tau_warm_epochs=0,
                 tau_decay_epochs=0,
                 hard_ste=True,      # straight-through estimator
                 lb_coef=0.0,        # 0.0 = disable load-balancing aux loss
                 eps=1e-12,

                 use_per_expert_bias = True,
                 use_global_head = True,
                 global_head_bias=True,
                 binomial_logit_level = False,
                 center_expert_covariates: bool = True):
        """
        A PyTorch Lightning module for training a mixture of experts model.

        :param group_getitem_cols: A dictionary mapping categories of data to the relevant keys from the
         pandas dataframe. This is useful for iterative data fetching (e.g. data loaders).
            These are used to define what columns/groups of columns are fetched in the __getitem__ method.
        :param gate_model_layers: A list of integers specifying the number of hidden units
        in the gating model.
        :param gate_add_batch_norm: If True, add batch normalization to the gating model.
        :param loss: The loss function to use. Options are: ('likelihood_mixture', 'ensemble_mixture')
        :param optimizer: The optimizer to use. Options are: ('Adam', 'LBFGS', 'SGD')
        :param family: The family of the likelihood. Options are: ('gaussian', 'binomial')
        :param learning_rate: The learning rate for the optimizer.
        :param weight_decay: The weight decay for the optimizer.
        """

        super().__init__()

        self.center_expert_covariates = bool(center_expert_covariates)
        self.training_scaler = None

        # -------------------------------------------------------
        # Sanity checks for the inputs:
        assert loss in ("likelihood_mixture",
                        "likelihood_mixture2",
                        "ensemble_mixture",
                        "likelihood_mixture_sigma",
                        "logit_level_bce")
        assert optimizer in ("Adam", "LBFGS", "SGD")
        assert family in ("gaussian", "binomial")

        assert 'phenotype' in group_getitem_cols
        assert 'gate_input' in group_getitem_cols
        assert 'experts' in group_getitem_cols

        # -------------------------------------------------------
        # Define / initialize the model components:

        self.group_getitem_cols = group_getitem_cols

        self.gate_model_layers = gate_model_layers
        self.gate_add_batch_norm = gate_add_batch_norm
        self.gate_add_layer_norm = gate_add_layer_norm

        assert not (self.gate_add_batch_norm and self.gate_add_layer_norm), \
            "Choose either BatchNorm or LayerNorm for the gate, not both."

        self.gate_model = GateModel(self.gate_input_dim,
                                    self.n_experts,
                                    hidden_layers=self.gate_model_layers,
                                    add_batch_norm=self.gate_add_batch_norm,
                                    add_layer_norm=self.gate_add_layer_norm,
                                    final_activation="softmax")  # The gating model

        # Expert linear scalers (gamma_k)
        # If per-expert covariates are provided, allow coefficients and use them alongside an intercept. (gamma_k*S_k + covariates * covar_coefficients + intercept)
        if self.n_expert_covariates > 0:
            self.expert_scaler = nn.ModuleList([
                LinearScaler(self.n_expert_covariates, family=family, bias=False)
                for _ in range(self.n_experts)
            ])
        # Otherwise, expert linear scaler only (gamma_k*S_k)
        else:
            self.expert_scaler = nn.ModuleList([
                LinearScaler(family=family, bias=False)
                for _ in range(self.n_experts)
            ])
        

        '''
        expert_intercept_k = kappa · b_k, where kappa is global scale (shared across experts)

        "group shrinkage" kappa acts like an on/off knob for the whole bias block, makes the model more robust by
        keeping the model parsimonious unless intercept differences are clearly supported by the data.
        '''
        self.use_per_expert_bias = use_per_expert_bias

        if self.use_per_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.n_experts))   # b_k
            self.expert_bias_log_scale = nn.Parameter(torch.tensor(-5.0))  # kappa ~ softplus(-5)

            # Floors / priors, mild L2 prior on kappa (scale) and bias themselves
            self.expert_bias_scale_floor = 0.0
            self.expert_bias_scale_prior = 1e-4
            self.expert_bias_l2 = 1e-4
        else:
            # No expert-specific intercepts
            self.expert_bias = None
            self.expert_bias_log_scale = None
            self.expert_bias_scale_floor = 0.0
            self.expert_bias_scale_prior = 0.0
            self.expert_bias_l2 = 0.0

        self.loss = loss
        self.metrics = {
            'likelihood_mixture': partial(likelihood_mixture_loss, family=family),
            'ensemble_mixture': partial(ensemble_mixture_loss, family=family),
            'likelihood_mixture2': partial(likelihood_mixture_loss2, family=family)
        }

        # Optimizer options:
        self.family = family

        self.binomial_logit_level = bool(binomial_logit_level)

        # If user asks for logit-level binomial, force the right loss key.
        # (Otherwise they'd accidentally optimize the wrong objective.)
        if self.family == "binomial" and self.binomial_logit_level:
            self.loss = "logit_level_bce"
        else:
            self.loss = loss

        self.optimizer = optimizer
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.l2_pen = l2_pen
        self.ent_coef = ent_coef

        # allow entropy coefficient scheduling:
        self.ent_coef_start = float(ent_coef)
        self.ent_coef_end = float(ent_coef) if ent_coef_end is None else float(ent_coef_end)
        self.ent_warm_epochs = int(ent_warm_epochs)
        self.ent_decay_epochs = int(ent_decay_epochs)

        # global covariate head + intercept
        self.use_global_head = use_global_head
        self.global_head_bias = global_head_bias

        if self.use_global_head:
            self.global_in_dim = len(group_getitem_cols.get('global_input',
                                                            group_getitem_cols['gate_input']))
            self.global_head = nn.Linear(self.global_in_dim, 1, bias=self.global_head_bias)
        else:
            self.global_in_dim = 0
            self.global_head = None

        self.topk_k   = topk_k  #top_k routing 

        #temperature scheduling
        self.tau_start = float(tau_start)
        self.tau_end   = float(tau_end)
        self.tau_warm_epochs = int(tau_warm_epochs)
        self.tau_decay_epochs = int(tau_decay_epochs)
        self.hard_ste  = bool(hard_ste)
        self.lb_coef   = float(lb_coef) #load balancing coefficient
        self.eps       = float(eps)

        # for guassian MoE with per-expert residual variance
        if family == "gaussian":
            # estimated residual variance per expert
            self.log_sigma2 = nn.Parameter(torch.zeros(self.n_experts))  # init sigma2=1
            self.min_sigma2 = 1e-2

            self.metrics["likelihood_mixture_sigma"] = (
                lambda w, yhat, y: gaussian_moe_nll(w, yhat, y, self.sigma2, eps=self.eps)
            )

    @property
    def n_experts(self):
        return len(self.group_getitem_cols['experts'])

    @property
    def gate_input_dim(self):
        return len(self.group_getitem_cols['gate_input'])

    @property
    def n_expert_covariates(self):
        if 'expert_covariates' in self.group_getitem_cols:
            return len(self.group_getitem_cols['expert_covariates'])
        else:
            return 0
    
    @property
    def sigma2(self):
        # softplus keeps it positive and smooth; add floor for stability
        return torch.nn.functional.softplus(self.log_sigma2) + self.min_sigma2

    @property
    def expert_bias_scale(self):
        # kappa >= 0, smooth; starts near ~softplus(-5) ~ 0.0067
        if not getattr(self, "use_per_expert_bias", False):
            # scalar zero on correct device
            return torch.zeros((), device=self.device)
        return torch.nn.functional.softplus(self.expert_bias_log_scale) + self.expert_bias_scale_floor

    @property
    def expert_bias_centered(self):
        """
        Enforce identifiability when BOTH:
          - global head has a bias (global intercept)
          - per-expert biases (in the form of ARD) are used

        Then we center expert biases so sum_k b_k = 0.
        """
        if not getattr(self, "use_per_expert_bias", False):
            return None

        b = self.expert_bias 

        # Only center when the model actually has a global intercept to absorb the mean shift
        if self.use_global_head and self.global_head is not None and self.global_head_bias:
            b = b - b.mean()

        return b

    def batch_step(self, batch, batch_idx):
        w = self.gate_forward(batch)  # (N,K)

        # ---- logit-mixing moe model ----
        if self.family == "binomial" and self.binomial_logit_level:
            y = batch["phenotype"].view(-1).float()          # (N,)
            g = self._global_logit(batch)                    # (N,)
            eta_k = self._expert_logits_only(batch)          # (N,K)

            eta = g + (w * eta_k).sum(dim=1)                 # (N,)  <-- your equation
            loss = F.binary_cross_entropy_with_logits(eta, y)

            return {"logit_level_bce": loss}

        # ---- Probability-mixing default moe model ----
        scaled_pred = self.scale_expert_predictions(batch)   # (N,K) probs for binomial, means for gaussian

        losses = {}
        for m, loss_fn in self.metrics.items():
            losses[m] = loss_fn(w, scaled_pred, batch["phenotype"])

            if self.weight_decay > 0. and self.optimizer == "LBFGS":
                losses[m] += self.weight_decay * torch.norm(self.gate_model.gate[0].weight, p=2)
                if self.n_expert_covariates > 0:
                    for expert in self.expert_scaler:
                        losses[m] += self.weight_decay * torch.norm(expert.linear_model.weight, p=2)

        return losses

    def training_step(self, batch, batch_idx):

        losses = self.batch_step(batch, batch_idx)
        total = losses[self.loss]

        # ----- Regularize shared bias scale and bias vector -----
        # Keeps kappa ~ 0 unless the data has evidence that it benefits from additional expert intercepts
        if self.use_per_expert_bias:
            scale_prior = self.expert_bias_scale_prior * (self.expert_bias_log_scale ** 2)
            b = self.expert_bias_centered
            bias_prior  = self.expert_bias_l2 * (b ** 2).mean()

            total = total + scale_prior + bias_prior

            self.log("aux_expert_bias_scale", self.expert_bias_scale.detach(), on_epoch=True)
            self.log("aux_expert_bias_scale_prior", scale_prior.detach(), on_epoch=True)
            self.log("aux_expert_bias_l2", bias_prior.detach(), on_epoch=True)


        if self.family == "gaussian" and hasattr(self, "log_sigma2"):
            # mild L2 prior on log sigma^2 
            sigma_prior = 1e-4 * (self.log_sigma2 ** 2).mean()
            total = total + sigma_prior
            self.log("aux_sigma_prior", sigma_prior, prog_bar=False, on_step=False, on_epoch=True)

        # ---- Load-balance penalty ----
        if self.lb_coef > 0.0:
            p_dense = self.gate_model.forward(batch['gate_input']).clamp_min(self.eps)  # (N,K)
            mean_usage = p_dense.mean(dim=0)                           # (K,)
            target = torch.full_like(mean_usage, 1.0 / self.n_experts)
            lb_loss = ((mean_usage - target) ** 2).mean()
            total = total + self.lb_coef * lb_loss
            self.log("aux_load_balance", lb_loss, prog_bar=False, on_step=False, on_epoch=True)
        
        # entropy reguarlization
        ent_coef = float(self._current_ent_coef())
        if ent_coef > 0.0:
            p = self.gate_forward(batch).clamp_min(self.eps)     # after temp/top-k
            entropy = -(p * p.log()).sum(dim=1).mean()
            ent_loss = ent_coef * (-entropy)                     # penalize low entropy
            total = total + ent_loss
            self.log("aux_entropy", ent_loss, prog_bar=False, on_step=False, on_epoch=True)

        self.log("ent_coef", ent_coef, prog_bar=False, on_step=False, on_epoch=True)

        # l2 regularization on gate parameters
        if self.l2_pen > 0.0:
            l2_gate = 0.0
            for p in self.gate_model.parameters():
                l2_gate = l2_gate + (p ** 2).sum()
            total = total + self.l2_pen * l2_gate

        self.log("train_loss", total, prog_bar=True)

        for m, loss in losses.items():
            if m != self.loss:
                self.log(m, loss, prog_bar=True)
        

        return total

    def validation_step(self, batch, batch_idx):
        losses = self.batch_step(batch, batch_idx)
        total = losses[self.loss]

        # Log final val loss
        self.log("val_loss", total, prog_bar=True)

        return total

    def _expert_logits_only(self, batch):
        """
        Returns eta_{ik} (N,K) WITHOUT global covariate term.
        Includes per-expert covariates (if enabled) because they are part of the expert scaler.
        Optionally includes per-expert bias (if enabled).
        """
        expert_covariates = batch.get("expert_covariates", None)

        logits = torch.cat([
            expert_scaler.forward(
                batch["experts"][:, i],
                covar=expert_covariates,
                return_logits=True
            )
            for i, expert_scaler in enumerate(self.expert_scaler)
        ], dim=1)  # (N,K)

        # per expert bias
        if self.use_per_expert_bias:
            kappa = self.expert_bias_scale
            b = self.expert_bias_centered
            logits = logits + (kappa * b).view(1, -1)

        return logits


    def _global_logit(self, batch):
        """
        Returns g_i (N,) = alpha_0 + C_i * alpha, added ONCE outside the mix of logits.
        """
        if self.use_global_head and (self.global_head is not None):
            global_in = batch.get("global_input", batch["gate_input"])
            return self.global_head(global_in).squeeze(-1)  # (N,)
        return torch.zeros(batch["experts"].shape[0], device=self.device)


    def scale_expert_predictions(self, batch):
        expert_covariates = batch.get("expert_covariates", None)

        # binomial
        if self.family == "binomial":
            # Compute per-expert logits (N,K): eta_{ik} from each expert scaler
            # includes per expert covariaties if used
            logits = torch.cat([
                expert_scaler.forward(
                    batch["experts"][:, i],
                    covar=expert_covariates,
                    return_logits=True
                )
                for i, expert_scaler in enumerate(self.expert_scaler)
            ], dim=1)  # (N,K)

            # if per expert bias i used
            if self.use_per_expert_bias:
                kappa = self.expert_bias_scale
                b = self.expert_bias_centered
                logits = logits + (kappa * b).view(1, -1)

            # If working on the mix of logits scale
            if getattr(self, "binomial_logit_level", False):
                # instead of probabilities, return the logits to be mixed
                return logits

            # Otherwise (default): probability-mixing / mixture losses expect per-expert probabilities,
            # and the global head should be added before the sigmoid to constrain the probabilities to be [0,1]
            if self.use_global_head and (self.global_head is not None):
                global_in = batch.get("global_input", batch["gate_input"])
                g = self.global_head(global_in).squeeze(-1)      # (N,)
                logits = logits + g.unsqueeze(1)                 # (N,K)

            # return probabilities for mixing
            return torch.sigmoid(logits)   

        # -------------------------
        # Gaussian
        # -------------------------
        preds = torch.cat([
            expert_scaler.forward(batch["experts"][:, i], covar=expert_covariates)
            for i, expert_scaler in enumerate(self.expert_scaler)
        ], dim=1)  # (N,K)

        if self.use_global_head and (self.global_head is not None):
            global_in = batch.get("global_input", batch["gate_input"])
            g = self.global_head(global_in).squeeze(-1)          # (N,)
            preds = preds + g.unsqueeze(1).expand(-1, preds.size(1))

        if self.use_per_expert_bias:
            kappa = self.expert_bias_scale
            b = self.expert_bias_centered
            preds = preds + (kappa * b).view(1, -1)

        return preds

    def _current_tau(self):
        # if no trainer, default to final value
        try:
            _ = self.trainer
        except Exception:
            return self.tau_end

        e = int(getattr(self, "current_epoch", 0))

        # No schedule => constant
        if self.tau_warm_epochs <= 0 and self.tau_decay_epochs <= 0:
            return self.tau_start

        # Warmup: keep tau_start
        if e < self.tau_warm_epochs:
            return self.tau_start

        # Linear decay
        if e < self.tau_warm_epochs + self.tau_decay_epochs:
            t = (e - self.tau_warm_epochs) / max(1, self.tau_decay_epochs)
            return self.tau_start + (self.tau_end - self.tau_start) * t

        # After schedule
        return self.tau_end

    def _project_expert_covariate_weights_(self):
        """
        Enforce identifiability when BOTH:
          - global covariate head exists (so alpha is present)
          - per-expert covariate slopes exist (beta_k is present)

        Constraint: Center covariates so 1/N * \sum_k beta_k = 0 so the model is identifiable.
        """
        if not getattr(self, "center_expert_covariates", False):
            return
        if self.n_expert_covariates <= 0:
            return
        if not (self.use_global_head and (self.global_head is not None)):
            # If no global head, don't center (would remove mean effect capacity).
            return

        # Each expert scaler has weight shape (1, 1 + n_expert_covariates)
        with torch.no_grad():
            cov_w = torch.stack(
                [m.linear_model.weight[0, 1:].detach() for m in self.expert_scaler],
                dim=0
            )  # (K, p)

            cov_mean = cov_w.mean(dim=0, keepdim=True)  # (1, p)
            cov_centered = cov_w - cov_mean             # (K, p)

            for k, m in enumerate(self.expert_scaler):
                m.linear_model.weight[0, 1:].copy_(cov_centered[k])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # This runs after Lightning has done the optimizer step for the batch.
        self._project_expert_covariate_weights_()

    def _current_ent_coef(self):
        # if no trainer, default to final value
        try:
            _ = self.trainer
        except Exception:
            return self.ent_coef_end

        e = int(getattr(self, "current_epoch", 0))

        # No schedule => constant
        if self.ent_warm_epochs <= 0 and self.ent_decay_epochs <= 0:
            return self.ent_coef_start

        # Warmup
        if e < self.ent_warm_epochs:
            return self.ent_coef_start

        # Linear decay
        if e < self.ent_warm_epochs + self.ent_decay_epochs:
            t = (e - self.ent_warm_epochs) / max(1, self.ent_decay_epochs)
            return self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * t

        # Floor
        return self.ent_coef_end

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
        mask = torch.sigmoid((threshold - ranks) * sharpness)         # (N, K)

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

    def gate_forward(self, batch, return_dense=False):
        # base dense softmax from GateModel: (N,K)
        p_soft = self.gate_model.forward(batch['gate_input'])
        # temperature sharpening
        p_tau = self._apply_temperature(p_soft)
        # optional top-k sparsification
        p_used = self._apply_topk(p_tau)

        if return_dense:
            return p_used, p_soft
        return p_used


    def forward(self, batch):
        w = self.gate_forward(batch)  # (N,K)

        # if working with mixture of logits
        if self.family == "binomial" and self.binomial_logit_level:
            g = self._global_logit(batch)              # (N,)
            eta_k = self._expert_logits_only(batch)    # (N,K)
            eta = g + (w * eta_k).sum(dim=1)           # (N,)
            return torch.sigmoid(eta)                  # (N,)

        # existing behavior
        return (w * self.scale_expert_predictions(batch)).sum(axis=1)

    def predict(self, batch):

        if isinstance(batch, dict):
            return self.forward(batch)
        else:
            return self.predict_from_dataset(batch)

    def predict_from_dataset(self, prs_dataset):

        # Sanity checks:
        assert 'experts' in prs_dataset.group_getitem_cols
        assert 'gate_input' in prs_dataset.group_getitem_cols
        assert self.group_getitem_cols['experts'] == prs_dataset.group_getitem_cols['experts']
        assert self.group_getitem_cols['gate_input'] == prs_dataset.group_getitem_cols['gate_input']

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.forward(next(iter(dat))).detach().numpy()

    def predict_cov_only_from_dataset(self, prs_dataset):
        """
        Covariate-only prediction by zeroing PRS inputs at inference.
        Used for null adjusted model including possible non-linear covariate contributions for
        proper evaluation of incremental-r2

        Keeps:
            - gate_forward(C)
            - per-expert covariate slopes (if add_covariates_to_experts)
            - global_head(C)
            - per-expert bias block (kappa*b)
            - per-expert sigma2 stuff (gaussian) is irrelevant here (prediction only)

        Removes:
            - PRS contribution (gamma_k * S_ik) because experts[:,k] is set to 0
        """
        prs_dataset.set_backend("torch")
        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)
        batch = next(iter(dat))

        batch2 = {}
        for k, v in batch.items():
                batch2[k] = v.clone() if torch.is_tensor(v) else v

        batch2["experts"] = torch.zeros_like(batch2["experts"])

        with torch.no_grad():
                return self.forward(batch2).detach().cpu().numpy()

    def predict_proba(self, batch):

        if isinstance(batch, dict):
            with torch.no_grad():
                return self.gate_forward(batch).detach().cpu().numpy()
            # return self.gate_model(batch)
        else:
            return self.predict_proba_from_dataset(batch)

    def predict_proba_from_dataset(self, prs_dataset):

        assert 'gate_input' in prs_dataset.group_getitem_cols
        assert self.group_getitem_cols['gate_input'] == prs_dataset.group_getitem_cols['gate_input']

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.gate_forward(next(iter(dat))).detach().numpy()

    def export_config(self):
        return {
            "loss": self.loss,
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "gate_model_layers": getattr(self, "gate_model_layers", None),
            "gate_add_batch_norm": self.gate_add_batch_norm,
            "gate_add_layer_norm": getattr(self, "gate_add_layer_norm", False),
            "family": self.family,
            "topk_k": getattr(self, "topk_k", None),
            "tau_start": getattr(self, "tau_start", 1.0),
            "tau_end": getattr(self, "tau_end", 1.0),
            "tau_warm_epochs": getattr(self, "tau_warm_epochs", 0),
            "tau_decay_epochs": getattr(self, "tau_decay_epochs", 0),
            "hard_ste": getattr(self, "hard_ste", True),
            "lb_coef": getattr(self, "lb_coef", 0.0),
            "eps": getattr(self, "eps", 1e-12),
            "ent_coef_start": getattr(
                self, "ent_coef_start", getattr(self, "ent_coef", 0.0)
            ),
            "ent_coef_end": getattr(
                self, "ent_coef_end", getattr(self, "ent_coef", 0.0)
            ),
            "ent_warm_epochs": getattr(self, "ent_warm_epochs", 0),
            "ent_decay_epochs": getattr(self, "ent_decay_epochs", 0),
            "use_per_expert_bias": getattr(self, "use_per_expert_bias", False),
            "use_global_head": getattr(self, "use_global_head", False),
            "global_head_bias": (
                getattr(self, "global_head", None) is not None
                and getattr(self.global_head, "bias", None) is not None
            ),
            "min_sigma2": float(getattr(self, "min_sigma2", 0.0))
            if hasattr(self, "min_sigma2")
            else None,
            "expert_bias_scale_floor": float(
                getattr(self, "expert_bias_scale_floor", 0.0)
            ),
            "has_expert_covariates": (
                "expert_covariates" in getattr(self, "group_getitem_cols", {})
            ),
            "binomial_logit_level": bool(
                getattr(self, "binomial_logit_level", False)
            ),
            "center_expert_covariates": bool(
                getattr(self, "center_expert_covariates", True)
            ),
        }

    def _gate_weights_dataframe(self):
        if not hasattr(self, "gate_model") or self.gate_model is None:
            return None

        cov_names = list(self.group_getitem_cols.get("gate_input", []))
        expert_names = list(self.group_getitem_cols.get("experts", []))
        C = len(cov_names)
        K = len(expert_names)

        linear = None
        for layer in reversed(list(self.gate_model.gate)):
            if isinstance(layer, nn.Linear):
                linear = layer
                break

        if linear is None or linear.in_features != C or linear.out_features != K:
            return None

        import pandas as pd

        W = linear.weight.detach().cpu().numpy()
        b = linear.bias.detach().cpu().numpy()

        rows = []
        for k, e_name in enumerate(expert_names):
            for j, c_name in enumerate(cov_names):
                rows.append(
                    {
                        "expert_idx": k,
                        "expert": e_name,
                        "covariate_idx": j,
                        "covariate": c_name,
                        "weight": float(W[k, j]),
                    }
                )
            rows.append(
                {
                    "expert_idx": k,
                    "expert": e_name,
                    "covariate_idx": -1,
                    "covariate": "bias",
                    "weight": float(b[k]),
                }
            )

        return pd.DataFrame(rows)

    def save(self, output_file):
        base_path, ext = osp.splitext(output_file)
        if ext.lower() != self.save_ext:
            output_file = base_path + self.save_ext

        output_dir = osp.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        model_name = osp.splitext(osp.basename(output_file))[0]
        config = self.export_config()
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": config,
        }

        if self.training_scaler is not None:
            checkpoint["scaler"] = copy.deepcopy(self.training_scaler)

        torch.save(checkpoint, output_file)

        if self.training_scaler is not None:
            scaler_path = osp.join(output_dir, f"{model_name}.scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(copy.deepcopy(self.training_scaler), f)

        config_path = osp.join(output_dir, f"inspect_{model_name}_config.json")
        with open(config_path, "w") as jf:
            json.dump(config, jf, indent=2)

        weights_df = self._gate_weights_dataframe()
        if weights_df is not None:
            weights_path = osp.join(output_dir, f"{model_name}_gate_weights.csv")
            weights_df.to_csv(weights_path, index=False)

    def configure_optimizers(self):

        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.LBFGS(self.parameters())

        return optimizer

# Define the gating model:
class GateModel(nn.Module):
    """
    A generic implementation for the gating model. This function can accommodate
    linear + non-linear gating models.
    """

    def __init__(self,
                 n_covar,
                 n_experts,
                 hidden_layers=None,
                 add_batch_norm=True,
                 add_layer_norm=False,
                 activation=nn.ReLU,
                 final_activation="softmax"):

        super(GateModel, self).__init__()

        self.n_covar = n_covar
        self.n_experts = n_experts

        input_dim = n_covar  # The input dimension for the gating model
        layers = []

        if hidden_layers is not None and len(hidden_layers) > 0:
            for layer_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, layer_dim))

                if add_batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dim))
                elif add_layer_norm:
                    layers.append(nn.LayerNorm(layer_dim))

                layers.append(activation())
                input_dim = layer_dim

        # Add the final layer:
        layers.append(nn.Linear(input_dim, n_experts))
        # Add the softmax activation:
        if final_activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif final_activation == "sparsemax":
            layers.append(Sparsemax(dim=1))

        self.gate = nn.Sequential(*layers)

    def forward(self, covar):
        return self.gate(covar)

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



def get_weighted_batch_sampler(dataset):

    try:
        targets = dataset.get_phenotype()
    except AttributeError:
        # If it's a subset of a dataset, extract the phenotype and then subset
        # for the given indices:
        targets = dataset.dataset.get_phenotype()[dataset.indices]

    # Compute samples weights
    class_sample_count = torch.tensor([(targets == t).sum() for t in [0, 1]])
    weight = 1. / class_sample_count.float()
    samples_weight = weight[targets.int()]

    # Create a weighted random sampler
    sampler = WeightedRandomSampler(samples_weight, targets.shape[0])

    return sampler

def get_ancestry_balanced_sampler(dataset, balance_lambda: float = 0.3):
    """
    Weighted sampler that interpolates between:
      - empirical ancestry distribution p_data
      - uniform ancestry distribution

    balance_lambda = 0.0  -> no rebalancing (p_data)
    balance_lambda = 1.0  -> fully uniform over ancestries
    """

    # get ancestry 
    try:
        ancestry = dataset.get_ancestry()          # e.g. np.array of strings or ints
    except AttributeError:
        # If we're passed a Subset, pull from the parent dataset and index
        ancestry = dataset.dataset.get_ancestry()[dataset.indices]

    # map if needed
    ancestries_unique, ancestry_ids = np.unique(ancestry, return_inverse=True)
    ancestry_tensor = torch.as_tensor(ancestry_ids, dtype=torch.long)  # (N,)

    # Counts and empirical distribution p_data
    class_sample_count = torch.bincount(ancestry_tensor)               # (K,)
    class_sample_count = class_sample_count.clamp_min(1)
    N = float(class_sample_count.sum().item())
    p_data = class_sample_count.float() / N                            # (K,)

    # target distribution q = (1-lam)*p_data + lam*uniform
    K = float(len(class_sample_count))
    uniform = torch.full_like(p_data, 1.0 / K)
    lam = float(balance_lambda)
    q = (1.0 - lam) * p_data + lam * uniform                           # (K,)

    # weights per class = q / p_data
    weights_per_class = q / class_sample_count.float()                 # (K,)
    samples_weight = weights_per_class[ancestry_tensor]                # (N,)

    sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),   # one "epoch" = N draws
        replacement=True
    )
    return sampler

#########################################################


def train_model(lit_model, dataset, max_epochs=100, prop_validation=0.2, batch_size=None, 
                weigh_samples=False, seed=8, split_seed=8, ancestry_balance_lambda=0.3):

    dataset.set_backend("torch")
    
    # Split the dataset into training and validation sets:

    if dataset.phenotype_likelihood == "binomial":
        stratify = dataset.get_phenotype()
    else:
        stratify = None

    # split
    dataset.standardize_data()
    train_idx, validation_idx = train_test_split(
        np.arange(dataset.N),
        test_size=prop_validation,
        shuffle=True,
        stratify=stratify,
        random_state=split_seed
    )

    # cache once AFTER standardization
    dataset.cache_data_matrix()

    training_dataset = IndexSubset(dataset, train_idx)
    validation_dataset = IndexSubset(dataset, validation_idx)

    if batch_size is not None:
        batch_size = min(batch_size, len(training_dataset), len(validation_dataset))

    # samplers (your existing helpers should still work because IndexSubset has .dataset and .indices)
    if dataset.phenotype_likelihood == "binomial" and weigh_samples:
        train_sampler = get_weighted_batch_sampler(training_dataset)
        validation_sampler = get_weighted_batch_sampler(validation_dataset)
    else:
        if ancestry_balance_lambda is not None:
            train_sampler = get_ancestry_balanced_sampler(
                training_dataset,
                balance_lambda=ancestry_balance_lambda,
            )
        else:
            train_sampler = None
        validation_sampler = None

    # IMPORTANT: collate_fn does the vectorized fetch
    collate = dataset.get_batch

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    # Good default for your cached+vectorized pipeline:
    train_workers = 3
    val_workers = 1
    prefetch_factor = 1

    configure_cpu_threads(cpus_per_task=cpus, num_workers=(train_workers+val_workers), interop_threads=1)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size or len(training_dataset),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_workers,              
        collate_fn=collate,
        worker_init_fn=dataloader_worker_init_fn,
        persistent_workers=(train_workers > 0),  
        prefetch_factor = prefetch_factor,   
        pin_memory = False
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size or len(validation_dataset),
        shuffle=False,
        sampler=validation_sampler,
        num_workers=val_workers,
        collate_fn=collate,
        worker_init_fn=dataloader_worker_init_fn,
        persistent_workers=(val_workers > 0),
        prefetch_factor = prefetch_factor,   
        pin_memory = False
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
                                save_top_k=1,
                                monitor="val_loss",
                            )

    from lightning.pytorch.callbacks import StochasticWeightAveraging

    import os.path as osp
    from lightning.pytorch.loggers import CSVLogger

    # one directory per phenotype (sufficient if one job per pheno)
    log_dir = osp.join("lightning_logs", dataset.phenotype_col)
    logger = CSVLogger(save_dir=log_dir, name="MoE-PyTorch")

    schedule_end = 150

    trainer = pl.Trainer(max_epochs=max_epochs,
                         deterministic=True,
                        #  logger = logger,
                        logger = False,
                        num_sanity_val_steps=0,
                         callbacks=[
                            DelayedEarlyStopping(monitor="val_loss", min_epoch=schedule_end, patience=10, mode="min"),
                            ckpt_callback,
                            ConvergenceCheck()
                        ])

    trainer.fit(model=lit_model,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)

    ckpt = torch.load(ckpt_callback.best_model_path, weights_only=False)
    lit_model.load_state_dict(ckpt['state_dict'])
    lit_model.eval()
    lit_model.training_scaler = copy.deepcopy(dataset.scaler)

    return trainer, lit_model

def make_deterministic(seed: int = 16384):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
