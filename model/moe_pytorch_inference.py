# model/moe_pytorch_inference.py

import os.path as osp
import copy
import pickle
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from moe import MoEPRS
from moe_pytorch import Lit_MoEPRS


def apply_saved_scaler(prs_dataset, model_dir, scaler_fname="MoE-PyTorch.scaler.pkl"):
    """
    Returns a deep-copied dataset with the training-time scaler applied (no refit).
    If no scaler exists, returns a deep-copied dataset unchanged.
    """
    d = copy.deepcopy(prs_dataset)

    scaler_path = osp.join(model_dir, scaler_fname)
    if osp.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        d.standardize_data(scaler=scaler, refit=False)

    return d


def build_group_getitem_cols(prs_dataset, cfg):
    """
    Reconstruct group_getitem_cols exactly like training did (based on cfg knobs).
    """
    group_getitem_cols = {
        "phenotype": [prs_dataset.phenotype_col],
        "gate_input": prs_dataset.covariates_cols,
        "experts": prs_dataset.prs_cols,
        "global_input": prs_dataset.covariates_cols,
    }
    if cfg.get("has_expert_covariates", False):
        group_getitem_cols["expert_covariates"] = prs_dataset.covariates_cols
    return group_getitem_cols


def _infer_gate_from_state(state_dict):
    """
    Infer gate architecture knobs from checkpoint state_dict so we can rebuild
    the exact same GateModel and keep strict=True loading working.

    Returns dict with:
      - gate_model_layers: list[int] or None
      - gate_add_batch_norm: bool
      - gate_add_layer_norm: bool
    """
    # BatchNorm detection (running stats are buffers in state_dict)
    has_bn = any(
        k.startswith("gate_model.gate.") and ("running_mean" in k or "running_var" in k)
        for k in state_dict.keys()
    )

    # Collect per-index "weight" tensor dimensionality
    # gate_model.gate.<idx>.weight can belong to Linear (2D) or Norm (1D)
    idx_to_wdim = {}
    idx_to_wshape0 = {}
    pat = re.compile(r"^gate_model\.gate\.(\d+)\.weight$")
    for k, v in state_dict.items():
        m = pat.match(k)
        if not m:
            continue
        idx = int(m.group(1))
        if not torch.is_tensor(v):
            continue
        idx_to_wdim[idx] = int(v.ndim)
        # useful for LN detection (1D weights)
        if v.ndim >= 1:
            idx_to_wshape0[idx] = int(v.shape[0])

    # LayerNorm detection:
    # If we have any 1D weights at gate_model.gate.<idx>.weight that are NOT BN
    # (BN already detected via running stats), treat as LN.
    has_ln = False
    if not has_bn:
        has_ln = any(dim == 1 for dim in idx_to_wdim.values())

    # Infer hidden layer sizes from Linear weights (2D) in order of Sequential index
    linear_outs = []
    for idx, dim in idx_to_wdim.items():
        if dim == 2:
            # out_features is shape[0]
            out_features = idx_to_wshape0.get(idx, None)
            if out_features is not None:
                linear_outs.append((idx, out_features))

    linear_outs.sort(key=lambda x: x[0])

    # Last Linear is logits layer -> exclude it from hidden sizes
    inferred_layers = None
    if len(linear_outs) >= 2:
        inferred_layers = [out for (_idx, out) in linear_outs[:-1]]
    elif len(linear_outs) == 1:
        inferred_layers = []  # linear-only gate (no hidden layers)

    # Never allow both norms in our implementation
    if has_bn and has_ln:
        has_ln = False

    return {
        "gate_model_layers": inferred_layers,
        "gate_add_batch_norm": bool(has_bn),
        "gate_add_layer_norm": bool(has_ln),
    }


def load_lit_from_pt(prs_dataset, pt_path, map_location="cpu", strict=True):
    """
    Load a Lit_MoEPRS from your custom .pt format:
      {"state_dict": ..., "config": ...}

    Robust to older checkpoints missing gate_add_layer_norm / stale gate_model_layers by
    inferring the correct gate structure from the state_dict.
    """
    checkpoint = torch.load(pt_path, map_location=map_location, weights_only=False)
    if "config" not in checkpoint:
        raise ValueError(f"Missing 'config' in checkpoint: {pt_path}")
    if "state_dict" not in checkpoint:
        raise ValueError(f"Missing 'state_dict' in checkpoint: {pt_path}")

    cfg = checkpoint["config"]
    state = checkpoint["state_dict"]

    # ---- Infer anything that might be missing / stale in cfg ----
    gate_inf = _infer_gate_from_state(state)

    cfg_layers = cfg.get("gate_model_layers", None)
    inf_layers = gate_inf["gate_model_layers"]

    # Prefer inferred layers if cfg is missing OR clearly disagrees
    if cfg_layers is None:
        gate_model_layers = inf_layers
    else:
        # normalize both to lists for comparison
        try:
            cfg_layers_list = list(cfg_layers) if cfg_layers is not None else None
        except Exception:
            cfg_layers_list = None
        inf_layers_list = list(inf_layers) if inf_layers is not None else None

        if (inf_layers_list is not None) and (cfg_layers_list is not None) and (cfg_layers_list != inf_layers_list):
            gate_model_layers = inf_layers
        else:
            gate_model_layers = cfg_layers

    # Prefer positive detection from state_dict; otherwise fall back to cfg
    gate_add_batch_norm = bool(gate_inf["gate_add_batch_norm"]) or bool(cfg.get("gate_add_batch_norm", False))
    gate_add_layer_norm = bool(gate_inf["gate_add_layer_norm"]) or bool(cfg.get("gate_add_layer_norm", False))
    if gate_add_batch_norm and gate_add_layer_norm:
        gate_add_layer_norm = False  # enforce exclusivity

    # Global head / bias robustness (older cfg may be missing these)
    use_global_head = cfg.get("use_global_head", any(k.startswith("global_head.") for k in state.keys()))
    global_head_bias = cfg.get("global_head_bias", ("global_head.bias" in state))

    # Per-expert bias robustness
    use_per_expert_bias = cfg.get(
        "use_per_expert_bias",
        ("expert_bias" in state) or ("expert_bias_log_scale" in state)
    )

    group_getitem_cols = build_group_getitem_cols(prs_dataset, cfg)

    lit = Lit_MoEPRS(
        group_getitem_cols=group_getitem_cols,
        gate_model_layers=gate_model_layers,
        gate_add_batch_norm=gate_add_batch_norm,
        gate_add_layer_norm=gate_add_layer_norm,
        loss=cfg["loss"],
        optimizer=cfg["optimizer"],
        family=cfg["family"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        topk_k=cfg.get("topk_k", None),
        tau_start=cfg.get("tau_start", 1.0),
        tau_end=cfg.get("tau_end", 1.0),
        tau_warm_epochs=cfg.get("tau_warm_epochs", 0),
        tau_decay_epochs=cfg.get("tau_decay_epochs", 0),
        hard_ste=cfg.get("hard_ste", True),
        lb_coef=cfg.get("lb_coef", 0.0),
        eps=cfg.get("eps", 1e-12),
        use_per_expert_bias=use_per_expert_bias,
        use_global_head=use_global_head,
        global_head_bias=global_head_bias,
        ent_coef=cfg.get("ent_coef_start", 0.0),
        ent_coef_end=cfg.get("ent_coef_end", None),
        ent_warm_epochs=cfg.get("ent_warm_epochs", 0),
        ent_decay_epochs=cfg.get("ent_decay_epochs", 0),
        binomial_logit_level=cfg.get("binomial_logit_level", False),
        center_expert_covariates=cfg.get("center_expert_covariates", True),
    )

    # Non-state attrs that affect forward
    if cfg.get("min_sigma2", None) is not None:
        lit.min_sigma2 = float(cfg["min_sigma2"])
    if "expert_bias_scale_floor" in cfg:
        lit.expert_bias_scale_floor = float(cfg["expert_bias_scale_floor"])

    lit.load_state_dict(state, strict=strict)
    lit.eval()
    lit.to(map_location)
    return lit


class IndexSubset(Dataset):
    """
    Like torch.utils.data.Subset, but returns ORIGINAL row index (int),
    so collate_fn can fetch the whole batch via PRSDataset.get_batch().
    Keeps .dataset and .indices to remain compatible with your samplers.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


class TorchMoEModel:
    """
    Unified inference wrapper for Lit_MoEPRS that supports:
      - predict(prs_dataset) -> (N,) numpy
      - predict_proba(prs_dataset) -> (N,K) numpy (gate probs)

    Batching optimization:
      - Uses IndexSubset + collate_fn=prs_dataset.get_batch (vectorized fetch)
      - Uses cached data matrix (prs_dataset.cache_data_matrix()) once
    """
    def __init__(
        self,
        lit_model,
        model_dir,
        expert_cols,
        pred_batch_size=2048,
        gate_batch_size=65536,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        scaler_fname="MoE-PyTorch.scaler.pkl",
        device="cpu",
    ):
        self.lit_model = lit_model
        self.model_dir = model_dir
        self.expert_cols = list(expert_cols)
        self.scaler_fname = scaler_fname

        self.pred_batch_size = int(pred_batch_size)
        self.gate_batch_size = int(gate_batch_size)

        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and (self.num_workers > 0)
        self.prefetch_factor = int(prefetch_factor)

        self.device = torch.device(device)
        self.lit_model.to(self.device)
        self.lit_model.eval()

    def _prepare_dataset(self, prs_dataset):
        # Apply scaler (deep copy)
        d = apply_saved_scaler(
            prs_dataset, self.model_dir, scaler_fname=self.scaler_fname
        )

        # Match group_getitem_cols expected by the Lightning model
        d.set_group_getitem_cols(self.lit_model.group_getitem_cols)

        # Torch backend + cache matrix for fast batched fetching
        d.set_backend("torch")
        d._data_matrix = None
        d.cache_data_matrix()
        return d

    def _batch_loader(self, d, batch_size):
        idx = np.arange(d.N)
        subset = IndexSubset(d, idx)
        bs = min(int(batch_size), d.N)

        # When num_workers==0, prefetch_factor must be omitted (PyTorch will error)
        dl_kwargs = dict(
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=d.get_batch,  # vectorized fetch
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        if self.num_workers > 0:
            dl_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(subset, **dl_kwargs)

    def predict(self, prs_dataset):
        d = self._prepare_dataset(prs_dataset)

        outs = []
        with torch.inference_mode():
            for batch in self._batch_loader(d, self.pred_batch_size):
                # ensure tensors are on the right device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)
                yhat = self.lit_model.forward(batch)  # (B,)
                outs.append(yhat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    def predict_cov_only_from_dataset(self, prs_dataset):
        d = self._prepare_dataset(prs_dataset)

        outs = []
        with torch.inference_mode():
            for batch in self._batch_loader(d, self.pred_batch_size):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)

                if "experts" in batch:
                    batch["experts"] = torch.zeros_like(batch["experts"])

                yhat = self.lit_model.forward(batch)
                outs.append(yhat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    def predict_proba(self, prs_dataset):
        d = self._prepare_dataset(prs_dataset)

        outs = []
        with torch.inference_mode():
            for batch in self._batch_loader(d, self.gate_batch_size):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)
                p = self.lit_model.gate_forward(batch)  # (B,K)
                outs.append(p.detach().cpu().numpy())

        return np.vstack(outs)


def load_model_any(prs_dataset, model_path, pred_batch_size=2048, gate_batch_size=65536, **wrapper_kwargs):
    """
    Unified loader:
      - .pt -> TorchMoEModel (predict + predict_proba)
      - else -> MoEPRS.from_saved_model (existing numpy model)

    wrapper_kwargs are passed to TorchMoEModel (e.g. device="cpu", num_workers=0).
    """
    if model_path.endswith(".pt"):
        lit = load_lit_from_pt(prs_dataset, model_path, map_location=wrapper_kwargs.get("device", "cpu"), strict=True)
        model_dir = osp.dirname(model_path)
        model_name = osp.splitext(osp.basename(model_path))[0]
        return TorchMoEModel(
            lit_model=lit,
            model_dir=model_dir,
            expert_cols=prs_dataset.prs_cols,
            pred_batch_size=pred_batch_size,
            gate_batch_size=gate_batch_size,
            scaler_fname=f"{model_name}.scaler.pkl",
            **wrapper_kwargs,
        )

    return MoEPRS.from_saved_model(model_path)
