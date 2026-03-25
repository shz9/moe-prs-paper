import os.path as osp
import os
import threading
import copy
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import time
import psutil
from viprs.eval.eval_utils import fit_linear_model

from moe_pytorch import Lit_MoEPRS

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

    @property
    def seconds(self):
        return self.elapsed

    @property
    def minutes(self):
        return self.elapsed / 60

class PeakMemory:
    """
    Samples process RSS (and optionally children RSS) and tracks the peak.
    Optionally tracks CUDA peak allocated memory too (if torch + GPU available).
    """

    def __init__(self, interval: float = 0.2, include_children: bool = True, track_gpu: bool = False):
        self.interval = float(interval)
        self.include_children = bool(include_children)
        self.track_gpu = bool(track_gpu)

        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None

        self.peak_rss_bytes = 0
        self.peak_cuda_alloc_bytes = 0

    def _read_rss_bytes(self) -> int:
        rss = 0
        try:
            rss += self._proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

        if self.include_children:
            try:
                for ch in self._proc.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return int(rss)

    def _runner(self):
        while not self._stop.is_set():
            rss = self._read_rss_bytes()
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_rss_bytes = 0
        self.peak_cuda_alloc_bytes = 0

        # Optional CUDA peak tracking
        if self.track_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        self._stop.clear()
        self._thread = threading.Thread(target=self._runner, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        # Capture CUDA peak (allocated) if requested
        if self.track_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.peak_cuda_alloc_bytes = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        return False  # don't suppress exceptions

    @property
    def peak_rss_gb(self) -> float:
        return self.peak_rss_bytes / (1024 ** 3)

    @property
    def peak_cuda_alloc_gb(self) -> float:
        return self.peak_cuda_alloc_bytes / (1024 ** 3)
    
def compare_scalers(scaler1, scaler2):
    return np.allclose(scaler1.mean_, scaler2.mean_) and \
           np.allclose(scaler1.var_, scaler2.var_) and \
           np.allclose(scaler1.scale_, scaler2.scale_)

def apply_saved_scaler(prs_dataset, scaler_dir, scaler_name="MoE-PyTorch.scaler.pkl"):
    """
    Deep-copy prs_dataset and apply the training-time scaler if present.
    """
    d = copy.deepcopy(prs_dataset)
    sp = osp.join(scaler_dir, scaler_name)
    if osp.exists(sp):
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
        d.standardize_data(scaler=scaler, refit=False)
    return d


def incremental_r2_matched_null(true_val, full_pred, null_pred, covariates):
    true_val = np.asarray(true_val).reshape(-1)
    full_pred = np.asarray(full_pred).reshape(-1)
    null_pred = np.asarray(null_pred).reshape(-1)

    # genetic increment (what changes when you "turn on PRS")
    gen_pred = full_pred - null_pred

    if covariates is None:
        covariates = pd.DataFrame({"const": np.ones(len(true_val))})
        add_intercept = False
    else:
        add_intercept = True

    # Null (matched) evaluation regression: y ~ C + null_pred
    null_res = fit_linear_model(
        true_val,
        covariates.assign(null_pred=null_pred),
        add_intercept=add_intercept
    )

    # Full (matched) evaluation regression: y ~ C + null_pred + gen_pred
    full_res = fit_linear_model(
        true_val,
        covariates.assign(null_pred=null_pred, gen_pred=gen_pred),
        # covariates.assign(null_pred=null_pred, full_pred=full_pred),
        add_intercept=add_intercept
    )

    return full_res.rsquared - null_res.rsquared


def load_lit_from_pt(prs_dataset, pt_path, map_location="cpu", strict=True):
    """
    Rebuild Lit_MoEPRS from checkpoint config, robust to global_head_bias changes.
    """
    ckpt = torch.load(pt_path, map_location=map_location, weights_only=False)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(f"Malformed checkpoint (need config + state_dict): {pt_path}")

    cfg = ckpt["config"]
    state = ckpt["state_dict"]

    # Robust inference so loading never depends on stale config defaults
    use_global_head = cfg.get(
        "use_global_head",
        any(k.startswith("global_head.") for k in state.keys())
    )
    global_head_bias = cfg.get(
        "global_head_bias",
        ("global_head.bias" in state)
    )
    use_ard_bias = cfg.get(
        "use_ard_bias",
        ("expert_bias" in state) or ("expert_bias_log_scale" in state)
    )

    group_getitem_cols = {
        "phenotype": [prs_dataset.phenotype_col],
        "gate_input": prs_dataset.covariates_cols,
        "experts": prs_dataset.prs_cols,
        "global_input": prs_dataset.covariates_cols,
    }
    if cfg.get("has_expert_covariates", False):
        group_getitem_cols["expert_covariates"] = prs_dataset.covariates_cols

    lit = Lit_MoEPRS(
        group_getitem_cols=group_getitem_cols,
        gate_model_layers=cfg["gate_model_layers"],
        gate_add_batch_norm=cfg["gate_add_batch_norm"],
        loss=cfg["loss"],
        optimizer=cfg["optimizer"],
        family=cfg["family"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        topk_k=cfg.get("topk_k", None),
        tau_start=cfg.get("tau_start", 1.0),
        tau_end=cfg.get("tau_end", 1.0),
        hard_ste=cfg.get("hard_ste", True),
        lb_coef=cfg.get("lb_coef", 0.0),
        eps=cfg.get("eps", 1e-12),

        use_ard_bias=use_ard_bias,
        use_global_head=use_global_head,
        global_head_bias=global_head_bias,

        ent_coef=cfg.get("ent_coef_start", 0.0),
        ent_coef_end=cfg.get("ent_coef_end", None),
        ent_warm_epochs=cfg.get("ent_warm_epochs", 0),
        ent_decay_epochs=cfg.get("ent_decay_epochs", 0),
    )

    # optional attrs used in forward
    if cfg.get("min_sigma2", None) is not None:
        lit.min_sigma2 = float(cfg["min_sigma2"])
    if "expert_bias_scale_floor" in cfg:
        lit.expert_bias_scale_floor = float(cfg["expert_bias_scale_floor"])

    lit.load_state_dict(state, strict=strict)
    lit.eval()
    return lit


class TorchMoEWrapper:
    ''' inference wrapper around a trained Lit_MoEPRS model to expose predict and predict_proba methods '''
    def __init__(self, lit_model, scaler_dir=None, batch_size=65536, device="cpu"):
        self.lit_model = lit_model
        self.scaler_dir = scaler_dir
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        self.lit_model.to(self.device)
        self.lit_model.eval()

        # for compatibility with plotting code expecting model.expert_cols
        self.expert_cols = self.lit_model.group_getitem_cols["experts"]

    def _prep_dataset(self, prs_dataset):
        d = apply_saved_scaler(prs_dataset, self.scaler_dir) if self.scaler_dir else copy.deepcopy(prs_dataset)

        # Ensure dataset groups match training-time groups
        expected = self.lit_model.group_getitem_cols
        if not getattr(d, "group_getitem_cols", None):
            d.set_group_getitem_cols(expected)
        else:
            for k, v in expected.items():
                if k not in d.group_getitem_cols or d.group_getitem_cols[k] != v:
                    d.set_group_getitem_cols(expected)
                    break

        d.set_backend("torch")
        return d

    def predict(self, prs_dataset):
        d = self._prep_dataset(prs_dataset)
        loader = DataLoader(d, batch_size=self.batch_size, shuffle=False)

        outs = []
        with torch.no_grad():
            for batch in loader:
                # move tensors to device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)
                yhat = self.lit_model.forward(batch)
                outs.append(yhat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    def predict_proba(self, prs_dataset):
        d = self._prep_dataset(prs_dataset)
        loader = DataLoader(d, batch_size=self.batch_size, shuffle=False)

        outs = []
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)
                p = self.lit_model.gate_forward(batch)
                outs.append(p.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)
