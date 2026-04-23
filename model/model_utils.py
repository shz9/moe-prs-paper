import copy
import glob
import os
import os.path as osp
import pickle
import threading
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import psutil
import torch
from moe_pytorch import Lit_MoEPRS
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


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

    def __init__(
        self,
        interval: float = 0.2,
        include_children: bool = True,
        track_gpu: bool = False,
    ):
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
        return self.peak_rss_bytes / (1024**3)

    @property
    def peak_cuda_alloc_gb(self) -> float:
        return self.peak_cuda_alloc_bytes / (1024**3)


def subset_standard_scaler(scaler, keep_features):
    """
    Subset a fitted StandardScaler using feature names stored in
    scaler.feature_names_in_.

    Parameters
    ----------
    scaler : fitted StandardScaler
    keep_features : list of str
        Features to retain (must be in scaler.feature_names_in_)

    Returns
    -------
    new_scaler : StandardScaler
    indices : np.ndarray
        Indices corresponding to kept features
    """
    if not hasattr(scaler, "feature_names_in_"):
        raise ValueError(
            "Scaler does not have feature_names_in_. "
            "Was it fitted on a pandas DataFrame?"
        )

    feature_names = list(scaler.feature_names_in_)

    # Map names → indices
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    missing = set(keep_features) - set(feature_names)
    if missing:
        raise ValueError(f"Features not found in scaler: {missing}")

    indices = np.array([name_to_idx[f] for f in keep_features])

    # Create new scaler
    new_scaler = StandardScaler()

    # Copy learned parameters
    new_scaler.mean_ = scaler.mean_[indices]
    new_scaler.scale_ = scaler.scale_[indices]

    if hasattr(scaler, "var_"):
        new_scaler.var_ = scaler.var_[indices]

    new_scaler.n_features_in_ = len(indices)

    new_scaler.feature_names_in_ = np.array(keep_features)

    return new_scaler


def compare_scalers(scaler1, scaler2):
    return (
        np.allclose(scaler1.mean_, scaler2.mean_)
        and np.allclose(scaler1.var_, scaler2.var_)
        and np.allclose(scaler1.scale_, scaler2.scale_)
    )


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
        "use_global_head", any(k.startswith("global_head.") for k in state.keys())
    )
    global_head_bias = cfg.get("global_head_bias", ("global_head.bias" in state))
    use_ard_bias = cfg.get(
        "use_ard_bias", ("expert_bias" in state) or ("expert_bias_log_scale" in state)
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
    """inference wrapper around a trained Lit_MoEPRS model to expose predict and predict_proba methods"""

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
        d = (
            apply_saved_scaler(prs_dataset, self.scaler_dir)
            if self.scaler_dir
            else copy.deepcopy(prs_dataset)
        )

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


@lru_cache(maxsize=None)
def get_analysis_tables():
    tabs = {}
    for f in glob.glob(
        osp.join(osp.dirname(osp.dirname(__file__)), "tables/*_prs_table.csv")
    ):
        tabs[osp.basename(f)] = pd.read_csv(f)

    return tabs


@lru_cache(maxsize=None)
def get_analysis_to_table_mapper():
    """
    Map each AnalysisID to the metadata table filename (without extension)
    where that AnalysisID is defined.
    """
    mapper = {}

    for table_name, df in get_analysis_tables().items():
        if "AnalysisID" not in df.columns:
            continue

        table_id = osp.splitext(table_name)[0]
        for analysis_id in df["AnalysisID"].dropna().astype(str).unique():
            mapper[analysis_id] = table_id

    return mapper


def get_analysis_ids_for_table(table_id):
    table_id = str(table_id)
    return sorted(
        [a for a, t in get_analysis_to_table_mapper().items() if t == table_id]
    )


def get_model_name_mapper():
    """
    Get a mapping between model ID and model name for
    each analysis ID.
    """

    tables = get_analysis_tables()

    mapper = {}

    for df in tables.values():
        mapper.update(
            {k: dict(zip(g["PGS"], g["PGS_Name"])) for k, g in df.groupby("AnalysisID")}
        )

    return mapper


def get_analysis_id_mapper(target_col):
    tabs = get_analysis_tables()

    combined_df = pd.concat(
        [df[["AnalysisID", target_col]].drop_duplicates() for df in tabs.values()]
    )

    return dict(zip(combined_df["AnalysisID"], combined_df[target_col]))
