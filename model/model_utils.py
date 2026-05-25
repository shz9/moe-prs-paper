import glob
import os
import os.path as osp
import threading
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.preprocessing import StandardScaler


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


def serialize_standard_scaler(scaler):
    if scaler is None:
        return None
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Unsupported scaler type: {type(scaler)}")

    payload = {
        "type": "StandardScaler",
        "with_mean": bool(getattr(scaler, "with_mean", True)),
        "with_std": bool(getattr(scaler, "with_std", True)),
    }
    for attr in (
        "mean_",
        "var_",
        "scale_",
        "n_features_in_",
        "n_samples_seen_",
        "feature_names_in_",
    ):
        if hasattr(scaler, attr):
            value = getattr(scaler, attr)
            if isinstance(value, np.ndarray):
                payload[attr] = value.copy()
            else:
                payload[attr] = value
    return payload


def deserialize_standard_scaler(payload):
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("type") != "StandardScaler":
        raise ValueError("Malformed scaler payload.")

    scaler = StandardScaler(
        with_mean=payload.get("with_mean", True), with_std=payload.get("with_std", True)
    )
    for attr in (
        "mean_",
        "var_",
        "scale_",
        "n_features_in_",
        "n_samples_seen_",
        "feature_names_in_",
    ):
        if attr in payload:
            setattr(scaler, attr, payload[attr])
    return scaler


def compare_scalers(scaler1, scaler2):
    return (
        np.allclose(scaler1.mean_, scaler2.mean_)
        and np.allclose(scaler1.var_, scaler2.var_)
        and np.allclose(scaler1.scale_, scaler2.scale_)
    )


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
