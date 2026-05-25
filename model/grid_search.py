import copy
import os.path as osp
import sys
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from clustering import cluster_data
from joblib import Parallel, delayed
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "evaluation/"))

from evaluate_predictive_performance import stratified_evaluation


def combine_cluster_label_with_binary_phenotype(
    cluster_label,
    binary_phenotype,
    threshold=1e-3,
    min_cases=10,
):
    """
    This helper function takes as input a vector with cluster label as well as a vector
    capturing a binary phenotype and aims to merge them so that we have a cluster-disease status
    groupings. This will help with stratified splits when we're trying to capture information
    about both clusters as well as disease status.
    """

    df = pd.DataFrame({"Cluster": cluster_label, "Phenotype": binary_phenotype})

    g_df = df.groupby("Cluster")["Phenotype"].agg(["mean", "sum", "size"]).reset_index()

    valid_cluster = g_df.loc[
        (threshold <= g_df["mean"])
        & (g_df["mean"] <= 1.0 - threshold)
        & (min_cases <= g_df["sum"])
        & (g_df["sum"] <= g_df["size"] - min_cases),
        ["Cluster"],
    ]

    merged_df = df.merge(valid_cluster, how="left", indicator=True)

    df["New Cluster"] = np.where(
        merged_df["_merge"] == "left_only",
        merged_df["Cluster"],
        merged_df["Cluster"].astype(str) + merged_df["Phenotype"].astype(str),
    )

    return pd.Categorical(df["New Cluster"]).codes


def _stratified_sample_and_split(
    dataset: Any,
    n_splits: int,
    max_size: int = None,
    rng=np.random,
    cluster_fn=None,
    return_cluster_labels=False,
) -> List[np.ndarray]:
    """
    Use sklearn to (1) sample `max_size` indices with stratification (preserving cluster_fn labels),
    then (2) split those sampled indices into `n_splits` stratified folds.

    Returns:
        folds: list of length n_splits, each an np.ndarray of indices (relative to original dataset)
    Behavior / fallbacks:
      - If some classes are too rare for StratifiedKFold (class_count < n_splits),
        fall back to a robust proportional sampling / splitting implementation that ensures
        small labels are represented as best as possible.
    """

    N = int(getattr(dataset, "N", None))
    if N is None:
        raise ValueError(
            "dataset must have attribute 'N' indicating total sample size."
        )

    max_size = max_size or N

    # --------------------------------------------------------
    cluster_labels = None
    # Get labels; call cluster_fn on a deepcopy if it might mutate dataset
    if cluster_fn is not None:
        cluster_labels = cluster_fn(dataset)

    if dataset.phenotype_likelihood == "binomial":
        if cluster_labels is None:
            cluster_labels = dataset.get_phenotype().flatten()
        else:
            cluster_labels = combine_cluster_label_with_binary_phenotype(
                cluster_labels, dataset.get_phenotype().flatten()
            )

    # --------------------------------------------------------
    # If requested max_size >= N, just use all indices and perform StratifiedKFold on entire dataset (if possible)
    all_indices = np.arange(N)
    if max_size >= N:
        sampled_indices = all_indices
    else:
        # Use StratifiedShuffleSplit to sample `total_used` indices with preserved label proportions
        train_size = float(max_size) / N  # StratifiedShuffleSplit uses train_size param
        if cluster_labels is not None and train_size < 0.8:
            # For binary phenotypes, incorporate case-control status along with cluster label:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=train_size,
                random_state=rng,
            )
            # sss.split yields train_idx, test_idx relative to 0..N-1; we treat train_idx as the sampled set
            train_idx, _ = next(sss.split(all_indices, cluster_labels))
            sampled_indices = all_indices[train_idx]
        else:
            train_idx, _ = train_test_split(all_indices, test_size=1.0 - train_size)
            sampled_indices = all_indices[train_idx]

    # --------------------------------------------------------
    # Now sampled_indices is our subset; get labels for sampled set
    if cluster_labels is not None:
        sampled_labels = cluster_labels[sampled_indices]

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rng,
        )

        ss = skf.split(np.zeros(len(sampled_indices)), sampled_labels)

    else:
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rng,
        )
        ss = kf.split(np.zeros(len(sampled_indices)))

    # --------------------------------------------------------
    folds = []
    for _, test_pos in ss:
        fold_indices = sampled_indices[test_pos]
        folds.append(np.asarray(fold_indices, dtype=int))

    if return_cluster_labels:
        return folds, cluster_labels
    else:
        return folds


def get_gate_penalty_ladder(n_steps=8):
    """
    Get a ladder of ridge penalty parameters for the gating model
    """
    return np.concatenate([[0.0], np.logspace(-6, np.log10(0.1), num=n_steps - 1)])


def _instantiate_model_with_params(
    baseline_model: Any, dataset: Any, params: Dict[str, Any]
) -> Any:
    """
    Try to create a fresh model instance that takes (dataset, **params) at instantiation.
    """
    # 1) If baseline_model is a callable (class or factory), try to call it
    return baseline_model(dataset, **params)


def _train_model_instance(model: Any, fit_params=None) -> Any:
    """
    Train model in-place using model.fit() (no args). Returns trained model.
    """
    if not hasattr(model, "fit"):
        raise RuntimeError(
            "Model instance does not expose 'fit()' method required by this routine."
        )

    fit_params = fit_params or {}
    model.fit(**fit_params)
    return model


def _get_model_name(model):
    # partial(Model, ...)
    if isinstance(model, partial):
        return model.func.__name__

    # class
    if hasattr(model, "__name__"):
        return model.__name__

    # instance
    return model.__class__.__name__


def _name_for_params(base_name: str, params: Dict[str, Any]) -> str:
    """Deterministic human-readable name for model variant used to match evaluate_prs_models 'PGS'."""
    if not params:
        return base_name
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return f"{base_name}[" + ",".join(parts) + "]"


# ---------- Per-parameter worker (executed in parallel) ----------


def _evaluate_param_combo(
    params: Dict[str, Any],
    dataset: Any,
    baseline_model: Any,
    folds: List[np.ndarray],
    base_name: str,
    evaluation_metric: str,
    fit_params=None,
) -> Dict[str, Any]:
    """
    Evaluate one hyperparameter combination across the provided folds.
    Returns a dict with 'params', 'mean_metric', 'std_metric', 'fold_metrics'.
    This function will be pickled and run in a worker process when using joblib.
    """
    fold_metric_values = []

    for fold_idx in range(len(folds)):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate(
            [folds[j] for j in range(len(folds)) if j != fold_idx]
        )

        # deepcopy dataset before in-place filtering
        train_ds = copy.deepcopy(dataset)
        train_ds.filter_samples(train_idx.astype(int))

        test_ds = copy.deepcopy(dataset)
        test_ds.filter_samples(test_idx.astype(int))

        # instantiate and train
        try:
            model = _instantiate_model_with_params(baseline_model, train_ds, params)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate model for params={params} on fold {fold_idx}: {e}"
            )

        try:
            trained_model = _train_model_instance(model, fit_params=fit_params)
        except Exception as e:
            raise RuntimeError(
                f"Training failed for params={params} on fold {fold_idx}: {e}"
            )

        # evaluate - model name must match 'PGS' produced by evaluate_prs_models
        model_name = _name_for_params(base_name, params)
        try:
            eval_df = stratified_evaluation(
                test_ds,
                trained_models={model_name: trained_model},
                evaluate_base_models=False,
                metrics=evaluation_metric,
            )
        except Exception as e:
            raise RuntimeError(
                f"evaluate_prs_models failed for params={params} on fold {fold_idx}: {e}"
            )

        if "PGS" not in eval_df.columns:
            raise RuntimeError(
                "evaluate_prs_models output must contain a 'PGS' column."
            )

        filtered = eval_df[eval_df["PGS"] == model_name]
        if filtered.shape[0] == 0:
            raise RuntimeError(
                f"evaluate_prs_models did not return an entry for model '{model_name}' on fold {fold_idx}."
            )

        if evaluation_metric not in filtered.columns:
            raise RuntimeError(
                f"Evaluation metric '{evaluation_metric}' not found in evaluation DataFrame columns: {list(filtered.columns)}"
            )

        # mean across multiple rows if present
        metric_val = float(filtered[evaluation_metric].astype(float).mean())
        fold_metric_values.append(metric_val)

    mean_metric = float(np.mean(fold_metric_values))
    std_metric = float(np.std(fold_metric_values, ddof=0))

    return {
        "params": params,
        "mean_metric": mean_metric,
        "std_metric": std_metric,
        "fold_metrics": fold_metric_values,
    }


# ---------------- Main function (parallelized) ----------------


def custom_cv_grid_search(
    dataset: Any,
    baseline_model: Any,
    param_grid: Dict[str, List[Any]],
    n_splits: int = 2,
    max_validation_size: Optional[int] = 10_000,
    evaluation_metric: str = "Incremental_R2",
    rescale_penalty=True,
    random_state: Optional[int] = None,
    minimize_metric: bool = False,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    validation_fit_params=None,
) -> Any:
    """
    Parallelized custom cross-validation + grid-search.

    Parameters
    ----------
    dataset : object
        Must expose:
          - attribute `N` : int (total samples)
          - method `filter_samples(indices)` which mutates in-place
    baseline_model : class or instance
        Model factory/class or prototype instance. Preferred constructor signature:
            MyModel(dataset, **params)
    param_grid : dict
        Hyperparameter grid (sklearn-style).
    n_splits : int
        Number of folds.
    max_validation_size : int or None
        Max size per validation subset (or None to use floor(N/n_splits)).
    evaluation_metric : str
        Column name inside evaluate_prs_models output used to choose best model.
    random_state : int or None
        RNG seed used to sample indices for the folds.
    minimize_metric : bool
        If True, pick hyperparams minimizing the metric.
    n_jobs : int
        Number of parallel jobs for joblib. Use -1 to use all CPUs.
    joblib_backend : str
        Backend for joblib. 'loky' (default) is robust for CPU-bound tasks. 'multiprocessing' or 'threading' are alternatives.
    verbose : int
        Verbosity forwarded to joblib.Parallel.

    Returns
    -------
    best_model_trained : object
        Model instance trained on the full dataset with best hyperparameters.
        Will have attributes `_cv_search_results` and `_cv_best_params` attached when possible.
    """

    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Validate dataset.N
    N = getattr(dataset, "N", None)
    if N is None:
        raise ValueError(
            "dataset must have attribute 'N' indicating total sample size."
        )
    N = int(N)

    # Determine per-fold size
    if max_validation_size is None:
        per_fold_target = N // n_splits
    else:
        per_fold_target = min(max_validation_size, N // n_splits)

    if per_fold_target <= 0:
        raise ValueError(
            "Computed per-fold size <= 0. Check N, n_splits and max_validation_size."
        )

    total_used = per_fold_target * n_splits

    # create stratified folds using sklearn utilities
    gate_input = getattr(baseline_model, "gate_input_cols", None)
    if gate_input is not None:
        gate_input += [dataset.phenotype_col]
    cluster_fn = partial(cluster_data, data_columns=gate_input, n_jobs=n_jobs)

    folds = _stratified_sample_and_split(
        dataset,
        n_splits=n_splits,
        max_size=total_used,
        rng=rng,
        cluster_fn=cluster_fn,
    )

    # ----------------------------------------------------
    # Update the evaluation metric based on phenotype likelihood:

    if (
        evaluation_metric == "Incremental_R2"
        and dataset.phenotype_likelihood == "binomial"
    ):
        evaluation_metric = "Nagelkerke_R2"

    # ----------------------------------------------------
    # build parameter grid
    grid = list(ParameterGrid(param_grid))
    if not grid:
        grid = [{}]

    # base name for model naming
    base_name = _get_model_name(baseline_model)

    print(f"> Performing grid search for {base_name}")
    print("Parameter grid:\n", param_grid)

    # Use joblib.Parallel to evaluate each param combo in parallel
    # NOTE: dataset and baseline_model must be picklable for this to work
    parallel = Parallel(n_jobs=n_jobs, backend=joblib_backend)
    tasks = (
        delayed(_evaluate_param_combo)(
            params,
            dataset,
            baseline_model,
            folds,
            base_name,
            evaluation_metric,
            fit_params=validation_fit_params,
        )
        for params in grid
    )

    # Execute parallel jobs
    try:
        results_list = parallel(tasks)
    except Exception as e:
        raise RuntimeError(
            f"Parallel grid search failed (possible non-picklable object): {e}"
        )

    # Convert results into DataFrame
    results_df = pd.DataFrame(
        [
            {**r, **{f"param_{k}": v for k, v in r["params"].items()}}
            for r in results_list
        ]
    )

    # choose best
    if minimize_metric:
        best_idx = results_df["mean_metric"].idxmin()
    else:
        best_idx = results_df["mean_metric"].idxmax()

    best_params = results_df.loc[best_idx, "params"].copy()

    print("> Best set of hyperparameters:", best_params)

    if rescale_penalty:
        for p in best_params:
            if "penalty" in p:
                best_params[p] *= float(per_fold_target) / N

        print(
            "> Rescaled best hyperparameters (accounting for subsampling):", best_params
        )

    # instantiate final model on deepcopy of dataset (safe) and train on full dataset
    final_dataset = dataset
    try:
        best_model = _instantiate_model_with_params(
            baseline_model, final_dataset, best_params
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate best model on full dataset with params={best_params}: {e}"
        )

    try:
        best_model_trained = _train_model_instance(best_model)
    except Exception as e:
        raise RuntimeError(
            f"Final training on full dataset failed for params={best_params}: {e}"
        )

    # attach results for inspection if allowed
    try:
        setattr(best_model_trained, "_cv_search_results", results_df)
        setattr(best_model_trained, "_cv_best_params", best_params)
    except Exception:
        pass

    return best_model_trained
