import numpy as np
import pandas as pd
from viprs.eval import eval_incremental_metrics as INCREMENTAL_METRICS
from viprs.eval import eval_metric_names as EVAL_METRICS

# Minimum number of samples in a group used for evaluation:
DEFAULT_MIN_GROUP_SIZE = 30
# Minimum number of cases in a group used for evaluation (for binary classification):
DEFAULT_MIN_CASES = 15
# Default number of bootstrap replicates used for metric confidence intervals:
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
# Default bootstrap confidence interval coverage:
DEFAULT_BOOTSTRAP_CI = 0.95

CONT_EVAL_METRICS = {
    m: f
    for m, f in EVAL_METRICS.items()
    if m
    in (
        "Pearson_R",
        "Incremental_R2",
        "Partial_Correlation",
    )
}


BINARY_EVAL_METRICS = {
    m: f
    for m, f in EVAL_METRICS.items()
    if m in [
        'Liability_R2',
        'Nagelkerke_R2',
        'AUROC',
        'AUPRC'
    ]
}

PSEUDO_R2_METRICS = {
    "Liability_R2",
    "Nagelkerke_R2",
    "CoxSnell_R2",
    "McFadden_R2",
    "Liability_Probit_R2",
    "Liability_Logit_R2",
}


def is_pseudo_r2_metric(metric):
    """
    Return True for R²-like metrics whose uncertainty should not be estimated
    with the closed-form linear-model R² standard error.
    """
    return str(metric) in PSEUDO_R2_METRICS


def _is_binary_vector(y):
    y = np.asarray(y).reshape(-1)
    y = y[~pd.isna(y)]
    if y.size == 0:
        return False

    try:
        y_float = y.astype(float)
    except (TypeError, ValueError):
        return False

    return set(np.unique(y_float)).issubset({0.0, 1.0})


def _row_missing_mask(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return pd.isna(arr)
    return pd.isna(arr).reshape(arr.shape[0], -1).any(axis=1)


def bootstrap_metric_ci(
    true_val,
    pred_val,
    metric=None,
    metric_func=None,
    metric_args=None,
    phenotype_likelihood=None,
    n_bootstrap=DEFAULT_BOOTSTRAP_RESAMPLES,
    ci=DEFAULT_BOOTSTRAP_CI,
    random_state=None,
    min_samples=DEFAULT_MIN_GROUP_SIZE,
    min_cases=DEFAULT_MIN_CASES,
    return_distribution=False,
):
    """
    Estimate a metric standard error and confidence interval with non-parametric
    bootstrap resampling.

    For binary phenotypes, resampling is stratified by case/control status and
    keeps the original number of cases and controls in every bootstrap replicate.
    This avoids invalid replicates for metrics such as AUROC or pseudo-R² and
    keeps the case/control ratio fixed within the evaluated group.

    For continuous phenotypes, rows are sampled with replacement from the full
    evaluated group.

    Parameters
    ----------
    true_val : array-like of shape (n_samples,)
        Observed phenotype values.
    pred_val : array-like of shape (n_samples,)
        Predicted scores or fitted values.
    metric : str, optional
        Metric name in ``EVAL_METRICS``. Required when ``metric_func`` is None.
    metric_func : callable, optional
        Callable with signature ``metric_func(y, pred, *metric_args)``. This is
        useful for incremental metrics that need a null prediction or covariate
        matrix in addition to ``y`` and ``pred``.
    metric_args : tuple/list, optional
        Additional row-aligned arrays passed to ``metric_func``.
    phenotype_likelihood : {"binomial", "gaussian"}, optional
        Phenotype likelihood. If omitted, binary phenotypes are inferred from
        0/1 values in ``true_val``.
    n_bootstrap : int, default=1000
        Number of bootstrap replicates.
    ci : float, default=0.95
        Confidence interval coverage.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducible resampling.
    min_samples : int, default=DEFAULT_MIN_GROUP_SIZE
        Minimum number of complete rows required.
    min_cases : int, default=DEFAULT_MIN_CASES
        Minimum number of cases and controls required for binary resampling.
    return_distribution : bool, default=False
        If True, include the finite bootstrap replicate values in the result.

    Returns
    -------
    dict
        Dictionary containing ``value``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``n_bootstrap``, and ``n_bootstrap_valid``.
    """

    if metric_func is None:
        if metric is None:
            raise ValueError("Either metric or metric_func must be provided.")
        metric_func = EVAL_METRICS[metric]

    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    ci = float(ci)
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1).")

    y = np.asarray(true_val).reshape(-1)
    pred = np.asarray(pred_val).reshape(-1)
    if y.shape[0] != pred.shape[0]:
        raise ValueError(
            f"true_val and pred_val length mismatch: {y.shape[0]} vs {pred.shape[0]}."
        )

    if metric_args is None:
        metric_args = ()
    elif not isinstance(metric_args, (tuple, list)):
        metric_args = (metric_args,)

    extra_args = []
    valid = (~pd.isna(y)) & (~pd.isna(pred))
    for i, arg in enumerate(metric_args):
        arr = arg.to_numpy() if hasattr(arg, "to_numpy") else np.asarray(arg)
        if arr.shape[0] != y.shape[0]:
            raise ValueError(
                f"metric_args[{i}] length mismatch: expected {y.shape[0]}, got {arr.shape[0]}."
            )
        valid &= ~_row_missing_mask(arr)
        extra_args.append(arr)

    y = y[valid]
    pred = pred[valid]
    extra_args = [arr[valid] for arr in extra_args]
    n = int(y.shape[0])

    out = {
        "value": np.nan,
        "se": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n": n,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_valid": 0,
    }

    if n < min_samples:
        if return_distribution:
            out["bootstrap_values"] = np.array([], dtype=float)
        return out

    is_binary = phenotype_likelihood == "binomial" or (
        phenotype_likelihood is None and _is_binary_vector(y)
    )

    if is_binary:
        y_float = y.astype(float)
        case_idx = np.where(y_float == 1.0)[0]
        control_idx = np.where(y_float == 0.0)[0]
        if len(case_idx) < min_cases or len(control_idx) < min_cases:
            if return_distribution:
                out["bootstrap_values"] = np.array([], dtype=float)
            return out
        sample_indices = (
            lambda rng: np.concatenate(
                [
                    rng.choice(case_idx, size=len(case_idx), replace=True),
                    rng.choice(control_idx, size=len(control_idx), replace=True),
                ]
            )
        )
    else:
        all_idx = np.arange(n)
        sample_indices = lambda rng: rng.choice(all_idx, size=n, replace=True)

    observed_args = [arr for arr in extra_args]
    try:
        out["value"] = float(metric_func(y, pred, *observed_args))
    except Exception:
        if return_distribution:
            out["bootstrap_values"] = np.array([], dtype=float)
        return out

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    boot_values = []
    for _ in range(n_bootstrap):
        idx = sample_indices(rng)
        if is_binary:
            rng.shuffle(idx)

        try:
            boot_val = float(
                metric_func(y[idx], pred[idx], *[arr[idx] for arr in extra_args])
            )
        except Exception:
            continue

        if np.isfinite(boot_val):
            boot_values.append(boot_val)

    boot_values = np.asarray(boot_values, dtype=float)
    out["n_bootstrap_valid"] = int(boot_values.shape[0])

    if boot_values.shape[0] >= 2:
        alpha = (1.0 - ci) / 2.0
        out["se"] = float(np.std(boot_values, ddof=1))
        out["ci_lower"] = float(np.quantile(boot_values, alpha))
        out["ci_upper"] = float(np.quantile(boot_values, 1.0 - alpha))

    if return_distribution:
        out["bootstrap_values"] = boot_values

    return out


def rowwise_cosine_similarity(X, Y, eps=1e-12):
    """
    Compute cosine similarity row-wise between two matrices.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_components)
    Y : array-like of shape (n_samples, n_components)
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Cosine similarity for each row.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: {X.shape} vs {Y.shape}")

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)

    denom = np.clip(X_norm * Y_norm, eps, None)
    return np.sum(X * Y, axis=1) / denom


def generate_predictions(prs_dataset, models):
    preds = {}

    for m_name, m in models.items():
        try:
            preds[m_name + "-PRS-only"] = m.predict_prs(prs_dataset).flatten()
        except Exception as e:
            pass

        try:
            preds[m_name] = m.predict(prs_dataset).flatten()
        except Exception as e:
            print(f"Failed to predict for {m_name}: {e}")
            pass

    return pd.DataFrame(preds)


def subsample_to_prevalence(
    prs_dataset,
    desired_prevalence,
    mask=None,
    random_state=None,
    return_info=False,
):
    """
    Subsample individuals to match a target case prevalence for binary phenotypes.

    Parameters
    ----------
    prs_dataset : PRSDataset
        Dataset containing phenotype values.
    desired_prevalence : float
        Target prevalence in [0, 1].
    mask : array-like, optional
        Optional eligible-sample selector. Can be:
        - boolean array of length prs_dataset.N
        - integer index array
        If None, all samples are eligible.
    random_state : int, optional
        Seed for reproducible sampling.
    return_info : bool, default=False
        If True, also return a summary dictionary.

    Returns
    -------
    sampled_mask : np.ndarray of shape (N,), dtype=bool
        Boolean mask indicating sampled individuals.
    info : dict, optional
        Returned only when return_info=True.
    """

    if prs_dataset.phenotype_likelihood != "binomial":
        raise ValueError(
            "subsample_to_prevalence requires a binomial phenotype likelihood."
        )

    desired_prevalence = float(desired_prevalence)
    if not (0.0 <= desired_prevalence <= 1.0):
        raise ValueError("desired_prevalence must be in [0, 1].")

    N = prs_dataset.N
    prs_dataset.set_backend("numpy")
    y = np.asarray(prs_dataset.get_phenotype()).reshape(-1)
    if y.shape[0] != N:
        raise ValueError(f"Phenotype length mismatch: expected {N}, got {y.shape[0]}.")

    if mask is None:
        eligible = np.ones(N, dtype=bool)
    else:
        mask_arr = np.asarray(mask)
        if mask_arr.dtype == bool:
            if mask_arr.shape[0] != N:
                raise ValueError(
                    f"Boolean mask length mismatch: expected {N}, got {mask_arr.shape[0]}."
                )
            eligible = mask_arr.copy()
        else:
            eligible = np.zeros(N, dtype=bool)
            idx = mask_arr.astype(int).reshape(-1)
            if np.any((idx < 0) | (idx >= N)):
                raise ValueError("Index mask contains out-of-range values.")
            eligible[idx] = True

    # Exclude missing phenotype values.
    eligible = eligible & (~pd.isna(y))
    y_eligible = y[eligible]
    uniq = np.unique(y_eligible)
    if not set(uniq).issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(
            "Phenotype values in eligible set must be binary (0/1) for this utility."
        )

    eligible_idx = np.where(eligible)[0]
    case_idx = eligible_idx[y_eligible.astype(float) == 1.0]
    control_idx = eligible_idx[y_eligible.astype(float) == 0.0]

    n_cases = int(case_idx.shape[0])
    n_controls = int(control_idx.shape[0])
    if n_cases + n_controls == 0:
        raise ValueError(
            "No eligible samples after applying mask and missingness filter."
        )

    if desired_prevalence in (0.0, 1.0):
        n_case_keep = n_cases if desired_prevalence == 1.0 else 0
        n_ctrl_keep = n_controls if desired_prevalence == 0.0 else 0
    else:
        if n_cases == 0 or n_controls == 0:
            raise ValueError(
                "Cannot achieve prevalence in (0, 1) when only one class is present."
            )

        # Maximize retained sample size under downsampling-only constraints.
        if (n_cases / desired_prevalence) <= (n_controls / (1.0 - desired_prevalence)):
            n_case_keep = n_cases
            n_ctrl_keep = int(
                np.floor(n_case_keep * (1.0 - desired_prevalence) / desired_prevalence)
            )
        else:
            n_ctrl_keep = n_controls
            n_case_keep = int(
                np.floor(n_ctrl_keep * desired_prevalence / (1.0 - desired_prevalence))
            )

        n_case_keep = min(n_case_keep, n_cases)
        n_ctrl_keep = min(n_ctrl_keep, n_controls)

        if n_case_keep + n_ctrl_keep == 0:
            raise ValueError(
                "No samples selected. Try a different desired_prevalence or eligibility mask."
            )

    rng = np.random.default_rng(random_state)
    sampled_case_idx = (
        rng.choice(case_idx, size=n_case_keep, replace=False)
        if n_case_keep > 0
        else np.array([], dtype=int)
    )
    sampled_ctrl_idx = (
        rng.choice(control_idx, size=n_ctrl_keep, replace=False)
        if n_ctrl_keep > 0
        else np.array([], dtype=int)
    )

    sampled_mask = np.zeros(N, dtype=bool)
    sampled_mask[sampled_case_idx] = True
    sampled_mask[sampled_ctrl_idx] = True

    if not return_info:
        return sampled_mask

    n_keep = int(sampled_mask.sum())
    achieved_prev = float(n_case_keep / n_keep) if n_keep > 0 else np.nan
    info = {
        "desired_prevalence": desired_prevalence,
        "achieved_prevalence": achieved_prev,
        "n_eligible": int(eligible.sum()),
        "n_cases_eligible": n_cases,
        "n_controls_eligible": n_controls,
        "n_cases_sampled": int(n_case_keep),
        "n_controls_sampled": int(n_ctrl_keep),
        "n_sampled": n_keep,
    }
    return sampled_mask, info


def generate_pc_cluster_masks(prs_dataset, reference="median", n_clusters=5):
    """
    Cluster samples based on their distance in Principal Component space from
    a reference point (mean or median). This function takes a PRSDataset object
    and returns a dictionary of masks, where each key is the quantile distance
    index and the value is a boolean mask for the samples in that quantile.

    :param prs_dataset: A PRSDataset object.
    :param reference: The reference point to use for the distance calculation.
                      Can be either 'median' or 'mean'.
    :param n_clusters: The number of clusters to use for the quantile distance calculation.

    """

    masks = {"PC_DIST": {}}

    pc_dist_clust = rank_individuals_by_pc_distance(
        prs_dataset, reference, n_clusters=n_clusters
    )

    for pc_clust in np.unique(pc_dist_clust):
        masks["PC_DIST"][pc_clust] = pc_dist_clust == pc_clust

    return masks


def generate_continuous_masks(prs_dataset, cont_group_cols, n_bins=4):
    """
    Generate masks based on the quantiles of continuous columns in the
    PRS dataset. This function takes a PRSDataset object and a list of
    continuous columns by which to group the samples. It returns a nested
    dictionary of masks, where each key is a group name and the value is a
    dictionary of masks for that group. For example, if Age is used as a
    continuous variable, we find the appropriate quantiles based on
    the number of bins and return the following dictionary:

    {
        'Age': {
            'Age (Q1)': age == q1_age,
            'Age (Q2)': age == q2_age,
            'Age (Q3)': age == q3_age,
            'Age (Q4)': age == q4_age
        }
    }

    :param prs_dataset: A PRSDataset object
    :param cont_group_cols: A list of continuous columns by which to group the samples
    :param n_bins: The number of bins to use for the quantiles. Can be an integer or a list of
    integers the same length as cont_group_cols.

    """

    prs_dataset.set_backend("numpy")

    if isinstance(n_bins, int):
        n_bins = [n_bins] * len(cont_group_cols)

    if isinstance(cont_group_cols, str):
        cont_group_cols = [cont_group_cols]

    masks = {}

    for gcol, gbins in zip(cont_group_cols, n_bins):
        col_data = prs_dataset.get_data_columns(gcol).flatten()

        masks[gcol] = {}

        try:
            qcut_groups = pd.qcut(col_data, gbins, labels=list(range(gbins)))
            for i in range(gbins):
                masks[gcol][f"{gcol} (Q{i + 1})"] = qcut_groups == i
        except ValueError as e:
            print(e)
            continue

    return masks


def generate_coarse_ancestry_masks(
    prs_dataset,
    ancestry_col="Ancestry",
    ref_ancestry="EUR",
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
):
    """
    Generate masks for coarse ancestry groupings: EUR and non-EUR.
    """

    prs_dataset.set_backend("numpy")

    masks = {}

    # Get the data for the group column:
    col_data = prs_dataset.get_data_columns(ancestry_col).flatten()

    # Initialize the masks dictionary for the group column:
    masks["Coarse Ancestry"] = {
        ref_ancestry: col_data == ref_ancestry,
        f"non-{ref_ancestry}": col_data != ref_ancestry,
    }

    return masks


def generate_categorical_masks(
    prs_dataset, cat_group_cols, min_group_size=DEFAULT_MIN_GROUP_SIZE
):
    """
    Generate masks for the different groups in the dataset.
    This function takes a PRSDataset object and a list of categorical columns
    by which to group the samples. It returns a nested dictionary of masks, where each
    key is a group name and the value is a dictionary of masks for that group.
    For example, if Sex is used as a categorical variable to stratify the samples,
    the output would be:

    {
        'Sex': {
            'Males': mask
            'Females': ~mask
        }
    }

    :param prs_dataset: A PRSDataset object
    :param cat_group_cols: A list of categorical columns by which to group the samples

    """

    prs_dataset.set_backend("numpy")

    if isinstance(cat_group_cols, str):
        cat_group_cols = [cat_group_cols]

    masks = {}

    for gcol in cat_group_cols:
        # Get the data for the group column:
        col_data = prs_dataset.get_data_columns(gcol).flatten()

        # Determine the unique categories in the column:
        uniq_cats = np.unique(col_data)

        # If the categorical variable contains a single category, skip it
        if len(uniq_cats) < 2:
            print(
                "> Skipping", gcol, "as it contains a single category in this dataset."
            )
            continue

        # Initialize the masks dictionary for the group column:
        masks[gcol] = {}

        for cat in np.unique(col_data):
            msk = col_data == cat
            if msk.sum() < min_group_size:
                print(
                    f"> Skipping {gcol}={cat} as it contains less than {min_group_size} samples."
                )
                continue

            # If the category is a numeric value, first check that it can be converted
            # to an integer and then convert to a string:
            try:
                if float(cat) == int(cat):
                    cat = str(int(cat))
            except ValueError:
                pass

            masks[gcol][cat] = msk

    return masks


def rank_groups_by_pc_distance(prs_dataset, group_col, reference_group="largest"):
    prs_dataset.set_backend("numpy")

    data_standardized = prs_dataset.scaled_data

    if data_standardized:
        prs_dataset.inverse_standardize_data()

    df_col = [c for c in prs_dataset.data.columns if c.upper().startswith("PC")] + [
        group_col
    ]

    pc_df = pd.DataFrame(prs_dataset.get_data_columns(df_col), columns=df_col)

    mean_pcs = pc_df.groupby(group_col).mean()

    # Get the reference cluster (if name is not specified):
    if reference_group == "largest":
        uniq_clust, counts = np.unique(pc_df[group_col], return_counts=True)
        reference_group = uniq_clust[np.argmax(counts)]

    if data_standardized:
        prs_dataset.standardize_data()

    return sorted(
        mean_pcs.index,
        key=lambda x: np.sqrt(
            ((mean_pcs.loc[x, :] - mean_pcs.loc[reference_group, :]) ** 2).sum()
        ),
    )


def rank_individuals_by_pc_distance(prs_dataset, reference="median", n_clusters=5):
    assert reference in ["median", "mean"], (
        f"Reference must be either 'median' or 'mean'. Got: {reference}."
    )

    prs_dataset.set_backend("numpy")

    data_standardized = prs_dataset.scaled_data

    if data_standardized:
        prs_dataset.inverse_standardize_data()

    pc_cols = [c for c in prs_dataset.data.columns if c.upper().startswith("PC")]

    pc_df = pd.DataFrame(prs_dataset.get_data_columns(pc_cols), columns=pc_cols)

    if reference == "median":
        ref_val = pc_df.median(axis=0)
    elif reference == "mean":
        ref_val = pc_df.mean(axis=0)

    if data_standardized:
        prs_dataset.standardize_data()

    return pd.qcut(
        np.sqrt(((pc_df - ref_val) ** 2).sum(axis=1)),
        n_clusters,
        labels=[f"PC_DIST (Q{i})" for i in range(1, n_clusters + 1)],
    )


def average_precision_at_top_percentile(y_true, y_pred, percentile=0.05):
    """
    Computes average precision for identifying the top percentile of y_true using y_pred.

    Parameters:
    - y_true: array-like, true continuous target values
    - y_pred: array-like, predicted scores
    - percentile: float, e.g., 0.05 for top 5%

    Returns:
    - average_precision: float, average precision score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine the threshold to consider top percentile
    threshold = np.percentile(y_true, 100 * (1 - percentile))  # top X%

    # Binary labels: 1 if in top X%, else 0
    y_top = (y_true >= threshold).astype(int)

    from sklearn.metrics import average_precision_score

    # Average precision score
    ap = average_precision_score(y_top, y_pred)

    return ap


def incremental_r2_from_predictions(
    true_val, full_pred, null_pred, metric="Incremental_R2"
):
    """
    This function assumes that the null and full models have already incorporated all
    the shared components (e.g. covariates) into their predictions.
    """

    assert metric in INCREMENTAL_METRICS

    true_val = np.asarray(true_val).reshape(-1)
    full_pred = np.asarray(full_pred).reshape(-1)
    null_pred = np.asarray(null_pred).reshape(-1)

    r2_null = EVAL_METRICS[metric](true_val, null_pred)
    r2_full = EVAL_METRICS[metric](true_val, full_pred)

    return r2_full - r2_null
