import argparse
import glob
import os.path as osp
import sys
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from baseline_models import MultiPRS
from combined_accuracy_plots import plot_combined_accuracy_metrics
from error_bars import add_error_bars
from eval_utils import (
    DEFAULT_BOOTSTRAP_CI,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_MIN_GROUP_SIZE,
    generate_predictions,
    subsample_to_prevalence,
)
from evaluate_predictive_performance import (
    _parse_model_id,
    average_fold_predictions,
    evaluate_prs_models,
    stratified_evaluation,
)
from moe import GroupMeanWeightedPRS, MoEPRS
from plot_pgs_admixture import plot_admixture_graphs
from plot_utils import (
    ANALYSIS_TO_PHENOTYPE_MAP,
    ANALYSIS_TO_TABLE_MAP,
    BIOBANK_NAME_MAP,
    BIOBANK_NAME_MAP_SHORT,
    METRIC_NAME_MAP,
    MODEL_NAME_MAP,
    SEX_LABEL_MAP,
    aggregate_cross_validation_metrics,
    assign_models_consistent_colors,
    extract_accuracy_data_all_phenotypes,
    read_transform_eval_metrics,
)
from PRSDataset import PRSDataset

# -----------------------------------------------------------------------------------------


DISEASE_MATCH_FLAG_COL = "Is_Disease_Matched"
SECTION4_FONT_SCALE = 1.0
SECTION4_STANDARD_ACCURACY_FONT_SCALE = 1.25
SECTION4_FULL_PANEL_FIGSIZE = (14.8, 4.2)
SECTION4_ACCURACY_PANEL_FIGSIZE = (12.5, 2.1)
SECTION4_HALF_PANEL_FIGSIZE = (7.6, 4.2)
SECTION4_HALF_TALL_PANEL_FIGSIZE = (7.6, 5.8)
DEFAULT_SECTION4_PLOTTING_FOLD = "fold_1"

DISEASE_LABEL_MAP = {
    "Type 2 Diabetes": "T2D",
    "Type 1 Diabetes": "T1D",
    "Coronary Artery Disease": "Coronary\nArtery Disease",
    "Atrial Fibrillation": "Atrial\nFibrillation",
    "Heart Failure": "Heart\nFailure",
}


def _shorten_disease_label(label):
    return DISEASE_LABEL_MAP.get(label, label)


def _as_bool_mask(x):
    return x.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


@lru_cache(maxsize=1)
def _load_mt_disease_prs_map():
    from model_utils import get_analysis_tables

    mapping = {}
    for _, df in get_analysis_tables().items():
        required = {"AnalysisID", "PGS", DISEASE_MATCH_FLAG_COL}
        if not required.issubset(df.columns):
            continue

        matched = df.loc[_as_bool_mask(df[DISEASE_MATCH_FLAG_COL])].copy()
        if matched.empty:
            continue

        name_col = "PGS_Name" if "PGS_Name" in matched.columns else "PGS"
        matched = matched[["AnalysisID", name_col]].drop_duplicates("AnalysisID")

        mapping.update(
            dict(zip(matched["AnalysisID"].astype(str), matched[name_col].astype(str)))
        )

    return mapping


def _get_disease_prs_name(analysis_id):
    analysis_id = str(analysis_id)
    mapping = _load_mt_disease_prs_map()

    if analysis_id in mapping:
        return mapping[analysis_id]
    if analysis_id.endswith("_CTRL"):
        return mapping.get(analysis_id[:-5], analysis_id)
    return analysis_id


def _retain_disease_specific_prs(metrics_df):
    """Replace the pool of single-source scores with the matched disease PRS."""
    if metrics_df.empty or "model_category" not in metrics_df:
        return metrics_df

    is_single_prs = metrics_df["model_category"].isin(
        {"SinglePRS", "SinglePRS+Covariates"}
    )
    disease_prs_names = metrics_df["analysis_id"].map(_get_disease_prs_name)
    is_disease_prs = is_single_prs & metrics_df["model_name"].eq(disease_prs_names)

    result = metrics_df.loc[~is_single_prs | is_disease_prs].copy()
    result.loc[
        result["model_category"].isin({"SinglePRS", "SinglePRS+Covariates"}),
        "Model Name",
    ] = "Disease-specific PRS"
    return result


@lru_cache(maxsize=None)
def _get_disease_prs_id(analysis_id):
    from model_utils import get_analysis_tables

    analysis_id = str(analysis_id)
    lookup_id = analysis_id[:-5] if analysis_id.endswith("_CTRL") else analysis_id

    for _, df in get_analysis_tables().items():
        required = {"AnalysisID", "PGS", DISEASE_MATCH_FLAG_COL}
        if not required.issubset(df.columns):
            continue
        m = (df["AnalysisID"].astype(str) == lookup_id) & _as_bool_mask(
            df[DISEASE_MATCH_FLAG_COL]
        )
        if m.any():
            return str(df.loc[m, "PGS"].iloc[0])

    raise ValueError(f"Could not resolve disease-specific PGS ID for {analysis_id}.")


def _normalize_fold_name(fold):
    if fold is None:
        return None
    fold = str(fold)
    return fold if fold.startswith("fold_") else f"fold_{fold}"


def _resolve_harmonized_dataset_path(
    analysis_id,
    biobank,
    dataset,
    fold=DEFAULT_SECTION4_PLOTTING_FOLD,
):
    """Resolve a fold-aware dataset, with legacy root-layout compatibility."""
    filename = dataset if str(dataset).endswith(".pkl") else f"{dataset}.pkl"
    biobank_dir = f"data/harmonized_data/{analysis_id}/{biobank}"
    fold = _normalize_fold_name(fold)
    candidates = []
    if fold is not None and filename != "full_data.pkl":
        candidates.append(osp.join(biobank_dir, fold, filename))
    candidates.append(osp.join(biobank_dir, filename))

    for path in candidates:
        if osp.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find the requested harmonized dataset. Checked: "
        + ", ".join(candidates)
    )


def _resolve_trained_model_path(
    analysis_id,
    biobank,
    model_dataset,
    model_name,
    fold=DEFAULT_SECTION4_PLOTTING_FOLD,
):
    """Resolve a fold-aware trained model, with legacy-layout compatibility."""
    filename = model_name if str(model_name).endswith(".pkl") else f"{model_name}.pkl"
    biobank_dir = f"data/trained_models/{analysis_id}/{biobank}"
    fold = _normalize_fold_name(fold)
    candidates = []
    if fold is not None:
        candidates.append(osp.join(biobank_dir, fold, model_dataset, filename))
    candidates.append(osp.join(biobank_dir, model_dataset, filename))

    for path in candidates:
        if osp.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find the requested trained model. Checked: "
        + ", ".join(candidates)
    )


def _trained_model_id_from_path(model_path, analysis_id):
    analysis_root = f"data/trained_models/{analysis_id}"
    relative_path = osp.relpath(model_path, analysis_root)
    path_parts = relative_path.split(osp.sep)
    if len(path_parts) < 3:
        raise ValueError(f"Unexpected trained model path: {model_path}")
    model_name = osp.splitext(path_parts[-1])[0]
    return f"{'/'.join(path_parts[:-1])}:{model_name}"


def _drop_cartagene_sparse_trait_rows(df):
    """Drop CARTaGENE traits excluded from section 4 plotting outputs."""
    if "Phenotype" not in df.columns:
        return df

    if "Biobank" in df.columns:
        is_cartagene = df["Biobank"].eq(BIOBANK_NAME_MAP["cartagene"])
    elif "test_biobank" in df.columns:
        is_cartagene = df["test_biobank"].eq("cartagene")
    else:
        return df

    drop_phenotypes = {"Heart Failure", "Stroke"}
    return df.loc[~(is_cartagene & df["Phenotype"].isin(drop_phenotypes))].copy()


def _parse_threshold_rule(rule_text):
    rule = str(rule_text).replace(" ", "")
    for op in ("<=", ">=", "<", ">", "=="):
        if rule.startswith(op):
            try:
                val = float(rule[len(op) :])
            except ValueError as e:
                raise ValueError(
                    f"Invalid threshold rule '{rule_text}'. "
                    "Expected forms like '<=0.1', '>0.5', '==0.2'."
                ) from e
            return op, val, rule

    raise ValueError(
        f"Invalid threshold rule '{rule_text}'. "
        "Expected forms like '<=0.1', '>0.5', '==0.2'."
    )


def _eval_threshold_rule(op, val, arr):
    if op == "<=":
        return arr <= val
    if op == ">=":
        return arr >= val
    if op == "<":
        return arr < val
    if op == ">":
        return arr > val
    if op == "==":
        return arr == val
    raise ValueError(f"Unsupported operator: {op}")


def _compact_mixing_group_label(label, disease_prs_name):
    if label == "All":
        return "All"

    prefix = f"P({disease_prs_name})"
    if not str(label).startswith(prefix):
        return label

    expr = str(label)[len(prefix) :].replace(" ", "")
    for op in ("<=", ">=", "<", ">", "=="):
        if expr.startswith(op):
            rhs = expr[len(op) :]
            op_tex = {
                "<=": r"\leq",
                ">=": r"\geq",
                "<": "<",
                ">": ">",
                "==": "=",
            }[op]
            return rf"${op_tex} {rhs}$"
    return label


def _build_mixing_groups_from_moe(
    prs_dataset,
    moe_model,
    analysis_id,
    disease_prs_name=None,
    partition_method="threshold",
    threshold=0.5,
    threshold_rules=None,
    n_quantiles=4,
):
    """
    Central utility for constructing disease-mixing groups from MoE probabilities.

    Returns a dict with:
    - disease_prs_name
    - disease_expert_idx
    - group_col
    - group_levels (without "All")
    - group_order (with "All")
    - group_masks (independent boolean masks by group label)
    - group_short_map / short_order (plot-friendly labels)
    """

    if disease_prs_name is None:
        disease_prs_name = _get_disease_prs_name(analysis_id)

    mapped_expert_names = [
        MODEL_NAME_MAP[analysis_id].get(prs_id, prs_id)
        for prs_id in moe_model.expert_cols
    ]
    if disease_prs_name not in mapped_expert_names:
        raise ValueError(
            f"Could not find disease PRS '{disease_prs_name}' among experts for {analysis_id}. "
            f"Available experts: {mapped_expert_names}"
        )

    disease_expert_idx = mapped_expert_names.index(disease_prs_name)

    mixing_proba = np.asarray(moe_model.predict_proba(prs_dataset), dtype=float)
    if mixing_proba.ndim != 2 or mixing_proba.shape[1] <= disease_expert_idx:
        raise ValueError(
            f"Unexpected MoE probability matrix shape: {mixing_proba.shape}. "
            f"Expected 2D with at least {disease_expert_idx + 1} columns."
        )

    prs_dataset.data["DiseasePRS_MixingProb"] = mixing_proba[:, disease_expert_idx]
    probs = prs_dataset.data["DiseasePRS_MixingProb"].values.astype(float)

    group_masks = {}

    if partition_method == "threshold":
        group_col = f"P({disease_prs_name}) threshold group"

        if threshold_rules is None:
            threshold = float(threshold)
            threshold_str = f"{threshold:g}"
            low_label = f"P({disease_prs_name})<={threshold_str}"
            high_label = f"P({disease_prs_name})>{threshold_str}"
            group_levels = [low_label, high_label]
            prs_dataset.data[group_col] = np.where(
                probs > threshold, high_label, low_label
            )
            group_masks = {
                low_label: probs <= threshold,
                high_label: probs > threshold,
            }
        else:
            if isinstance(threshold_rules, str):
                threshold_rules = [threshold_rules]
            if len(threshold_rules) == 0:
                raise ValueError("threshold_rules cannot be empty.")

            parsed_rules = [_parse_threshold_rule(r) for r in threshold_rules]
            group_levels = [
                f"P({disease_prs_name}){rule}" for _, _, rule in parsed_rules
            ]

            # Deterministic single assignment for models that require a unique group label.
            assigned = np.zeros(prs_dataset.N, dtype=bool)
            groups = np.full(prs_dataset.N, "", dtype=object)
            for (op, val, _), group_label in zip(parsed_rules, group_levels):
                msk = _eval_threshold_rule(op, val, probs)
                group_masks[group_label] = msk  # independent masks (can overlap)
                assign_msk = msk & (~assigned)
                groups[assign_msk] = group_label
                assigned |= assign_msk

            groups[~assigned] = f"P({disease_prs_name})_UNMATCHED"
            prs_dataset.data[group_col] = groups

    elif partition_method in {"quartile", "quantile"}:
        n_quantiles = int(n_quantiles)
        if n_quantiles < 2:
            raise ValueError(
                "n_quantiles must be >= 2 when using quartile/quantile partitioning."
            )

        group_col = f"P({disease_prs_name}) quantile"
        group_levels = [f"Q{i + 1}" for i in range(n_quantiles)]
        quant_groups = pd.qcut(
            prs_dataset.data["DiseasePRS_MixingProb"].rank(method="first"),
            q=n_quantiles,
            labels=group_levels,
        ).astype(str)
        prs_dataset.data[group_col] = quant_groups.values
        for g in group_levels:
            group_masks[g] = quant_groups.values == g
    else:
        raise ValueError(
            "partition_method must be one of {'threshold', 'quartile', 'quantile'}."
        )

    group_order = ["All"] + list(group_levels)
    if partition_method == "threshold":
        group_short_map = {
            g: _compact_mixing_group_label(g, disease_prs_name) for g in group_order
        }
    else:
        group_short_map = {g: g for g in group_order}
    short_order = [group_short_map[g] for g in group_order]

    return {
        "disease_prs_name": disease_prs_name,
        "disease_expert_idx": disease_expert_idx,
        "group_col": group_col,
        "group_levels": list(group_levels),
        "group_order": group_order,
        "group_masks": group_masks,
        "group_short_map": group_short_map,
        "short_order": short_order,
    }


def extract_disease_mixing_quartile_metrics(
    moe_model_name,
    analysis_id,
    test_biobank,
    metric="Nagelkerke_R2",
    disease_prs_name=None,
    dataset="test_data",
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    agg_features=None,
    partition_method="threshold",
    threshold=0.5,
    threshold_rules=None,
    n_quantiles=4,
):
    """
    Evaluate model performance stratified by disease-PRS mixing probabilities.

    Parameters
    ----------
    metric : str or list[str]
        Metric(s) to extract from evaluation
        (e.g. "Liability_R2", "AUROC", "AUPRC").

    Supports:
    - partition_method="threshold" (default): groups by probability threshold.
      Can use either:
      (a) single numeric `threshold`, or
      (b) ordered textual `threshold_rules` (e.g. ["<=0.05", "<=0.1", ">0.1"]).
      For threshold_rules, each rule is evaluated independently (cumulative/overlapping groups).
    - partition_method="quartile": groups by quantile bins (Q1..QK).
    Includes the All reference row.
    Note: Function name is kept for backward compatibility.
    Optionally appends group-wise means/SDs for dataset columns in `agg_features`.
    """

    dat = PRSDataset.from_pickle(
        f"data/harmonized_data/{analysis_id}/{test_biobank}/{dataset}.pkl"
    )

    # Restrict to European samples:
    dat.filter_samples(dat.data["Ancestry"] == "EUR")

    ukb_moe = MoEPRS.from_saved_model(
        f"data/trained_models/{analysis_id}/ukbb/train_data/{moe_model_name}.pkl"
    )

    grouping = _build_mixing_groups_from_moe(
        dat,
        ukb_moe,
        analysis_id=analysis_id,
        disease_prs_name=disease_prs_name,
        partition_method=partition_method,
        threshold=threshold,
        threshold_rules=threshold_rules,
        n_quantiles=n_quantiles,
    )
    disease_prs_name = grouping["disease_prs_name"]
    disease_expert_idx = grouping["disease_expert_idx"]
    group_col = grouping["group_col"]
    group_levels = grouping["group_levels"]

    group_masks = grouping["group_masks"]

    weighted_label = "Weighted PRS"
    weighted_excl_label = "Weighted PRS (exc. disease)"

    if isinstance(metric, str):
        requested_metrics = [metric]
    elif isinstance(metric, (list, tuple)):
        requested_metrics = [m for m in metric]
    else:
        raise ValueError("metric must be a string or a list/tuple of strings.")

    if len(requested_metrics) == 0:
        raise ValueError("At least one metric must be requested.")

    trained_models = {
        f"{moe_model_name} (UKB)": ukb_moe,
        weighted_label: GroupMeanWeightedPRS(ukb_moe, group_col),
        weighted_excl_label: GroupMeanWeightedPRS(
            ukb_moe, group_col, exclude_models=[disease_expert_idx]
        ),
    }

    def _to_plotting_wide(df_long):
        if df_long.empty:
            return df_long
        base_df = df_long.loc[df_long["metric_kind"] == "base"].copy()
        if base_df.empty:
            return pd.DataFrame()

        val_wide = (
            base_df.pivot_table(
                index=["model_id", "model_name", "eval_group", "eval_category", "n"],
                columns="metric",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"model_name": "PGS", "eval_group": "EvalGroup"})
        )

        se_wide = (
            base_df.pivot_table(
                index=["model_id", "eval_group"],
                columns="metric",
                values="se",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"eval_group": "EvalGroup"})
        )
        se_wide.columns = [
            c if c in ["model_id", "EvalGroup"] else f"{c}_err" for c in se_wide.columns
        ]

        out = val_wide.merge(se_wide, on=["model_id", "EvalGroup"], how="left")
        return out

    # Use stratified_evaluation for all partition modes.
    # For threshold_rules, create one binary categorical column per rule so each
    # rule can be evaluated independently while preserving overlap semantics.
    if partition_method == "threshold" and threshold_rules is not None:
        strat_cols = []
        for i, g in enumerate(group_levels):
            col = f"{group_col}__rule_{i + 1}"
            msk = np.asarray(group_masks[g], dtype=bool)
            dat.data[col] = np.where(msk, g, "__other__")
            strat_cols.append(col)
    else:
        strat_cols = [group_col]

    eval_df_long = stratified_evaluation(
        dat,
        trained_models=trained_models,
        metrics=requested_metrics,
        cat_group_cols=strat_cols,
        min_group_size=min_group_size,
    )

    # Keep only "All" and target mixing groups.
    eval_df_long = eval_df_long.loc[
        eval_df_long["eval_group"].isin(set(group_levels).union({"All"}))
    ].copy()

    missing_metrics = [
        m for m in requested_metrics if m not in set(eval_df_long["metric"].unique())
    ]
    if missing_metrics:
        raise ValueError(
            f"Requested metric(s) not found in evaluation output: {missing_metrics}"
        )

    out_df = _to_plotting_wide(eval_df_long)
    out_df["PGS"] = out_df["PGS"].map(lambda x: MODEL_NAME_MAP[analysis_id].get(x, x))

    if agg_features is None:
        agg_features = []
    elif isinstance(agg_features, str):
        agg_features = [agg_features]

    missing_features = [f for f in agg_features if f not in dat.data.columns]
    if missing_features:
        raise ValueError(
            f"agg_features not found in dataset columns: {missing_features}"
        )

    def summarize_group(df):
        out = {
            "Case_Proportion": float(
                pd.to_numeric(df[dat.phenotype_col], errors="coerce").mean()
            )
        }
        for f in agg_features:
            vals = pd.to_numeric(df[f], errors="coerce")
            out[f"{f}_mean"] = float(vals.mean())
            out[f"{f}_sd"] = float(vals.std(ddof=0))
        return out

    group_rows = [{"EvalGroup": "All", **summarize_group(dat.data)}]
    if partition_method == "threshold" and threshold_rules is not None:
        for group_label in group_levels:
            msk = np.asarray(group_masks[group_label], dtype=bool)
            group_rows.append(
                {"EvalGroup": group_label, **summarize_group(dat.data.loc[msk])}
            )
    else:
        for g, gdf in dat.data.groupby(group_col, sort=False):
            group_rows.append({"EvalGroup": str(g), **summarize_group(gdf)})
    group_summary = pd.DataFrame(group_rows)
    out_df = out_df.merge(group_summary, on="EvalGroup", how="left")

    out_df["Mixing_Group"] = out_df["EvalGroup"]
    out_df["Quartile"] = out_df["Mixing_Group"]
    out_df["Disease_PRS"] = disease_prs_name
    out_df["AnalysisID"] = analysis_id
    out_df["Test biobank"] = test_biobank

    group_order = grouping["group_order"]
    out_df["Mixing_Group"] = pd.Categorical(
        out_df["Mixing_Group"], categories=group_order, ordered=True
    )
    out_df["Quartile"] = pd.Categorical(
        out_df["Quartile"], categories=group_order, ordered=True
    )
    out_df = out_df.sort_values(["Mixing_Group", "PGS"]).reset_index(drop=True)

    return out_df


def plot_binary_mixing_group_panels(
    moe_model_name,
    analysis_id,
    test_biobank="ukbb",
    dataset="test_data",
    metric="AUROC",
    incremental_metric="Nagelkerke_R2",
    disease_prs_name=None,
    output_file=None,
    partition_method="threshold",
    threshold=0.5,
    threshold_rules=None,
    n_quantiles=4,
    include_prs_distribution_panel=False,
):
    """
    Plot one analysis at a time:
    1x4 (or 1x5) panels comparing partitioned mixing-weight groups for
    male proportion, mean age (with SD), case prevalence, and model performance.
    If `metric` is not `Liability_R2`, an additional Liability R² panel is added.
    Accuracy panels show MoEPRS, disease PRS, Weighted PRS, and
    Weighted PRS (exc. disease).
    For threshold partitioning, you can pass either:
    - threshold=<float> (binary split), or
    - threshold_rules=[...], e.g. ["<=0.05", "<=0.1", "<=0.25", "<=0.5", ">0.5"].
    """

    if output_file is None:
        output_file = (
            f"figures/section_4/mixing_group_summary_{analysis_id}_{test_biobank}.png"
        )

    requested_metrics = (
        [metric] if metric == incremental_metric else [metric, incremental_metric]
    )

    df = extract_disease_mixing_quartile_metrics(
        moe_model_name=moe_model_name,
        analysis_id=analysis_id,
        test_biobank=test_biobank,
        metric=requested_metrics,
        disease_prs_name=disease_prs_name,
        dataset=dataset,
        agg_features=["Sex", "Age"],
        partition_method=partition_method,
        threshold=threshold,
        threshold_rules=threshold_rules,
        n_quantiles=n_quantiles,
    )

    resolved_disease_prs = df["Disease_PRS"].iloc[0]

    if isinstance(df["Mixing_Group"].dtype, pd.CategoricalDtype):
        group_order = list(df["Mixing_Group"].cat.categories)
    else:
        non_all = sorted([g for g in df["EvalGroup"].unique() if g != "All"])
        group_order = ["All"] + non_all
    group_order = [g for g in group_order if g in set(df["EvalGroup"].astype(str))]

    if partition_method == "threshold":

        def compact_threshold_label(label):
            if label == "All":
                return "All"
            prefix = f"P({resolved_disease_prs})"
            if not str(label).startswith(prefix):
                return label

            expr = str(label)[len(prefix) :].replace(" ", "")
            for op in ("<=", ">=", "<", ">", "=="):
                if expr.startswith(op):
                    rhs = expr[len(op) :]
                    op_tex = {
                        "<=": r"\leq",
                        ">=": r"\geq",
                        "<": "<",
                        ">": ">",
                        "==": "=",
                    }[op]
                    return rf"${op_tex} {rhs}$"
            return label

        group_short_map = {g: compact_threshold_label(g) for g in group_order}
        short_order = [group_short_map[g] for g in group_order]
    else:
        group_short_map = {g: g for g in group_order}
        short_order = [group_short_map[g] for g in group_order]

    # Use a single sequential gradient scheme across threshold and quartile scenarios.
    non_all_groups = [g for g in short_order if g != "All"]
    non_all_colors = sns.color_palette("viridis", max(len(non_all_groups), 1))
    color_map = {"All": "#B3B3B3"}
    for g, c in zip(non_all_groups, non_all_colors):
        color_map[g] = c

    group_cols = ["EvalGroup", "Sex_mean", "Age_mean", "Case_Proportion"]
    if "Age_sd" in df.columns:
        group_cols.append("Age_sd")
    plot_df = (
        df.loc[df["EvalGroup"].isin(group_order), group_cols]
        .drop_duplicates(subset=["EvalGroup"])
        .assign(GroupShort=lambda x: x["EvalGroup"].map(group_short_map))
    )

    metric_specs = [
        ("Sex_mean", "Proportion male"),
        ("Age_mean", "Recruitment age"),
        ("Case_Proportion", "Case prevalence"),
    ]

    n_panels = 3 + len(requested_metrics)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.5 * n_panels, 3.4),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.flatten()
    for ax, (metric_col, metric_title) in zip(axes[:3], metric_specs):
        metric_df = plot_df[["GroupShort", metric_col]].dropna(subset=[metric_col])
        if metric_df.empty:
            ax.set_axis_off()
            continue

        if metric_col == "Age_mean":
            y_map = {g: i for i, g in enumerate(short_order)}
            age_df = (
                plot_df[["GroupShort", "Age_mean", "Age_sd"]].drop_duplicates(
                    subset=["GroupShort"]
                )
                if "Age_sd" in plot_df.columns
                else plot_df[["GroupShort", "Age_mean"]]
                .drop_duplicates(subset=["GroupShort"])
                .assign(Age_sd=np.nan)
            )

            for g in short_order:
                row = age_df.loc[age_df["GroupShort"] == g]
                if row.empty:
                    continue
                y_pos = y_map[g]
                x_val = float(row["Age_mean"].iloc[0])
                x_sd = row["Age_sd"].iloc[0]
                g_color = color_map.get(g, "#4C78A8")
                if pd.isna(x_sd):
                    ax.plot(x_val, y_pos, "o", color=g_color, markersize=6)
                else:
                    ax.errorbar(
                        x_val,
                        y_pos,
                        xerr=float(x_sd),
                        fmt="o",
                        color=g_color,
                        ecolor=g_color,
                        lw=1.5,
                        capsize=0,
                        markersize=6,
                    )
            ax.set_yticks(range(len(short_order)))
            ax.set_yticklabels(short_order)
        else:
            if metric_col == "Sex_mean":
                # Reference parity line in the background.
                ax.axvline(0.5, ls="--", lw=1, color="grey", alpha=0.8, zorder=0)
            sns.barplot(
                data=metric_df,
                y="GroupShort",
                x=metric_col,
                hue="GroupShort",
                order=short_order,
                hue_order=short_order,
                palette=color_map,
                dodge=False,
                legend=False,
                ax=ax,
            )
        ax.set_title(metric_title)
        ax.set_ylabel("")
        ax.set_xlabel("")

        if metric_col == "Age_mean":
            ax.set_xlim(left=40)
        if metric_col == "Sex_mean":
            ax.set_xlim(0.0, 1.0)

    moe_label = f"{moe_model_name} (UKB)"
    weighted_label = "Weighted PRS"
    weighted_excl_label = "Weighted PRS (exc. disease)"
    keep_models = [moe_label, resolved_disease_prs, weighted_label, weighted_excl_label]
    disease_color = assign_models_consistent_colors([resolved_disease_prs])[
        resolved_disease_prs
    ]
    model_palette = {
        "MoEPRS": "#375E97",
        resolved_disease_prs: disease_color,
        weighted_label: "#111111",
        weighted_excl_label: "#6F6F6F",
    }

    def plot_accuracy_panel(acc_ax, source_df, metric_name, show_legend=True):
        metric_err_col = f"{metric_name}_err"
        acc_cols = ["EvalGroup", "PGS", metric_name]
        if metric_err_col in source_df.columns:
            acc_cols.append(metric_err_col)

        acc_df = source_df.loc[
            source_df["EvalGroup"].isin(group_order)
            & source_df["PGS"].isin(keep_models)
            & source_df[metric_name].notna(),
            acc_cols,
        ].drop_duplicates(subset=["EvalGroup", "PGS"])

        if acc_df.empty:
            acc_ax.set_axis_off()
            return

        acc_df["GroupShort"] = acc_df["EvalGroup"].map(group_short_map)
        acc_df["Model"] = acc_df["PGS"].replace({moe_label: "MoEPRS"})

        model_order = [
            m
            for m in [
                "MoEPRS",
                resolved_disease_prs,
                weighted_label,
                weighted_excl_label,
            ]
            if m in set(acc_df["Model"])
        ]
        if len(model_order) == 0:
            acc_ax.set_axis_off()
            return

        sns.barplot(
            data=acc_df,
            y="GroupShort",
            x=metric_name,
            hue="Model",
            order=short_order,
            hue_order=model_order,
            palette=model_palette,
            errorbar=None,
            ax=acc_ax,
        )

        if metric_err_col in acc_df.columns and acc_df[metric_err_col].notna().any():
            num_hues = len(model_order)
            dodge_width = 0.8
            bar_height = dodge_width / max(num_hues, 1)
            dodge_offsets = np.linspace(
                -dodge_width / 2 + bar_height / 2,
                dodge_width / 2 - bar_height / 2,
                max(num_hues, 1),
            )
            y_base = {g: i for i, g in enumerate(short_order)}

            for i_h, hval in enumerate(model_order):
                for gval in short_order:
                    row = acc_df.loc[
                        (acc_df["Model"] == hval) & (acc_df["GroupShort"] == gval)
                    ]
                    if row.empty:
                        continue
                    x_err = row[metric_err_col].iloc[0]
                    if pd.isna(x_err):
                        continue
                    x_val = float(row[metric_name].iloc[0])
                    y_pos = y_base[gval] + dodge_offsets[i_h]
                    acc_ax.errorbar(
                        x_val,
                        y_pos,
                        xerr=float(x_err),
                        fmt="none",
                        ecolor="black",
                        lw=1.0,
                        capsize=0,
                    )

        acc_ax.set_title(METRIC_NAME_MAP[metric_name])
        acc_ax.set_ylabel("")
        acc_ax.set_xlabel("")
        if show_legend:
            acc_ax.legend(loc="best", frameon=False, fontsize=7, title=None)
        else:
            if acc_ax.get_legend() is not None:
                acc_ax.get_legend().remove()

        if metric_name == "AUROC":
            acc_ax.set_xlim(0.5, 1.0)

    for i, metric_name in enumerate(requested_metrics):
        plot_accuracy_panel(
            axes[3 + i],
            df,
            metric_name,
            show_legend=(i == 0),
        )

    bb_short = BIOBANK_NAME_MAP_SHORT.get(test_biobank.lower(), test_biobank.upper())
    fig.suptitle(
        f"{ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)} ({bb_short})"
    )
    if partition_method in {"quartile", "quantile"}:
        y_label = f"Quartiles of\nP({resolved_disease_prs}) mixing weight"
    else:
        y_label = f"Groups of\nP({resolved_disease_prs}) mixing weight"

    fig.supylabel(y_label)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_mixing_quartile_metric_panels_across_phenotypes(
    moe_model_name,
    analysis_ids=None,
    test_biobank="ukbb",
    dataset="test_data",
    model_biobank="ukbb",
    model_dataset="train_data",
    fold=None,
    phenotype_order=None,
    keep_ancestry=None,
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    panel_order=None,
    figsize=None,
    output_file=None,
):
    """Plot mean and SE of phenotype-wide summaries across held-out CV folds.

    Quartiles are recalculated within each fold from that fold's held-out samples
    and matching fold-trained MoE model. Corresponding fold summaries are then
    averaged, with SE equal to the sample SD across folds divided by sqrt(K).
    Supplying ``fold`` explicitly retains a single-fold diagnostic mode.
    """

    if analysis_ids is None:
        analysis_ids = [
            analysis_id
            for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP
            if ANALYSIS_TO_TABLE_MAP.get(analysis_id) == "multitrait_prs_table"
        ]
    elif isinstance(analysis_ids, str):
        analysis_ids = [analysis_ids]
    else:
        analysis_ids = list(analysis_ids)
    if len(analysis_ids) == 0:
        raise ValueError("analysis_ids must contain at least one analysis.")

    if output_file is None:
        output_file = (
            "figures/section_4/"
            f"mixing_quartile_metric_panels_all_phenotypes_{test_biobank}.pdf"
        )

    metric_rows = []
    phenotype_by_analysis = {}

    dataset_filename = dataset if str(dataset).endswith(".pkl") else f"{dataset}.pkl"

    for analysis_id in analysis_ids:
        if fold is None:
            fold_data_paths = glob.glob(
                osp.join(
                    "data/harmonized_data",
                    analysis_id,
                    test_biobank,
                    "fold_*",
                    dataset_filename,
                )
            )
            analysis_folds = sorted(
                {
                    osp.basename(osp.dirname(path))
                    for path in fold_data_paths
                },
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )
        else:
            analysis_folds = [_normalize_fold_name(fold)]

        if not analysis_folds:
            print(
                f"> Skipping {analysis_id}: no held-out "
                f"{dataset_filename} folds found.",
                file=sys.stderr,
            )
            continue

        for fold_name in analysis_folds:
            try:
                data_path = _resolve_harmonized_dataset_path(
                    analysis_id,
                    test_biobank,
                    dataset,
                    fold=fold_name,
                )
                moe_path = _resolve_trained_model_path(
                    analysis_id,
                    model_biobank,
                    model_dataset,
                    moe_model_name,
                    fold=fold_name,
                )
                covariates_path = _resolve_trained_model_path(
                    analysis_id,
                    model_biobank,
                    model_dataset,
                    "Covariates",
                    fold=fold_name,
                )
                for model_path in (moe_path, covariates_path):
                    if fold_name not in osp.normpath(model_path).split(osp.sep):
                        raise FileNotFoundError(
                            f"No fold-specific model found for {analysis_id}/"
                            f"{model_biobank}/{fold_name}: {model_path}"
                        )

                dat = PRSDataset.from_pickle(data_path)
                if keep_ancestry is not None:
                    keep_ancestry_vals = (
                        [keep_ancestry]
                        if isinstance(keep_ancestry, str)
                        else keep_ancestry
                    )
                    dat.filter_samples(
                        dat.data["Ancestry"].isin(keep_ancestry_vals)
                    )

                moe_model = MoEPRS.from_saved_model(moe_path)
                grouping = _build_mixing_groups_from_moe(
                    dat,
                    moe_model,
                    analysis_id=analysis_id,
                    partition_method="quartile",
                    n_quantiles=4,
                )
                disease_prs_name = grouping["disease_prs_name"]
                disease_prs_id = moe_model.expert_cols[
                    grouping["disease_expert_idx"]
                ]
                disease_model_name = f"{disease_prs_id}-covariates"
                disease_model_path = _resolve_trained_model_path(
                    analysis_id,
                    model_biobank,
                    model_dataset,
                    disease_model_name,
                    fold=fold_name,
                )
                if fold_name not in osp.normpath(disease_model_path).split(
                    osp.sep
                ):
                    raise FileNotFoundError(
                        f"No fold-specific disease model found for {analysis_id}/"
                        f"{model_biobank}/{fold_name}: {disease_model_path}"
                    )

                moe_model_id = _trained_model_id_from_path(moe_path, analysis_id)
                disease_model_id = _trained_model_id_from_path(
                    disease_model_path, analysis_id
                )
                covariates_model_id = _trained_model_id_from_path(
                    covariates_path, analysis_id
                )
                trained_models = {
                    moe_model_id: moe_model,
                    disease_model_id: MultiPRS.from_saved_model(disease_model_path),
                    covariates_model_id: MultiPRS.from_saved_model(covariates_path),
                }
                model_catalog = pd.DataFrame(
                    [_parse_model_id(model_id) for model_id in trained_models]
                )

                eval_df = stratified_evaluation(
                    dat,
                    trained_models=trained_models,
                    model_catalog=model_catalog,
                    test_biobank=test_biobank,
                    metrics=["Nagelkerke_R2", "AUROC"],
                    cat_group_cols=[grouping["group_col"]],
                    evaluate_base_models=False,
                    min_group_size=min_group_size,
                )
            except Exception as e:
                print(
                    f"> Skipping {analysis_id}/{fold_name} quartile metrics: {e}",
                    file=sys.stderr,
                )
                continue

            phenotype = _shorten_disease_label(
                ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            )
            phenotype_by_analysis[analysis_id] = phenotype
            group_levels = grouping["group_levels"]
            model_label_map = {
                moe_model_id: "MoEPRS",
                disease_model_id: "Disease-specific PRS",
            }
            all_group_levels = ["All"] + list(group_levels)

            nagelkerke_df = eval_df.loc[
                (eval_df["metric"] == "Nagelkerke_R2")
                & (eval_df["metric_kind"] == "incremental_vs_ref")
                & (eval_df["ref_model_biobank"] == model_biobank)
                & (eval_df["model_id"].isin(model_label_map))
                & (eval_df["eval_group"].isin(all_group_levels))
                & (eval_df["value"].notna())
            ].copy()
            auroc_df = eval_df.loc[
                (eval_df["metric"] == "AUROC")
                & (eval_df["metric_kind"] == "base")
                & (eval_df["model_id"].isin(model_label_map))
                & (eval_df["eval_group"].isin(all_group_levels))
                & (eval_df["value"].notna())
            ].copy()

            for panel_name, source_df in (
                ("Accuracy (Nagelkerke $R^2$)", nagelkerke_df),
                ("Accuracy (AUROC)", auroc_df),
            ):
                for _, row in source_df.iterrows():
                    metric_rows.append(
                        {
                            "AnalysisID": analysis_id,
                            "Phenotype": phenotype,
                            "Quartile": row["eval_group"],
                            "Panel": panel_name,
                            "Model": model_label_map[row["model_id"]],
                            "Value": float(row["value"]),
                            "Disease_PRS": disease_prs_name,
                            "Fold": fold_name,
                        }
                    )

            for quartile in all_group_levels:
                if quartile == "All":
                    mask = np.ones(dat.N, dtype=bool)
                else:
                    mask = np.asarray(grouping["group_masks"][quartile], dtype=bool)
                group_df = dat.data.loc[mask]
                phenotype_vals = pd.to_numeric(
                    group_df[dat.phenotype_col], errors="coerce"
                )
                sex_vals = pd.to_numeric(group_df["Sex"], errors="coerce")
                age_vals = pd.to_numeric(group_df["Age"], errors="coerce")
                for panel_name, value in (
                    ("Proportion male", sex_vals.mean()),
                    ("Proportion cases", phenotype_vals.mean()),
                    ("Mean age at recruitment", age_vals.mean()),
                ):
                    metric_rows.append(
                        {
                            "AnalysisID": analysis_id,
                            "Phenotype": phenotype,
                            "Quartile": quartile,
                            "Panel": panel_name,
                            "Model": None,
                            "Value": float(value),
                            "Disease_PRS": disease_prs_name,
                            "Fold": fold_name,
                        }
                    )

    fold_df = pd.DataFrame(metric_rows)
    if fold_df.empty:
        raise ValueError("No mixing-quartile metrics were available to plot.")

    summary_cols = [
        "AnalysisID",
        "Phenotype",
        "Quartile",
        "Panel",
        "Model",
        "Disease_PRS",
    ]

    def fold_standard_error(values):
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) < 2:
            return np.nan
        return float(values.std(ddof=1) / np.sqrt(len(values)))

    plot_df = (
        fold_df.groupby(summary_cols, dropna=False, observed=True)
        .agg(
            Value=("Value", "mean"),
            SE=("Value", fold_standard_error),
            n_folds=("Fold", "nunique"),
        )
        .reset_index()
    )

    if phenotype_order is None:
        plot_order = [
            phenotype_by_analysis[a]
            for a in analysis_ids
            if a in phenotype_by_analysis
        ]
    else:
        requested_order = [_shorten_disease_label(p) for p in phenotype_order]
        present = set(plot_df["Phenotype"])
        plot_order = [p for p in requested_order if p in present]
        plot_order.extend(
            p
            for p in [
                phenotype_by_analysis[a]
                for a in analysis_ids
                if a in phenotype_by_analysis
            ]
            if p not in plot_order
        )
    plot_order = list(dict.fromkeys(plot_order))
    if len(plot_order) == 0:
        plot_order = list(dict.fromkeys(plot_df["Phenotype"].dropna()))

    use_default_panel_order = panel_order is None
    if use_default_panel_order:
        panel_order = [
            "Accuracy (Nagelkerke $R^2$)",
            "Accuracy (AUROC)",
            "Proportion cases",
            "Proportion male",
            "Mean age at recruitment",
        ]
    else:
        panel_order = list(panel_order)

    available_panels = set(plot_df["Panel"].unique())
    missing_panels = [p for p in panel_order if p not in available_panels]
    if missing_panels:
        raise ValueError(f"Requested panel(s) not available: {missing_panels}")

    quartile_order = ["All", "Q1", "Q2", "Q3", "Q4"]
    model_order = ["Disease-specific PRS", "MoEPRS"]
    quartile_palette = {"All": "#9E9E9E"}
    quartile_palette.update(
        dict(zip(quartile_order[1:], sns.color_palette("viridis", 4)))
    )
    phenotype_spacing = 2.2
    phenotype_half_width = phenotype_spacing / 2
    quartile_offsets = dict(zip(quartile_order, np.linspace(-0.84, 0.84, 5)))
    model_offsets = {
        "Disease-specific PRS": -0.11,
        "MoEPRS": 0.11,
    }
    model_markers = {
        "Disease-specific PRS": "o",
        "MoEPRS": "^",
    }
    label_fontsize = 9 if use_default_panel_order else 11
    tick_fontsize = 9 if use_default_panel_order else 10
    title_fontsize = 12 if use_default_panel_order else 13
    legend_fontsize = 8 if use_default_panel_order else 9
    legend_title_fontsize = 8 if use_default_panel_order else 9
    panel_label_map = {
        "Mean age at recruitment": "Mean\nage at recruitment",
    }

    if figsize is None:
        fig_width = max(11.5, 1.55 * len(plot_order))
        fig_height = (
            10.5 if use_default_panel_order else max(2.4 * len(panel_order), 4.0)
        )
    else:
        fig_width, fig_height = figsize
    fig, axes = plt.subplots(
        len(panel_order),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
    )
    x_base = {
        phenotype: i * phenotype_spacing
        for i, phenotype in enumerate(plot_order)
    }

    for ax, panel_name in zip(axes, panel_order):
        for i, phenotype in enumerate(plot_order):
            if i % 2 == 0:
                x_mid = x_base[phenotype]
                ax.axvspan(
                    x_mid - phenotype_half_width,
                    x_mid + phenotype_half_width,
                    color="#F2F2F2",
                    alpha=0.65,
                    zorder=0,
                )

        panel_df = plot_df.loc[plot_df["Panel"] == panel_name].copy()
        if panel_df.empty:
            ax.set_axis_off()
            continue

        if panel_name.startswith("Accuracy"):
            for quartile in quartile_order:
                qdf = panel_df.loc[panel_df["Quartile"] == quartile]
                for model_name in model_order:
                    mdf = qdf.loc[qdf["Model"] == model_name]
                    if mdf.empty:
                        continue
                    xs = [
                        x_base[p]
                        + quartile_offsets[quartile]
                        + model_offsets[model_name]
                        for p in mdf["Phenotype"]
                    ]
                    color = quartile_palette[quartile]
                    ax.scatter(
                        xs,
                        mdf["Value"],
                        marker=model_markers[model_name],
                        s=30 if use_default_panel_order else 42,
                        color=color,
                        edgecolor="#222222",
                        linewidth=0.4,
                        zorder=4,
                    )
        else:
            for quartile in quartile_order:
                qdf = panel_df.loc[panel_df["Quartile"] == quartile]
                if qdf.empty:
                    continue
                xs = [
                    x_base[p] + quartile_offsets[quartile]
                    for p in qdf["Phenotype"]
                ]
                color = quartile_palette[quartile]
                ax.scatter(
                    xs,
                    qdf["Value"],
                    marker="o",
                    s=32 if use_default_panel_order else 42,
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.4,
                    zorder=4,
                )

        if panel_name == "Accuracy (AUROC)":
            ax.axhline(0.5, ls="--", lw=0.8, color="#888888", alpha=0.7, zorder=0)
        if panel_name == "Proportion male":
            ax.axhline(0.5, ls="--", lw=0.8, color="#888888", alpha=0.7, zorder=0)
            ax.set_ylim(0.0, 1.0)
        if panel_name == "Proportion cases":
            ax.set_ylim(bottom=0.0)

        ax.set_ylabel(
            panel_label_map.get(panel_name, panel_name),
            fontsize=label_fontsize,
            labelpad=12,
        )
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(
            -phenotype_half_width,
            x_base[plot_order[-1]] + phenotype_half_width,
        )
        sns.despine(ax=ax)

    axes[-1].set_xticks([x_base[p] for p in plot_order])
    axes[-1].set_xticklabels(plot_order, rotation=30, ha="right", fontsize=tick_fontsize)
    axes[-1].set_xlabel("Phenotypes", fontsize=label_fontsize)

    quartile_labels = {
        "All": "All",
        "Q1": "Q1 (lowest)",
        "Q2": "Q2",
        "Q3": "Q3",
        "Q4": "Q4 (highest)",
    }
    quartile_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=quartile_palette[q],
            label=quartile_labels[q],
            markersize=6,
        )
        for q in quartile_order
    ]
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=model_markers[m],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#222222",
            label=m,
            markersize=6,
        )
        for m in model_order
    ]
    quartile_legend = fig.legend(
        handles=quartile_handles,
        loc="center left",
        bbox_to_anchor=(0.875 if use_default_panel_order else 0.885, 0.59),
        frameon=False,
        fontsize=legend_fontsize,
        title="Mixing weight quartile",
        title_fontsize=legend_title_fontsize,
    )
    fig.add_artist(quartile_legend)
    fig.legend(
        handles=model_handles,
        loc="center left",
        bbox_to_anchor=(0.875 if use_default_panel_order else 0.885, 0.38),
        frameon=False,
        fontsize=legend_fontsize,
        title="PRS model",
        title_fontsize=legend_title_fontsize,
    )
    bb_short = BIOBANK_NAME_MAP_SHORT.get(test_biobank, test_biobank.upper())
    fig.suptitle(
        f"Mixing-weight quartile summaries across {bb_short} held-out CV folds",
        fontsize=title_fontsize,
        y=0.985,
    )
    if use_default_panel_order:
        fig.subplots_adjust(left=0.09, right=0.85, bottom=0.13, top=0.94, hspace=0.28)
    else:
        fig.subplots_adjust(left=0.14, right=0.865, bottom=0.16, top=0.91, hspace=0.42)

    output_dir = osp.dirname(output_file)
    if output_dir:
        makedir(output_dir)
    fig.savefig(output_file, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return plot_df


def plot_prs_covariate_accuracy_by_mixing_group(
    analysis_id,
    prs_names,
    stratification_prs_name,
    stratification_strategy,
    metric,
    moe_model_name="MoE-GS-prs-gating",
    test_biobank="ukbb",
    dataset="test_data",
    model_biobank="ukbb",
    model_dataset="train_data",
    keep_ancestry=("EUR",),
    threshold=0.5,
    threshold_rules=None,
    n_quantiles=4,
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    output_file=None,
):
    """Compare saved PRS+covariates models across MoE mixing-weight strata.

    The MoE model is used only to define strata for ``stratification_prs_name``.
    Accuracy is evaluated exclusively from the full predictions of the saved
    ``<PGS>-covariates.pkl`` models listed in ``prs_names``; neither MoEPRS nor
    PRS-only predictions are evaluated.
    """

    if isinstance(prs_names, str):
        prs_names = [prs_names]
    else:
        prs_names = list(prs_names)
    if len(prs_names) == 0:
        raise ValueError("prs_names must contain at least one PRS.")

    if output_file is None:
        strat_label = str(stratification_prs_name).replace(" ", "_")
        output_file = (
            "figures/section_4/"
            f"accuracy_{analysis_id}_{strat_label}_mixing_groups_{test_biobank}.png"
        )

    dat = PRSDataset.from_pickle(
        f"data/harmonized_data/{analysis_id}/{test_biobank}/{dataset}.pkl"
    )
    if keep_ancestry is not None:
        if isinstance(keep_ancestry, str):
            keep_ancestry = [keep_ancestry]
        dat.filter_samples(dat.data["Ancestry"].isin(keep_ancestry))

    moe_path = (
        f"data/trained_models/{analysis_id}/{model_biobank}/{model_dataset}/"
        f"{moe_model_name}.pkl"
    )
    moe_model = MoEPRS.from_saved_model(moe_path)

    expert_name_map = MODEL_NAME_MAP.get(analysis_id, {})
    expert_name_by_id = {
        prs_id: expert_name_map.get(prs_id, prs_id)
        for prs_id in moe_model.expert_cols
    }
    expert_id_by_name = {name: prs_id for prs_id, name in expert_name_by_id.items()}

    def resolve_prs(prs_name):
        if prs_name in expert_name_by_id:
            return prs_name, expert_name_by_id[prs_name]
        if prs_name in expert_id_by_name:
            return expert_id_by_name[prs_name], prs_name
        raise ValueError(
            f"PRS '{prs_name}' is unavailable for {analysis_id}. Available scores: "
            f"{sorted(expert_id_by_name)}"
        )

    resolved_prs = [resolve_prs(prs_name) for prs_name in prs_names]
    if len({prs_id for prs_id, _ in resolved_prs}) != len(resolved_prs):
        raise ValueError("prs_names contains duplicate scores after name resolution.")
    _, stratification_prs_label = resolve_prs(stratification_prs_name)

    grouping = _build_mixing_groups_from_moe(
        dat,
        moe_model,
        analysis_id=analysis_id,
        disease_prs_name=stratification_prs_label,
        partition_method=stratification_strategy,
        threshold=threshold,
        threshold_rules=threshold_rules,
        n_quantiles=n_quantiles,
    )
    group_col = grouping["group_col"]
    group_levels = grouping["group_levels"]
    group_order = grouping["group_order"]
    group_masks = grouping["group_masks"]
    group_short_map = grouping["group_short_map"]
    short_order = grouping["short_order"]

    trained_models = {}
    model_labels = {}
    for prs_id, prs_label in resolved_prs:
        model_id = f"{prs_id}-covariates"
        model_path = (
            f"data/trained_models/{analysis_id}/{model_biobank}/{model_dataset}/"
            f"{model_id}.pkl"
        )
        if not osp.exists(model_path):
            raise FileNotFoundError(
                f"Could not find the saved PRS+covariates model: {model_path}"
            )
        trained_models[model_id] = MultiPRS.from_saved_model(model_path)
        model_labels[model_id] = f"{prs_label} + covariates"

    if stratification_strategy == "threshold" and threshold_rules is not None:
        strat_cols = []
        for i, group_label in enumerate(group_levels):
            col = f"{group_col}__rule_{i + 1}"
            mask = np.asarray(group_masks[group_label], dtype=bool)
            dat.data[col] = np.where(mask, group_label, "__other__")
            strat_cols.append(col)
    else:
        strat_cols = [group_col]

    eval_df = stratified_evaluation(
        dat,
        trained_models=trained_models,
        metrics=[metric],
        cat_group_cols=strat_cols,
        evaluate_base_models=False,
        min_group_size=min_group_size,
    )
    eval_df = eval_df.loc[
        (eval_df["metric"] == metric)
        & (eval_df["metric_kind"] == "base")
        & (eval_df["model_id"].isin(trained_models))
        & (eval_df["eval_group"].isin(set(group_order)))
        & (eval_df["value"].notna())
    ].copy()
    if eval_df.empty:
        raise ValueError("No valid PRS+covariates accuracy estimates were produced.")

    eval_df["Model"] = eval_df["model_id"].map(model_labels)
    eval_df["Group"] = eval_df["eval_group"].map(group_short_map)
    model_order = [model_labels[f"{prs_id}-covariates"] for prs_id, _ in resolved_prs]
    prs_palette = assign_models_consistent_colors(
        [prs_label for _, prs_label in resolved_prs]
    )
    model_palette = {
        model_labels[f"{prs_id}-covariates"]: prs_palette[prs_label]
        for prs_id, prs_label in resolved_prs
    }

    fig_height = max(3.0, 0.55 * len(short_order) + 1.4)
    fig, ax = plt.subplots(figsize=(6.4, fig_height), constrained_layout=True)
    sns.barplot(
        data=eval_df,
        y="Group",
        x="value",
        hue="Model",
        order=short_order,
        hue_order=model_order,
        palette=model_palette,
        errorbar=None,
        ax=ax,
    )

    if eval_df["se"].notna().any():
        num_models = len(model_order)
        dodge_width = 0.8
        bar_height = dodge_width / num_models
        offsets = np.linspace(
            -dodge_width / 2 + bar_height / 2,
            dodge_width / 2 - bar_height / 2,
            num_models,
        )
        y_base = {group: i for i, group in enumerate(short_order)}
        for model_idx, model_label in enumerate(model_order):
            for group_label in short_order:
                row = eval_df.loc[
                    (eval_df["Model"] == model_label)
                    & (eval_df["Group"] == group_label)
                ]
                if row.empty or pd.isna(row["se"].iloc[0]):
                    continue
                ax.errorbar(
                    float(row["value"].iloc[0]),
                    y_base[group_label] + offsets[model_idx],
                    xerr=float(row["se"].iloc[0]),
                    fmt="none",
                    ecolor="black",
                    lw=1.0,
                    capsize=0,
                )

    metric_label = METRIC_NAME_MAP.get(metric, metric)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("")
    ax.set_title(
        f"{ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)} "
        f"({BIOBANK_NAME_MAP_SHORT.get(test_biobank, test_biobank.upper())})"
    )
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(title=None, frameon=False)
    if metric == "AUROC":
        ax.set_xlim(0.5, 1.0)

    if stratification_strategy in {"quartile", "quantile"}:
        y_label = f"Quantiles of P({stratification_prs_label}) mixing weight"
    else:
        y_label = f"Groups of P({stratification_prs_label}) mixing weight"
    fig.supylabel(y_label)

    output_dir = osp.dirname(output_file)
    if output_dir:
        makedir(output_dir)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    return eval_df


def plot_disease_prs_mixing_weights_across_phenotypes(
    moe_model_name,
    analysis_ids=None,
    comparison_analysis_ids=None,
    test_biobank="ukbb",
    dataset="full_data",
    model_biobank="ukbb",
    model_dataset="train_data",
    fold=DEFAULT_SECTION4_PLOTTING_FOLD,
    phenotype_order=None,
    max_strip_points=1200,
    random_state=42,
    title=None,
    figsize=None,
    output_file=None,
):
    """Plot disease-specific PRS mixing weights across disease phenotypes.

    Violins use every finite mixing weight in each full cohort. The overlaid
    strip points are reproducibly subsampled to keep dense biobank data legible.
    The full dataset is evaluated with one model from the selected reference fold.
    If comparison_analysis_ids is provided, the figure overlays those analyses
    as the standard-analysis reference beside the requested analyses.
    """

    if analysis_ids is None:
        analysis_ids = [
            analysis_id
            for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP
            if ANALYSIS_TO_TABLE_MAP.get(analysis_id) == "multitrait_prs_table"
        ]
    elif isinstance(analysis_ids, str):
        analysis_ids = [analysis_ids]
    else:
        analysis_ids = list(analysis_ids)
    if len(analysis_ids) == 0:
        raise ValueError("analysis_ids must contain at least one analysis.")

    if comparison_analysis_ids is not None:
        if isinstance(comparison_analysis_ids, str):
            comparison_analysis_ids = [comparison_analysis_ids]
        else:
            comparison_analysis_ids = list(comparison_analysis_ids)
        if len(comparison_analysis_ids) == 0:
            raise ValueError(
                "comparison_analysis_ids must contain at least one analysis."
            )

    if max_strip_points is not None:
        max_strip_points = int(max_strip_points)
        if max_strip_points < 1:
            raise ValueError("max_strip_points must be positive or None.")

    if output_file is None:
        output_file = (
            "figures/section_4/"
            f"mixing_weight_disease_prs_all_phenotypes_{test_biobank}.pdf"
        )

    weight_frames = []
    phenotype_to_prs = {}
    analysis_to_phenotype = {}

    if comparison_analysis_ids is None:
        analysis_sets = [(None, analysis_ids)]
    else:
        analysis_sets = [
            ("Standard analysis", comparison_analysis_ids),
            ("Control analysis", analysis_ids),
        ]

    for analysis_type, analysis_id_set in analysis_sets:
        for analysis_id in analysis_id_set:
            try:
                data_path = _resolve_harmonized_dataset_path(
                    analysis_id,
                    test_biobank,
                    dataset,
                    fold=fold,
                )
                model_path = _resolve_trained_model_path(
                    analysis_id,
                    model_biobank,
                    model_dataset,
                    moe_model_name,
                    fold=fold,
                )
            except FileNotFoundError as e:
                print(f"> Skipping {analysis_id}: {e}", file=sys.stderr)
                continue

            prs_dataset = PRSDataset.from_pickle(data_path)
            moe_model = MoEPRS.from_saved_model(model_path)
            disease_prs_name = _get_disease_prs_name(analysis_id)
            expert_names = [
                MODEL_NAME_MAP.get(analysis_id, {}).get(prs_id, prs_id)
                for prs_id in moe_model.expert_cols
            ]
            if disease_prs_name not in expert_names:
                raise ValueError(
                    f"Could not find disease PRS '{disease_prs_name}' among experts "
                    f"for {analysis_id}. Available experts: {expert_names}"
                )

            disease_expert_idx = expert_names.index(disease_prs_name)
            mixing_weights = np.asarray(
                moe_model.predict_proba(prs_dataset), dtype=float
            )
            if mixing_weights.ndim != 2 or mixing_weights.shape != (
                prs_dataset.N,
                len(expert_names),
            ):
                raise ValueError(
                    f"Unexpected MoE probability matrix shape for {analysis_id}: "
                    f"{mixing_weights.shape}; expected "
                    f"({prs_dataset.N}, {len(expert_names)})."
                )

            disease_weights = mixing_weights[:, disease_expert_idx]
            finite = np.isfinite(disease_weights)
            if not finite.any():
                raise ValueError(
                    f"No finite disease-specific mixing weights found for {analysis_id}."
                )

            phenotype = _shorten_disease_label(
                ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            )
            analysis_to_phenotype[analysis_id] = phenotype
            phenotype_to_prs[phenotype] = disease_prs_name
            weight_df = pd.DataFrame(
                {
                    "AnalysisID": analysis_id,
                    "Phenotype": phenotype,
                    "Disease_PRS": disease_prs_name,
                    "Mixing_Weight": disease_weights[finite],
                }
            )
            if analysis_type is not None:
                weight_df["Analysis_Type"] = analysis_type
            weight_frames.append(weight_df)

    if len(weight_frames) == 0:
        raise ValueError("No fold-aware disease-specific mixing weights were found.")
    weights_df = pd.concat(weight_frames, ignore_index=True)

    if phenotype_order is None:
        plot_order = [
            analysis_to_phenotype[a]
            for a in analysis_ids
            if a in analysis_to_phenotype
        ]
    else:
        requested_order = [_shorten_disease_label(p) for p in phenotype_order]
        present = set(weights_df["Phenotype"])
        plot_order = [p for p in requested_order if p in present]
        plot_order.extend(
            p
            for p in [
                analysis_to_phenotype[a]
                for a in analysis_ids
                if a in analysis_to_phenotype
            ]
            if p not in plot_order
        )
    plot_order = list(dict.fromkeys(plot_order))
    if len(plot_order) == 0:
        plot_order = list(dict.fromkeys(weights_df["Phenotype"].dropna()))
    weights_df["Phenotype"] = pd.Categorical(
        weights_df["Phenotype"], categories=plot_order, ordered=True
    )

    rng = np.random.default_rng(random_state)
    strip_frames = []
    for phenotype in plot_order:
        phenotype_df = weights_df.loc[weights_df["Phenotype"] == phenotype]
        if max_strip_points is not None and len(phenotype_df) > max_strip_points:
            keep_idx = rng.choice(
                phenotype_df.index.to_numpy(), size=max_strip_points, replace=False
            )
            phenotype_df = phenotype_df.loc[keep_idx]
        strip_frames.append(phenotype_df)
    strip_df = pd.concat(strip_frames, ignore_index=True)

    if figsize is None:
        fig_width = max(9.0, 1.15 * len(plot_order))
        figsize = (fig_width, 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    if comparison_analysis_ids is None:
        prs_colors = assign_models_consistent_colors(list(phenotype_to_prs.values()))
        phenotype_palette = {
            phenotype: prs_colors[phenotype_to_prs[phenotype]]
            for phenotype in plot_order
        }

        sns.violinplot(
            data=weights_df,
            x="Phenotype",
            y="Mixing_Weight",
            hue="Phenotype",
            order=plot_order,
            hue_order=plot_order,
            palette=phenotype_palette,
            inner="quartile",
            cut=0,
            density_norm="width",
            bw_adjust=0.75,
            linewidth=1.0,
            saturation=0.85,
            legend=False,
            ax=ax,
        )
    else:
        analysis_type_order = ["Standard analysis", "Control analysis"]
        analysis_type_palette = {
            "Standard analysis": "#375E97",
            "Control analysis": "#FFBB00",
        }
        sns.violinplot(
            data=weights_df,
            x="Phenotype",
            y="Mixing_Weight",
            hue="Analysis_Type",
            order=plot_order,
            hue_order=analysis_type_order,
            palette=analysis_type_palette,
            inner="quartile",
            cut=0,
            density_norm="width",
            bw_adjust=0.75,
            linewidth=1.0,
            saturation=0.85,
            legend=True,
            ax=ax,
        )
    """
    sns.stripplot(
        data=strip_df,
        x="Phenotype",
        y="Mixing_Weight",
        hue="Phenotype",
        order=plot_order,
        hue_order=plot_order,
        palette=phenotype_palette,
        jitter=0.22,
        size=1.5,
        alpha=0.18,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    """

    if comparison_analysis_ids is None:
        medians = weights_df.groupby("Phenotype", observed=True)[
            "Mixing_Weight"
        ].median()
        median_values = [float(medians.loc[p]) for p in plot_order]
        ax.scatter(
            np.arange(len(plot_order)),
            median_values,
            s=30,
            facecolor="white",
            edgecolor="#222222",
            linewidth=1.0,
            zorder=5,
            label="Median",
        )
    else:
        medians = weights_df.groupby(
            ["Phenotype", "Analysis_Type"], observed=True
        )["Mixing_Weight"].median()
        offsets = {
            "Standard analysis": -0.2,
            "Control analysis": 0.2,
        }
        for analysis_type in analysis_type_order:
            xs = []
            ys = []
            for i, phenotype in enumerate(plot_order):
                key = (phenotype, analysis_type)
                if key not in medians.index:
                    continue
                xs.append(i + offsets[analysis_type])
                ys.append(float(medians.loc[key]))
            ax.scatter(
                xs,
                ys,
                s=26,
                facecolor="white",
                edgecolor="#222222",
                linewidth=1.0,
                zorder=5,
            )

    ax.set_xlabel("Phenotypes")
    ax.set_ylabel("Mixing weight for\nDisease-specific PRS")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    if title is None:
        title = (
            "Distribution of PRS mixing weights for disease-specific scores\n"
            f"{BIOBANK_NAME_MAP.get(test_biobank, test_biobank.upper())}"
        )
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(axis="x", labelrotation=25, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    legend_kwargs = {
        "loc": "center left",
        "bbox_to_anchor": (1.005, 0.5),
        "borderaxespad": 0.2,
        "frameon": False,
        "fontsize": 8,
        "handlelength": 1.1,
        "handletextpad": 0.4,
        "labelspacing": 0.35,
    }
    if comparison_analysis_ids is None:
        ax.legend(**legend_kwargs)
    else:
        ax.legend(title=None, **legend_kwargs)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.22, top=0.84)

    output_dir = osp.dirname(output_file)
    if output_dir:
        makedir(output_dir)
    fig.savefig(output_file, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return weights_df


def plot_prevalence_subsampled_mixing_accuracy_panels(
    moe_model_name,
    analysis_id,
    test_biobank="ukbb",
    dataset="test_data",
    incremental_metric="Nagelkerke_R2",
    disease_prs_name=None,
    output_file=None,
    partition_method="threshold",
    threshold=0.5,
    threshold_rules=None,
    n_quantiles=4,
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    random_state=42,
):
    """
    Plot 2x3 accuracy panels across three prevalence settings for mixing-weight groups.

    Columns:
    1) Original group prevalence
    2) Group prevalence matched to overall sample prevalence
    3) Group prevalence matched to 1:1 case-control

    Rows:
    - Top: ROC AUC
    - Bottom: Incremental R^2 (Liability R^2)
    """

    if output_file is None:
        output_file = (
            f"figures/section_4/accuracy_mixing_group_prevalence_subsampled_"
            f"{analysis_id}_{test_biobank}.png"
        )

    dat = PRSDataset.from_pickle(
        f"data/harmonized_data/{analysis_id}/{test_biobank}/{dataset}.pkl"
    )
    dat.filter_samples(dat.data["Ancestry"] == "EUR")
    if dat.phenotype_likelihood != "binomial":
        raise ValueError(
            "plot_prevalence_subsampled_mixing_accuracy_panels requires a binomial phenotype."
        )

    ukb_moe = MoEPRS.from_saved_model(
        f"data/trained_models/{analysis_id}/ukbb/train_data/{moe_model_name}.pkl"
    )

    grouping = _build_mixing_groups_from_moe(
        dat,
        ukb_moe,
        analysis_id=analysis_id,
        disease_prs_name=disease_prs_name,
        partition_method=partition_method,
        threshold=threshold,
        threshold_rules=threshold_rules,
        n_quantiles=n_quantiles,
    )
    disease_prs_name = grouping["disease_prs_name"]
    disease_expert_idx = grouping["disease_expert_idx"]
    group_assign_col = grouping["group_col"]
    group_levels = grouping["group_levels"]
    group_order = grouping["group_order"]
    group_masks = grouping["group_masks"]
    group_short_map = grouping["group_short_map"]
    short_order = grouping["short_order"]

    moe_label = f"{moe_model_name} (UKB)"
    weighted_label = "Weighted PRS"
    weighted_excl_label = "Weighted PRS (exc. disease)"
    keep_models = [moe_label, disease_prs_name, weighted_label, weighted_excl_label]

    trained_models = {
        moe_label: ukb_moe,
        weighted_label: GroupMeanWeightedPRS(ukb_moe, group_assign_col),
        weighted_excl_label: GroupMeanWeightedPRS(
            ukb_moe, group_assign_col, exclude_models=[disease_expert_idx]
        ),
    }
    preds = generate_predictions(dat, trained_models)
    metrics = ["AUROC", incremental_metric]

    phenotype_vals = np.asarray(dat.get_phenotype()).reshape(-1)
    overall_prevalence = float(np.nanmean(phenotype_vals))

    scenario_specs = [
        ("Original prevalence", None),
        ("Matched to overall prevalence", overall_prevalence),
        ("Matched to 1:1 prevalence", 0.5),
    ]

    def evaluate_scenario(scenario_label, target_prevalence, scenario_idx):
        eval_rows = []
        eval_groups = [("All", np.ones(dat.N, dtype=bool))] + [
            (g, np.asarray(group_masks[g], dtype=bool)) for g in group_levels
        ]

        for g_idx, (g_label, base_mask) in enumerate(eval_groups):
            eval_mask = base_mask
            if target_prevalence is not None:
                seed = (
                    None
                    if random_state is None
                    else int(random_state + 1000 * scenario_idx + g_idx)
                )
                try:
                    eval_mask = subsample_to_prevalence(
                        dat,
                        target_prevalence,
                        mask=eval_mask,
                        random_state=seed,
                    )
                except ValueError:
                    continue

            if eval_mask.sum() < min_group_size:
                continue

            try:
                gdf = evaluate_prs_models(
                    dat,
                    fitted_models=preds,
                    mask=eval_mask,
                    metrics=metrics,
                    evaluate_base_models=True,
                    min_group_size=min_group_size,
                )
            except Exception:
                continue

            if gdf is None or gdf.empty:
                continue

            gdf["EvalGroup"] = g_label
            gdf["Scenario"] = scenario_label
            gdf["N"] = int(eval_mask.sum())
            eval_rows.append(gdf)

        if len(eval_rows) == 0:
            return pd.DataFrame()

        sdf_long = pd.concat(eval_rows, ignore_index=True)
        sdf_long = sdf_long.loc[sdf_long["metric_kind"] == "base"].copy()

        val_wide = (
            sdf_long.pivot_table(
                index=["model_id", "model_name", "EvalGroup", "Scenario", "N"],
                columns="metric",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"model_name": "PGS"})
        )

        se_wide = sdf_long.pivot_table(
            index=["model_id", "EvalGroup", "Scenario"],
            columns="metric",
            values="se",
            aggfunc="first",
        ).reset_index()
        se_wide.columns = [
            c if c in ["model_id", "EvalGroup", "Scenario"] else f"{c}_err"
            for c in se_wide.columns
        ]

        sdf = val_wide.merge(
            se_wide, on=["model_id", "EvalGroup", "Scenario"], how="left"
        )
        sdf["PGS"] = sdf["PGS"].map(lambda x: MODEL_NAME_MAP[analysis_id].get(x, x))
        return sdf

    scenario_results = []
    for s_idx, (scenario_label, target_prev) in enumerate(scenario_specs):
        sdf = evaluate_scenario(scenario_label, target_prev, s_idx)
        if not sdf.empty:
            scenario_results.append(sdf)

    if len(scenario_results) == 0:
        raise ValueError(
            "No valid scenario/group evaluation results available to plot."
        )

    plot_df = pd.concat(scenario_results, ignore_index=True)
    plot_df = plot_df.loc[
        plot_df["PGS"].isin(keep_models) & plot_df["EvalGroup"].isin(group_order)
    ].copy()
    if plot_df.empty:
        raise ValueError("No target model results available after filtering.")

    plot_df["Model"] = plot_df["PGS"].replace({moe_label: "MoEPRS"})
    plot_df["GroupShort"] = plot_df["EvalGroup"].map(group_short_map)

    model_order = [
        m
        for m in ["MoEPRS", disease_prs_name, weighted_label, weighted_excl_label]
        if m in set(plot_df["Model"])
    ]
    disease_color = assign_models_consistent_colors([disease_prs_name]).get(
        disease_prs_name, "#4C78A8"
    )
    model_palette = {
        "MoEPRS": "#375E97",
        disease_prs_name: disease_color,
        weighted_label: "#111111",
        weighted_excl_label: "#6F6F6F",
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 6.8),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )

    def draw_accuracy_panel(
        ax, metric_name, scenario_label, show_legend=False, show_y=True
    ):
        metric_err_col = f"{metric_name}_err"
        cols = ["Scenario", "EvalGroup", "GroupShort", "Model", metric_name]
        if metric_err_col in plot_df.columns:
            cols.append(metric_err_col)

        mdf = plot_df.loc[
            (plot_df["Scenario"] == scenario_label)
            & plot_df["Model"].isin(model_order)
            & plot_df[metric_name].notna(),
            cols,
        ].drop_duplicates(subset=["Scenario", "EvalGroup", "Model"])

        if mdf.empty:
            ax.set_title(scenario_label)
            ax.set_axis_off()
            return

        sns.barplot(
            data=mdf,
            y="GroupShort",
            x=metric_name,
            hue="Model",
            order=short_order,
            hue_order=model_order,
            palette=model_palette,
            errorbar=None,
            ax=ax,
        )

        if metric_err_col in mdf.columns and mdf[metric_err_col].notna().any():
            num_hues = len(model_order)
            dodge_width = 0.8
            bar_height = dodge_width / max(num_hues, 1)
            dodge_offsets = np.linspace(
                -dodge_width / 2 + bar_height / 2,
                dodge_width / 2 - bar_height / 2,
                max(num_hues, 1),
            )
            y_base = {g: i for i, g in enumerate(short_order)}

            for i_h, hval in enumerate(model_order):
                for gval in short_order:
                    row = mdf.loc[(mdf["Model"] == hval) & (mdf["GroupShort"] == gval)]
                    if row.empty:
                        continue
                    x_err = row[metric_err_col].iloc[0]
                    if pd.isna(x_err):
                        continue
                    x_val = float(row[metric_name].iloc[0])
                    y_pos = y_base[gval] + dodge_offsets[i_h]
                    ax.errorbar(
                        x_val,
                        y_pos,
                        xerr=float(x_err),
                        fmt="none",
                        ecolor="black",
                        lw=1.0,
                        capsize=0,
                    )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(scenario_label)
        ax.set_axisbelow(True)
        ax.grid(True, axis="x", alpha=0.25)

        if metric_name == "AUROC":
            ax.set_xlim(0.5, 1.0)

        if show_legend:
            ax.legend(loc="best", frameon=False, fontsize=7, title=None)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        if not show_y:
            ax.set_yticklabels([])

    scenario_order = [s[0] for s in scenario_specs]
    for col_idx, scenario_label in enumerate(scenario_order):
        draw_accuracy_panel(
            axes[0, col_idx],
            "AUROC",
            scenario_label,
            show_legend=(col_idx == 0),
            show_y=(col_idx == 0),
        )
        draw_accuracy_panel(
            axes[1, col_idx],
            incremental_metric,
            scenario_label,
            show_legend=False,
            show_y=(col_idx == 0),
        )

    for col_idx in range(3):
        axes[0, col_idx].set_xlabel("ROC AUC")
        axes[1, col_idx].set_xlabel("Incremental $R^2$")

    bb_short = BIOBANK_NAME_MAP_SHORT.get(test_biobank.lower(), test_biobank.upper())
    fig.suptitle(
        f"{ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)} ({bb_short})"
    )
    if partition_method in {"quartile", "quantile"}:
        y_label = f"Quartiles of\nP({disease_prs_name}) mixing weight"
    else:
        y_label = f"Groups of\nP({disease_prs_name}) mixing weight"
    fig.supylabel(y_label)

    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_disease_prs_age_sex_accuracy(
    analysis_id,
    test_biobank="ukbb",
    dataset="test_data",
    metric="Nagelkerke_R2",
    keep_ancestry=("EUR",),
    moe_model_name="MoE-GS-prs-gating",
    model_biobank=None,
    model_dataset="train_data",
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_ci=DEFAULT_BOOTSTRAP_CI,
    bootstrap_seed=42,
    output_file=None,
):
    """
    Plot disease PRS+covariates accuracy across sex and recruitment-age strata.

    The figure shows the requested R²-like metric together with AUROC in separate
    panels because the two metrics have different natural scales. When the test
    and model biobanks match, each fold's held-out test data and matching model
    are used, and values are reported as fold means with fold SE. Evaluation on
    a different external biobank remains supported through fold-averaged
    predictions and participant bootstrapping.
    """

    if model_biobank is None:
        model_biobank = test_biobank

    if isinstance(analysis_id, str):
        analysis_ids = [analysis_id]
    else:
        analysis_ids = list(analysis_id)

    if output_file is None:
        if len(analysis_ids) == 1:
            output_file = (
                "figures/section_4/"
                f"accuracy_disease_prs_age_sex_{analysis_ids[0]}_{test_biobank}.png"
            )
        else:
            output_file = (
                "figures/section_4/"
                f"accuracy_disease_prs_age_sex_selected_phenotypes_{test_biobank}.pdf"
            )

    if isinstance(keep_ancestry, str):
        keep_ancestry = [keep_ancestry]
    elif keep_ancestry is not None:
        keep_ancestry = list(keep_ancestry)

    age_order = ["All", "Age<50", "Age 50–60", "Age>60"]
    display_label_map = {
        "All": "All",
        "Age<50": "<50",
        "Age 50–60": "50-60",
        "Age>60": ">60",
    }
    metrics_to_plot = list(dict.fromkeys([metric, "AUROC"]))
    external_evaluation = test_biobank.lower() != model_biobank.lower()
    uncertainty_label = "bootstrap SE" if external_evaluation else "fold SE"
    metric_panel_labels = {
        "Liability_R2": rf"Liability $R^2$ ($\pm 1$ {uncertainty_label})",
        "Liability_Probit_R2": rf"Liability $R^2$ ($\pm 1$ {uncertainty_label})",
        "Liability_Logit_R2": rf"Liability $R^2$ ($\pm 1$ {uncertainty_label})",
        "Nagelkerke_R2": rf"Nagelkerke $R^2$ ($\pm 1$ {uncertainty_label})",
        "AUROC": rf"AUC-ROC ($\pm 1$ {uncertainty_label})",
    }
    rows = []

    def fold_sort_key(path):
        fold = next(
            (
                part
                for part in osp.normpath(path).split(osp.sep)
                if part.startswith("fold_")
            ),
            "",
        )
        try:
            return (0, int(fold.removeprefix("fold_")))
        except ValueError:
            return (1, fold)

    def prepare_dataset(data_path):
        dat = PRSDataset.from_pickle(data_path)
        if keep_ancestry is not None:
            dat.filter_samples(dat.data["Ancestry"].isin(keep_ancestry))

        dat.data["SexG"] = (
            dat.data["Sex"].astype(int).astype(str).map(SEX_LABEL_MAP)
        )
        dat.data["AgeGroup3"] = pd.cut(
            dat.data["Age"],
            bins=[0, 50, 60, float("inf")],
            labels=["Age<50", "Age 50–60", "Age>60"],
            right=False,
        ).astype(str)
        dat.data["SexAgeGroup"] = dat.data["SexG"] + ": " + dat.data["AgeGroup3"]
        return dat

    def evaluate_fold_models(dat, fold_model_paths, aid, evaluation_scope):
        trained_models = {}
        model_catalog = []
        canonical_model_id = f"{model_biobank}/{model_dataset}:{disease_model_name}"

        for disease_path in fold_model_paths:
            fold = osp.basename(osp.dirname(osp.dirname(disease_path)))
            model_id = (
                f"{model_biobank}/{fold}/{model_dataset}:{disease_model_name}"
            )
            trained_models[model_id] = MultiPRS.from_saved_model(disease_path)
            model_catalog.append(_parse_model_id(model_id))

        if evaluation_scope == "external_ensemble_bootstrap":
            ensemble_predictions, ensemble_catalog = average_fold_predictions(
                dat, trained_models
            )
            eval_df = stratified_evaluation(
                dat,
                trained_predictions=ensemble_predictions,
                model_catalog=ensemble_catalog,
                test_biobank=test_biobank,
                metrics=metrics_to_plot,
                cat_group_cols=["SexG", "SexAgeGroup"],
                evaluate_base_models=False,
                min_group_size=min_group_size,
                bootstrap=True,
                n_bootstrap=bootstrap_resamples,
                bootstrap_ci=bootstrap_ci,
                random_state=bootstrap_seed,
            )
        else:
            eval_df = stratified_evaluation(
                dat,
                trained_models=trained_models,
                model_catalog=pd.DataFrame(model_catalog),
                test_biobank=test_biobank,
                metrics=metrics_to_plot,
                cat_group_cols=["SexG", "SexAgeGroup"],
                evaluate_base_models=False,
                min_group_size=min_group_size,
            )
        eval_df = eval_df.loc[
            (eval_df["metric"].isin(metrics_to_plot))
            & (eval_df["metric_kind"] == "base")
            & (eval_df["model_name"] == disease_model_name)
            & (eval_df["prediction_type"] == "full")
            & (eval_df["value"].notna())
        ].copy()
        eval_df["model_id"] = canonical_model_id
        eval_df["analysis_id"] = aid
        eval_df["test_biobank"] = test_biobank
        eval_df["test_dataset"] = (
            "full_data"
            if evaluation_scope == "external_ensemble_bootstrap"
            else "test_data"
        )
        eval_df["evaluation_scope"] = evaluation_scope
        eval_df["test_fold"] = (
            "ensemble"
            if evaluation_scope == "external_ensemble_bootstrap"
            else eval_df["train_fold"]
        )
        return eval_df

    for aid in analysis_ids:
        disease_prs_id = _get_disease_prs_id(aid)
        disease_model_name = f"{disease_prs_id}-covariates"
        model_pattern = (
            f"data/trained_models/{aid}/{model_biobank}/fold_*/"
            f"{model_dataset}/{disease_model_name}.pkl"
        )
        fold_model_paths = sorted(glob.glob(model_pattern), key=fold_sort_key)
        if not fold_model_paths:
            raise FileNotFoundError(
                "Could not find any fold-specific disease PRS+covariates models: "
                f"{model_pattern}"
            )

        if external_evaluation:
            # For an external cohort, use the mean fold-model prediction and
            # bootstrap participants to estimate test-cohort uncertainty.
            data_path = f"data/harmonized_data/{aid}/{test_biobank}/full_data.pkl"
            if not osp.exists(data_path):
                raise FileNotFoundError(f"Could not find dataset: {data_path}")
            fold_eval_df = evaluate_fold_models(
                prepare_dataset(data_path),
                fold_model_paths,
                aid,
                evaluation_scope="external_ensemble_bootstrap",
            )
        else:
            fold_eval_dfs = []
            model_paths_by_fold = {
                osp.basename(osp.dirname(osp.dirname(path))): path
                for path in fold_model_paths
            }
            data_pattern = (
                f"data/harmonized_data/{aid}/{test_biobank}/fold_*/{dataset}.pkl"
            )
            fold_data_paths = sorted(glob.glob(data_pattern), key=fold_sort_key)
            if not fold_data_paths:
                raise FileNotFoundError(
                    f"Could not find any fold-specific datasets: {data_pattern}"
                )

            for data_path in fold_data_paths:
                fold = osp.basename(osp.dirname(data_path))
                disease_path = model_paths_by_fold.get(fold)
                if disease_path is None:
                    raise FileNotFoundError(
                        "Could not find the matching disease PRS+covariates model "
                        f"for {aid}/{model_biobank}/{fold}."
                    )
                fold_eval_dfs.append(
                    evaluate_fold_models(
                        prepare_dataset(data_path),
                        [disease_path],
                        aid,
                        evaluation_scope="held_out_fold",
                    )
                )
            fold_eval_df = pd.concat(fold_eval_dfs, ignore_index=True)

        if external_evaluation:
            edf = fold_eval_df
        else:
            edf = aggregate_cross_validation_metrics(fold_eval_df)
        for _, row in edf.iterrows():
            sex = None
            age_group = None
            if row["eval_category"] == "SexG" and row["eval_group"] in {
                "Female",
                "Male",
            }:
                sex = row["eval_group"]
                age_group = "All"
            elif row["eval_category"] == "SexAgeGroup":
                parts = str(row["eval_group"]).split(": ", 1)
                if len(parts) == 2 and parts[0] in {"Female", "Male"}:
                    sex, age_group = parts
            if sex is None or age_group not in age_order:
                continue

            rows.append(
                {
                    "AnalysisID": aid,
                    "Phenotype": _shorten_disease_label(
                        ANALYSIS_TO_PHENOTYPE_MAP.get(aid, aid)
                    ),
                    "AgeGroup": age_group,
                    "Sex": sex,
                    "Metric": row["metric"],
                    "Value": float(row["value"]),
                    "SE": float(row["se"]) if pd.notna(row["se"]) else np.nan,
                }
            )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise ValueError("No age/sex accuracy results were available to plot.")

    plot_key = ["AnalysisID", "Metric", "AgeGroup", "Sex"]
    duplicated = plot_df.duplicated(plot_key, keep=False)
    if duplicated.any():
        duplicate_keys = (
            plot_df.loc[duplicated, plot_key]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "Multiple age/sex accuracy values were generated for the same "
            f"plotting group: {duplicate_keys}"
        )

    plot_df["AgeGroup"] = pd.Categorical(
        plot_df["AgeGroup"], categories=age_order, ordered=True
    )
    plot_df["Metric"] = pd.Categorical(
        plot_df["Metric"], categories=metrics_to_plot, ordered=True
    )
    plot_df = plot_df.sort_values(["AnalysisID", "Metric", "AgeGroup", "Sex"])

    def nice_limits(lower, upper, hard_lower=None, hard_upper=None):
        """Round padded limits to readable values with roughly five intervals."""
        if not np.isfinite(lower) or not np.isfinite(upper):
            return lower, upper
        if upper <= lower:
            upper = lower + max(0.01, abs(lower) * 0.1)

        raw_step = (upper - lower) / 5.0
        magnitude = 10.0 ** np.floor(np.log10(raw_step))
        normalized_step = raw_step / magnitude
        for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
            if normalized_step <= candidate:
                step = candidate * magnitude
                break

        lower = np.floor(lower / step) * step
        upper = np.ceil(upper / step) * step
        if hard_lower is not None:
            lower = max(hard_lower, lower)
        if hard_upper is not None:
            upper = min(hard_upper, upper)
        return float(lower), float(upper)

    metric_y_limits = {}
    for metric_name in metrics_to_plot:
        metric_df = plot_df.loc[plot_df["Metric"] == metric_name]
        vals = metric_df["Value"].dropna()
        if vals.empty:
            continue

        lower_extent = float(vals.min())
        upper_extent = float(vals.max())
        valid_uncertainty = metric_df[["Value", "SE"]].dropna()
        if not valid_uncertainty.empty:
            lower_extent = min(
                lower_extent,
                float((valid_uncertainty["Value"] - valid_uncertainty["SE"]).min()),
            )
            upper_extent = max(
                upper_extent,
                float((valid_uncertainty["Value"] + valid_uncertainty["SE"]).max()),
            )

        if metric_name == "AUROC":
            val_range = upper_extent - lower_extent
            pad = max(0.015, 0.12 * val_range)
            lower = lower_extent - pad
            upper = upper_extent + pad
            if upper - lower < 0.1:
                midpoint = (lower + upper) / 2.0
                lower, upper = midpoint - 0.05, midpoint + 0.05
            auroc_floor = 0.5 if lower_extent >= 0.5 else 0.0
            lower, upper = nice_limits(
                lower,
                upper,
                hard_lower=auroc_floor,
                hard_upper=1.0,
            )
        else:
            val_range = upper_extent - lower_extent
            pad = max(0.01, 0.1 * val_range)
            lower = lower_extent - pad
            upper = upper_extent + pad
            if upper - lower < 0.1:
                midpoint = (lower + upper) / 2.0
                lower, upper = midpoint - 0.05, midpoint + 0.05
            lower, upper = nice_limits(
                lower,
                upper,
                hard_lower=0.0,
                hard_upper=1.0,
            )
        metric_y_limits[metric_name] = (lower, upper)

    n_phenotypes = len(analysis_ids)
    single_phenotype = n_phenotypes == 1
    fig_width = 8.2
    fig_height = 3.0 if single_phenotype else 1.55 * n_phenotypes + 1.35
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Keep row labels in their own column so long phenotype names cannot push
    # either metric panel out of alignment.  For a single phenotype the name is
    # moved into the figure title and both panels receive equal width.
    if single_phenotype:
        grid = fig.add_gridspec(
            n_phenotypes,
            len(metrics_to_plot),
            left=0.09,
            right=0.98,
            bottom=0.21,
            top=0.68,
            wspace=0.28,
        )
        label_axes = []
    else:
        grid = fig.add_gridspec(
            n_phenotypes,
            len(metrics_to_plot) + 1,
            width_ratios=[1.45] + [4.0] * len(metrics_to_plot),
            left=0.025,
            right=0.98,
            bottom=max(0.06, 0.58 / fig_height),
            top=1.0 - 1.02 / fig_height,
            wspace=0.28,
            hspace=0.24,
        )
        label_axes = [
            fig.add_subplot(grid[row_idx, 0]) for row_idx in range(n_phenotypes)
        ]

    axes = np.empty((n_phenotypes, len(metrics_to_plot)), dtype=object)
    for row_idx in range(n_phenotypes):
        for col_idx in range(len(metrics_to_plot)):
            sharex_ax = axes[0, 0] if row_idx + col_idx > 0 else None
            sharey_ax = axes[0, col_idx] if row_idx > 0 else None
            grid_col = col_idx if single_phenotype else col_idx + 1
            axes[row_idx, col_idx] = fig.add_subplot(
                grid[row_idx, grid_col],
                sharex=sharex_ax,
                sharey=sharey_ax,
            )

    sex_palette = {"Female": "#B05A9D", "Male": "#375E97"}
    sex_offsets = {"Female": -0.08, "Male": 0.08}
    x_pos = {g: i for i, g in enumerate(age_order)}
    bb_short = BIOBANK_NAME_MAP_SHORT.get(
        test_biobank.lower(), test_biobank.upper()
    )

    for row_idx, aid in enumerate(analysis_ids):
        phenotype = _shorten_disease_label(ANALYSIS_TO_PHENOTYPE_MAP.get(aid, aid))

        for col_idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[row_idx, col_idx]
            sdf = plot_df.loc[
                (plot_df["AnalysisID"] == aid)
                & (plot_df["Metric"].astype(str) == metric_name)
            ].copy()
            if sdf.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
            else:
                for sex in ["Female", "Male"]:
                    sex_df = sdf.loc[sdf["Sex"] == sex].copy()
                    if sex_df.empty:
                        continue
                    all_df = sex_df.loc[sex_df["AgeGroup"].astype(str) == "All"]
                    if not all_df.empty:
                        ax.scatter(
                            [x_pos["All"] + sex_offsets[sex]],
                            [all_df["Value"].iloc[0]],
                            s=28,
                            color=sex_palette[sex],
                            label=sex,
                            zorder=4,
                        )
                        if pd.notna(all_df["SE"].iloc[0]):
                            ax.errorbar(
                                [x_pos["All"] + sex_offsets[sex]],
                                [all_df["Value"].iloc[0]],
                                yerr=[all_df["SE"].iloc[0]],
                                fmt="none",
                                ecolor=sex_palette[sex],
                                elinewidth=0.9,
                                capsize=2.5,
                                capthick=0.9,
                                zorder=2,
                            )

                    age_df = sex_df.loc[
                        sex_df["AgeGroup"].astype(str) != "All"
                    ].copy()
                    xs = [
                        x_pos[g] + sex_offsets[sex]
                        for g in age_df["AgeGroup"].astype(str).values
                    ]
                    ax.plot(
                        xs,
                        age_df["Value"].values,
                        marker="o",
                        lw=1.2,
                        ms=5,
                        color=sex_palette[sex],
                        zorder=3,
                    )
                    has_se = age_df["SE"].notna().to_numpy()
                    if has_se.any():
                        ax.errorbar(
                            np.asarray(xs)[has_se],
                            age_df.loc[has_se, "Value"],
                            yerr=age_df.loc[has_se, "SE"],
                            fmt="none",
                            ecolor=sex_palette[sex],
                            elinewidth=0.9,
                            capsize=2.5,
                            capthick=0.9,
                            zorder=2,
                        )

            if row_idx == 0:
                ax.set_title(
                    metric_panel_labels.get(
                        metric_name, METRIC_NAME_MAP.get(metric_name, metric_name)
                    ),
                    fontsize=10,
                    pad=2,
                )
            ax.set_ylabel("")
            if metric_name in metric_y_limits:
                ax.set_ylim(*metric_y_limits[metric_name])
            ax.set_xlim(-0.45, len(age_order) - 0.55)
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            ax.axvline(
                0.5,
                color="#C7C7C7",
                linewidth=0.7,
                linestyle=":",
                zorder=0,
            )
            ax.tick_params(axis="both", labelsize=8.5)
            sns.despine(ax=ax)
            if row_idx != len(analysis_ids) - 1:
                ax.set_xlabel("")

        if not single_phenotype:
            label_ax = label_axes[row_idx]
            label_ax.text(
                0.9,
                0.5,
                phenotype,
                ha="right",
                va="center",
                fontsize=9.5,
                linespacing=1.05,
            )
            label_ax.set_axis_off()

    for ax in axes[-1, :]:
        ax.set_xticks(range(len(age_order)))
        ax.set_xticklabels([display_label_map[g] for g in age_order])
        ax.set_xlabel("Recruitment age group", fontsize=9.5, labelpad=4)

    legend_entries = {}
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_entries.setdefault(label, handle)
    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries),
            loc="upper center",
            ncol=2,
            frameon=False,
            title="Sex",
            bbox_to_anchor=(0.5, 0.88 if single_phenotype else 0.94),
            fontsize=9,
            title_fontsize=9,
            handletextpad=0.5,
            columnspacing=1.4,
        )
    if single_phenotype:
        title_phenotype = ANALYSIS_TO_PHENOTYPE_MAP.get(
            analysis_ids[0], analysis_ids[0]
        )
        figure_title = (
            f"{title_phenotype}: "
            f"disease PRS + covariates ({bb_short})"
        )
    else:
        figure_title = (
            f"Disease PRS + covariates accuracy by age and sex ({bb_short})"
        )
    fig.suptitle(figure_title, fontsize=11.5, y=0.98)

    output_dir = osp.dirname(output_file)
    if output_dir:
        makedir(output_dir)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def extract_minority_ancestry_accuracy_panels(
    moe_model_name,
    analysis_ids=None,
    biobanks=("ukbb", "cartagene"),
    dataset_by_biobank=None,
    disease_prs_name_map=None,
    metric="Nagelkerke_R2",
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
):
    """
    Plot pooled non-EUR (Ancestry != EUR) performance across phenotypes and biobanks.
    Models shown: disease-specific PRS, MoEPRS, Weighted PRS, Weighted PRS (exc. disease).
    """

    if analysis_ids is None:
        analysis_ids = sorted(
            [
                a
                for a in ANALYSIS_TO_PHENOTYPE_MAP.keys()
                if ANALYSIS_TO_TABLE_MAP.get(a) == "multitrait_prs_table"
            ]
        )

    if dataset_by_biobank is None:
        dataset_by_biobank = {"ukbb": "test_data", "cartagene": "full_data"}

    if disease_prs_name_map is None:
        disease_prs_name_map = {}

    plot_rows = []
    metric_err_col = f"{metric}_err"

    for biobank in biobanks:
        dataset = dataset_by_biobank.get(biobank, "test_data")

        for analysis_id in analysis_ids:
            try:
                dat = PRSDataset.from_pickle(
                    f"data/harmonized_data/{analysis_id}/{biobank}/{dataset}.pkl"
                )
            except Exception as e:
                print(e)
                continue

            dat.filter_samples(dat.data["Ancestry"] != "EUR")
            if dat.N < min_group_size:
                continue

            try:
                ukb_moe = MoEPRS.from_saved_model(
                    f"data/trained_models/{analysis_id}/ukbb/train_data/{moe_model_name}.pkl"
                )
            except Exception as e:
                print(e)
                continue
            try:
                ukb_multiprs = MultiPRS.from_saved_model(
                    f"data/trained_models/{analysis_id}/ukbb/train_data/MultiPRS.pkl"
                )
            except Exception as e:
                print(e)
                continue

            dat.data["MinorityGroup"] = "non-EUR"

            disease_prs_name = disease_prs_name_map.get(
                analysis_id, _get_disease_prs_name(analysis_id)
            )
            mapped_expert_names = [
                MODEL_NAME_MAP[analysis_id].get(prs_id, prs_id)
                for prs_id in ukb_moe.expert_cols
            ]
            if disease_prs_name not in mapped_expert_names:
                print(disease_prs_name)
                print(mapped_expert_names)
                continue

            disease_expert_idx = mapped_expert_names.index(disease_prs_name)

            moe_label = f"{moe_model_name} (UKB)"
            multiprs_label = "MultiPRS (UKB)"
            weighted_label = "Weighted PRS"
            weighted_excl_label = "Weighted PRS (exc. disease)"

            trained_models = {
                moe_label: ukb_moe,
                multiprs_label: ukb_multiprs,
                weighted_label: GroupMeanWeightedPRS(ukb_moe, "MinorityGroup"),
                weighted_excl_label: GroupMeanWeightedPRS(
                    ukb_moe, "MinorityGroup", exclude_models=[disease_expert_idx]
                ),
            }

            try:
                edf = stratified_evaluation(
                    dat,
                    trained_models=trained_models,
                    cat_group_cols=None,
                    metrics=[metric],
                    evaluate_base_models=True,
                    min_group_size=min_group_size,
                )
            except Exception as e:
                print(e)
                continue

            edf = edf.loc[
                (edf["metric"] == metric)
                & (edf["metric_kind"] == "base")
                & (
                    (edf["prediction_type"] == "prs_only")
                    | (edf["model_category"] == "SinglePRS")
                )
                & (edf["eval_group"] == "All")
            ].copy()
            if edf.empty:
                continue

            edf["PGS"] = edf["model_name"].map(
                lambda x: MODEL_NAME_MAP[analysis_id].get(x, x)
            )
            edf.loc[edf["model_id"] == moe_label, "PGS"] = moe_label
            edf.loc[edf["model_id"] == multiprs_label, "PGS"] = multiprs_label
            edf.loc[edf["model_id"] == weighted_label, "PGS"] = weighted_label
            edf.loc[edf["model_id"] == weighted_excl_label, "PGS"] = weighted_excl_label

            keep_models = {
                moe_label,
                multiprs_label,
                disease_prs_name,
                weighted_label,
                weighted_excl_label,
            }
            edf = edf.loc[edf["PGS"].isin(keep_models)].copy()
            if edf.empty:
                continue

            edf["Model Name"] = edf["PGS"].replace(
                {
                    moe_label: "MoEPRS (UKB)",
                    multiprs_label: "MultiPRS (UKB)",
                    disease_prs_name: "Disease-specific PRS",
                    weighted_label: "Weighted PRS",
                    weighted_excl_label: "Weighted PRS (exc. disease)",
                }
            )
            edf[metric] = edf["value"]
            edf[metric_err_col] = edf["se"]
            edf["Phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            edf["Biobank"] = BIOBANK_NAME_MAP.get(biobank, biobank.upper())

            cols = ["Model Name", "Phenotype", "Biobank", metric]
            if metric_err_col in edf.columns:
                cols.append(metric_err_col)
            plot_rows.append(edf)  # [cols])

    if len(plot_rows) == 0:
        raise ValueError("No minority-ancestry evaluation results available to plot.")

    plot_df = pd.concat(plot_rows, ignore_index=True)
    return plot_df


def plot_minority_ancestry_accuracy_panels(
    moe_model_name,
    analysis_ids=None,
    biobanks=("ukbb", "cartagene"),
    dataset_by_biobank=None,
    disease_prs_name_map=None,
    metric="Nagelkerke_R2",
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    phenotype_order=None,
    output_file=None,
):
    """
    Plot pooled non-EUR (Ancestry != EUR) performance across phenotypes and biobanks.
    Models shown: disease-specific PRS, MoEPRS, Weighted PRS, Weighted PRS (exc. disease).
    """

    if output_file is None:
        output_file = "figures/section_4/accuracy_minority_ancestry_metrics_all_mixed.pdf"

    if analysis_ids is None:
        analysis_ids = sorted(
            [
                a
                for a in ANALYSIS_TO_PHENOTYPE_MAP.keys()
                if ANALYSIS_TO_TABLE_MAP.get(a) == "multitrait_prs_table"
            ]
        )

    if dataset_by_biobank is None:
        dataset_by_biobank = {"ukbb": "test_data", "cartagene": "full_data"}

    if disease_prs_name_map is None:
        disease_prs_name_map = {}

    hue_order = [
        "MoEPRS (UKB)",
        "MultiPRS (UKB)",
        "Disease-specific PRS",
        "Weighted PRS",
        "Weighted PRS (exc. disease)",
    ]
    palette = {
        "MoEPRS (UKB)": "#375E97",
        "MultiPRS (UKB)": "#FFBB00",
        "Disease-specific PRS": "#BC80BD",
        "Weighted PRS": "#111111",
        "Weighted PRS (exc. disease)": "#6F6F6F",
    }

    plot_rows = []
    metric_err_col = f"{metric}_err"

    for biobank in biobanks:
        dataset = dataset_by_biobank.get(biobank, "test_data")

        for analysis_id in analysis_ids:
            try:
                dat = PRSDataset.from_pickle(
                    f"data/harmonized_data/{analysis_id}/{biobank}/{dataset}.pkl"
                )
            except Exception as e:
                print(e)
                continue

            dat.filter_samples(dat.data["Ancestry"] != "EUR")
            if dat.N < min_group_size:
                continue

            try:
                ukb_moe = MoEPRS.from_saved_model(
                    f"data/trained_models/{analysis_id}/ukbb/train_data/{moe_model_name}.pkl"
                )
            except Exception as e:
                print(e)
                continue
            try:
                ukb_multiprs = MultiPRS.from_saved_model(
                    f"data/trained_models/{analysis_id}/ukbb/train_data/MultiPRS.pkl"
                )
            except Exception as e:
                print(e)
                continue

            dat.data["MinorityGroup"] = "non-EUR"

            disease_prs_name = disease_prs_name_map.get(
                analysis_id, _get_disease_prs_name(analysis_id)
            )
            mapped_expert_names = [
                MODEL_NAME_MAP[analysis_id].get(prs_id, prs_id)
                for prs_id in ukb_moe.expert_cols
            ]
            if disease_prs_name not in mapped_expert_names:
                print(disease_prs_name)
                print(mapped_expert_names)
                continue

            disease_expert_idx = mapped_expert_names.index(disease_prs_name)

            moe_label = f"{moe_model_name} (UKB)"
            multiprs_label = "MultiPRS (UKB)"
            weighted_label = "Weighted PRS"
            weighted_excl_label = "Weighted PRS (exc. disease)"

            trained_models = {
                moe_label: ukb_moe,
                multiprs_label: ukb_multiprs,
                weighted_label: GroupMeanWeightedPRS(ukb_moe, "MinorityGroup"),
                weighted_excl_label: GroupMeanWeightedPRS(
                    ukb_moe, "MinorityGroup", exclude_models=[disease_expert_idx]
                ),
            }

            try:
                edf = stratified_evaluation(
                    dat,
                    trained_models=trained_models,
                    cat_group_cols=None,
                    metrics=[metric],
                    evaluate_base_models=True,
                    min_group_size=min_group_size,
                )
            except Exception as e:
                print(e)
                continue

            edf = edf.loc[
                (edf["metric"] == metric)
                & (edf["metric_kind"] == "base")
                & (
                    (edf["prediction_type"] == "prs_only")
                    | (edf["model_category"] == "SinglePRS")
                )
                & (edf["eval_group"] == "All")
            ].copy()
            if edf.empty:
                continue

            edf["PGS"] = edf["model_name"].map(
                lambda x: MODEL_NAME_MAP[analysis_id].get(x, x)
            )
            edf.loc[edf["model_id"] == moe_label, "PGS"] = moe_label
            edf.loc[edf["model_id"] == multiprs_label, "PGS"] = multiprs_label
            edf.loc[edf["model_id"] == weighted_label, "PGS"] = weighted_label
            edf.loc[edf["model_id"] == weighted_excl_label, "PGS"] = weighted_excl_label

            keep_models = {
                moe_label,
                multiprs_label,
                disease_prs_name,
                weighted_label,
                weighted_excl_label,
            }
            edf = edf.loc[edf["PGS"].isin(keep_models)].copy()
            if edf.empty:
                continue

            edf["Model Name"] = edf["PGS"].replace(
                {
                    moe_label: "MoEPRS (UKB)",
                    multiprs_label: "MultiPRS (UKB)",
                    disease_prs_name: "Disease-specific PRS",
                    weighted_label: "Weighted PRS",
                    weighted_excl_label: "Weighted PRS (exc. disease)",
                }
            )
            edf[metric] = edf["value"]
            edf[metric_err_col] = edf["se"]
            edf["Phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            edf["Biobank"] = BIOBANK_NAME_MAP.get(biobank, biobank.upper())

            cols = ["Model Name", "Phenotype", "Biobank", metric]
            if metric_err_col in edf.columns:
                cols.append(metric_err_col)
            plot_rows.append(edf[cols])

    if len(plot_rows) == 0:
        raise ValueError("No minority-ancestry evaluation results available to plot.")

    plot_df = pd.concat(plot_rows, ignore_index=True)
    plot_df = _drop_cartagene_sparse_trait_rows(plot_df)

    if phenotype_order is None:
        phenotype_order = sorted(plot_df["Phenotype"].dropna().unique().tolist())

    plot_combined_accuracy_metrics(
        plot_df,
        output_f=output_file,
        x="Phenotype",
        metric=metric,
        palette=palette,
        order=phenotype_order,
        hue_order=hue_order,
        column=None,
        row="Biobank",
        sharey=False,
        height=3,
        aspect=4,
        x_tick_rotation=90,
    )


def plot_minority_ancestry_accuracy_from_eval_metrics(
    moe_model_name,
    analysis_ids=None,
    biobanks=("ukbb",),
    dataset_by_biobank=None,
    train_biobank="test_biobank",
    metric="Nagelkerke_R2",
    metric_kind="incremental_vs_ref",
    ref_model_biobank="train_biobank",
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    phenotype_order=None,
    output_file=None,
):
    """
    Plot non-EUR performance directly from the evaluation CSVs.

    This mirrors the standard section 4 accuracy definition by using
    incremental_vs_ref metrics relative to the covariates model from the
    model-training biobank. ``train_biobank='test_biobank'`` selects each
    evaluation cohort's own cross-validated models.
    """

    if output_file is None:
        output_file = (
            "figures/section_4/"
            "accuracy_minority_ancestry_eval_metrics_all_mixed.pdf"
        )

    if analysis_ids is None:
        analysis_ids = sorted(
            [
                a
                for a in ANALYSIS_TO_PHENOTYPE_MAP.keys()
                if ANALYSIS_TO_TABLE_MAP.get(a) == "multitrait_prs_table"
            ]
        )

    if dataset_by_biobank is None:
        dataset_by_biobank = {biobank: "test_data" for biobank in biobanks}

    if train_biobank == "test_biobank":
        plot_train_biobanks = list(dict.fromkeys(biobanks))
    else:
        plot_train_biobanks = [train_biobank]

    hue_order = []
    for model_name in ("MoEPRS", "MultiPRS"):
        hue_order.extend(
            f"{model_name} ({BIOBANK_NAME_MAP_SHORT.get(b, b)})"
            for b in plot_train_biobanks
        )
    hue_order.append("Disease-specific PRS")
    palette = {
        "MoEPRS (UKB)": "#375E97",
        "MoEPRS (CaG)": "#8CA8D8",
        "MultiPRS (UKB)": "#FFBB00",
        "MultiPRS (CaG)": "#FFE066",
        "Disease-specific PRS": "#BC80BD",
    }

    plot_rows = []
    metric_err_col = f"{metric}_err"

    for biobank in biobanks:
        dataset = dataset_by_biobank.get(biobank, "test_data")
        current_train_biobank = (
            biobank if train_biobank == "test_biobank" else train_biobank
        )
        train_biobank_short = BIOBANK_NAME_MAP_SHORT.get(
            current_train_biobank, current_train_biobank
        )

        for analysis_id in analysis_ids:
            eval_f = f"data/evaluation/{analysis_id}/{biobank}/{dataset}.csv"
            try:
                df = read_transform_eval_metrics(
                    eval_f,
                    generate_missing_external=(
                        biobank == "cartagene" and dataset == "full_data"
                    ),
                    train_biobank=current_train_biobank,
                    external_metrics=metric,
                )
            except FileNotFoundError as e:
                print(e)
                continue
            disease_prs_name = _get_disease_prs_name(analysis_id)

            sub_df = df.loc[
                (df["metric"] == metric)
                & (df["metric_kind"] == metric_kind)
                & (df["eval_category"] == "Coarse Ancestry")
                & (df["eval_group"] == "non-EUR")
                & (df["prediction_type"] == "full")
                & (df["n"] >= min_group_size)
                & (~df["value"].isna())
            ].copy()

            if ref_model_biobank is not None and "ref_model_biobank" in sub_df.columns:
                if ref_model_biobank == "train_biobank":
                    sub_df = sub_df.loc[
                        sub_df["ref_model_biobank"] == sub_df["train_biobank"]
                    ]
                elif ref_model_biobank == "test_biobank":
                    sub_df = sub_df.loc[
                        sub_df["ref_model_biobank"] == sub_df["test_biobank"]
                    ]
                else:
                    sub_df = sub_df.loc[
                        sub_df["ref_model_biobank"] == ref_model_biobank
                    ]

            sub_df = sub_df.loc[
                (
                    sub_df["model_category"].isin(
                        ["MoE", "MultiPRS", "SinglePRS+Covariates"]
                    )
                )
                & (sub_df["train_biobank"] == current_train_biobank)
            ].copy()

            sub_df = sub_df.loc[
                (
                    (sub_df["model_category"] == "MoE")
                    & (sub_df["model_name"] == moe_model_name)
                )
                | (
                    (sub_df["model_category"] == "MultiPRS")
                    & (sub_df["model_name"] == "MultiPRS")
                )
                | (
                    (sub_df["model_category"] == "SinglePRS+Covariates")
                    & (sub_df["model_name"] == disease_prs_name)
                )
            ].copy()

            if sub_df.empty:
                continue

            sub_df["Model Name"] = sub_df["model_name"].replace(
                {
                    moe_model_name: f"MoEPRS ({train_biobank_short})",
                    "MultiPRS": f"MultiPRS ({train_biobank_short})",
                    disease_prs_name: "Disease-specific PRS",
                }
            )
            sub_df[metric] = sub_df["value"]
            sub_df[metric_err_col] = sub_df["se"]
            sub_df["Phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP.get(
                analysis_id, analysis_id
            )
            sub_df["Biobank"] = BIOBANK_NAME_MAP.get(biobank, biobank.upper())

            plot_rows.append(
                sub_df[["Model Name", "Phenotype", "Biobank", metric, metric_err_col]]
            )

    if len(plot_rows) == 0:
        raise ValueError("No minority-ancestry evaluation metrics available to plot.")

    plot_df = pd.concat(plot_rows, ignore_index=True)

    if phenotype_order is None:
        phenotype_order = sorted(plot_df["Phenotype"].dropna().unique().tolist())

    plot_df["Phenotype"] = plot_df["Phenotype"].map(_shorten_disease_label)
    plot_phenotype_order = [_shorten_disease_label(p) for p in phenotype_order]

    test_models = []
    significance_symbols = []
    for model_biobank in plot_train_biobanks:
        model_biobank_short = BIOBANK_NAME_MAP_SHORT.get(
            model_biobank, model_biobank
        )
        test_models.extend(
            [
                (
                    f"MoEPRS ({model_biobank_short})",
                    f"MultiPRS ({model_biobank_short})",
                ),
                (
                    f"MoEPRS ({model_biobank_short})",
                    "Disease-specific PRS",
                ),
            ]
        )
        significance_symbols.extend(["*", "+"])

    plot_combined_accuracy_metrics(
        plot_df,
        output_f=output_file,
        x="Phenotype",
        metric=metric,
        palette=palette,
        order=[p for p in plot_phenotype_order if p in set(plot_df["Phenotype"])],
        hue_order=hue_order,
        column=None,
        row="Biobank",
        sharey=False,
        height=3,
        aspect=4,
        x_tick_rotation=30,
        legend_title="Model Name (Training biobank)",
        test_models=test_models,
        significance_symbols=significance_symbols,
        ylim=(0.5, 1.0) if metric == "AUROC" else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot figures of section 4 of manuscript"
    )

    parser.add_argument(
        "--moe-model",
        dest="moe_model",
        type=str,
        default="MoE-GS-prs-gating",
        help="The name of the MoE model to plot as reference.",
    )

    parser.add_argument(
        "--binary-metric",
        dest="binary_metric",
        type=str,
        choices={
            "AUROC",
            "Liability_R2",
            "Nagelkerke_R2",
            "CoxSnell_R2",
            "McFadden_R2",
            "Liability_Probit_R2",
            "Liability_Logit_R2",
        },
        default="Nagelkerke_R2",
        help=(
            "The binary metric to plot. The default is incremental "
            "Nagelkerke R2 relative to the matching covariates-only model, "
            "reported as the mean and SE across held-out folds."
        ),
    )

    parser.add_argument(
        "--mixing-fold",
        dest="mixing_fold",
        type=str,
        default=DEFAULT_SECTION4_PLOTTING_FOLD,
        help=(
            "Reference model fold used for descriptive full-cohort mixing-weight "
            "figures; quartile accuracy figures use every available fold "
            f"(default: {DEFAULT_SECTION4_PLOTTING_FOLD})."
        ),
    )

    args = parser.parse_args()

    accuracy_metric_kind = (
        "base" if args.binary_metric == "AUROC" else "incremental_vs_ref"
    )
    accuracy_ref_model_biobank = (
        None if accuracy_metric_kind == "base" else "train_biobank"
    )

    sns.set_context("paper", font_scale=SECTION4_FONT_SCALE)
    makedir("figures/section_4/")

    palette = {
        "MoEPRS (UKB)": "#375E97",
        "MoEPRS (CaG)": "#8CA8D8",
        "MultiPRS (UKB)": "#FFBB00",
        "MultiPRS (CaG)": "#FFE066",
        "Disease-specific PRS": "#BC80BD",
        "Covariates only": "#888888",
    }

    hue_order = [
        "MoEPRS (UKB)",
        "MoEPRS (CaG)",
        "MultiPRS (UKB)",
        "MultiPRS (CaG)",
        "Disease-specific PRS",
        "Covariates only",
    ]
    control_palette = {
        "MoEPRS (irrelevant PRSs; UKB)": "#7FA6D6",
        "MoEPRS (irrelevant PRSs; CaG)": "#B4CAE5",
        "MoEPRS (relevant PRSs; UKB)": "#1F4E79",
        "MoEPRS (relevant PRSs; CaG)": "#4F769B",
        "MultiPRS (irrelevant PRSs; UKB)": "#F6D86B",
        "MultiPRS (irrelevant PRSs; CaG)": "#FAE9A7",
        "MultiPRS (relevant PRSs; UKB)": "#D89C00",
        "MultiPRS (relevant PRSs; CaG)": "#EDBE4C",
        "Disease-specific PRS": "#BC80BD",
        "Covariates only": "#888888",
    }
    control_hue_order = [
        "MoEPRS (irrelevant PRSs; UKB)",
        "MoEPRS (irrelevant PRSs; CaG)",
        "MoEPRS (relevant PRSs; UKB)",
        "MoEPRS (relevant PRSs; CaG)",
        "MultiPRS (irrelevant PRSs; UKB)",
        "MultiPRS (irrelevant PRSs; CaG)",
        "MultiPRS (relevant PRSs; UKB)",
        "MultiPRS (relevant PRSs; CaG)",
        "Disease-specific PRS",
        "Covariates only",
    ]

    phenotype_order = [
        "Type 2 Diabetes",
        "Type 1 Diabetes",
        "Asthma",
        "Gout",
        "Coronary Artery Disease",
        "Atrial Fibrillation",
        "Hypertension",
        "Stroke",
        "Heart Failure",
    ]

    analysis_tables = {
        "standard": "multitrait_prs_table",
        "control": "control_multitrait_prs_table",
    }

    print(">>> Section 4 Figures <<<")

    for tab_name, analysis_table in analysis_tables.items():
        paired_standard_analysis_ids = None
        if tab_name == "control":
            paired_standard_analysis_ids = [
                analysis_id[:-5]
                for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP
                if analysis_id.endswith("_CTRL")
                and ANALYSIS_TO_TABLE_MAP.get(analysis_id)
                == "control_multitrait_prs_table"
                and ANALYSIS_TO_TABLE_MAP.get(analysis_id[:-5])
                == "multitrait_prs_table"
            ]

        evaluation_biobanks = (
            ("ukbb",) if tab_name == "control" else ("ukbb", "cartagene")
        )
        metrics_dfs = []
        for biobank in evaluation_biobanks:
            bb_short = BIOBANK_NAME_MAP_SHORT[biobank]
            other_biobank = "cartagene" if biobank == "ukbb" else "ukbb"
            other_bb_short = BIOBANK_NAME_MAP_SHORT[other_biobank]
            metrics_df = extract_accuracy_data_all_phenotypes(
                args.moe_model,
                biobank,
                train_biobank=biobank,
                binary_metric=args.binary_metric,
                analysis_table_id=analysis_table,
                metric_kind=accuracy_metric_kind,
                ref_model_biobank=accuracy_ref_model_biobank,
                prediction_type="full",
                dataset="test_data",
                exclude_all_group=False,
                add_training_biobank_to_model_name=True,
                aggregate_single_prs=False,
            )

            metrics_df = _retain_disease_specific_prs(metrics_df)
            metrics_df = metrics_df.loc[metrics_df["Evaluation Group"] == "All"]
            metrics_df["Model Name"] = metrics_df["Model Name"].replace(
                {
                    f"Covariates ({bb_short})": "Covariates only",
                }
            )
            control_phenotypes_for_biobank = set(metrics_df["Phenotype"])
            if tab_name == "control":
                metrics_df["Model Name"] = metrics_df["Model Name"].replace(
                    {
                        f"MoEPRS ({bb_short})": (
                            f"MoEPRS (irrelevant PRSs; {bb_short})"
                        ),
                        f"MultiPRS ({bb_short})": (
                            f"MultiPRS (irrelevant PRSs; {bb_short})"
                        ),
                    }
                )
            metrics_df["Biobank"] = BIOBANK_NAME_MAP[biobank]
            metrics_dfs.append(metrics_df)

            if tab_name == "standard":
                portability_df = extract_accuracy_data_all_phenotypes(
                    args.moe_model,
                    biobank,
                    train_biobank=other_biobank,
                    binary_metric=args.binary_metric,
                    analysis_table_id=analysis_table,
                    metric_kind=accuracy_metric_kind,
                    ref_model_biobank=accuracy_ref_model_biobank,
                    prediction_type="full",
                    dataset="test_data",
                    exclude_all_group=False,
                    add_training_biobank_to_model_name=True,
                    aggregate_single_prs=False,
                )
                portability_df = portability_df.loc[
                    (portability_df["Evaluation Group"] == "All")
                    & portability_df["Model Name"].isin(
                        {
                            f"MoEPRS ({other_bb_short})",
                            f"MultiPRS ({other_bb_short})",
                        }
                    )
                ].copy()
                portability_df["Biobank"] = BIOBANK_NAME_MAP[biobank]
                metrics_dfs.append(portability_df)

            if tab_name == "control" and paired_standard_analysis_ids:
                relevant_metrics_df = extract_accuracy_data_all_phenotypes(
                    args.moe_model,
                    biobank,
                    train_biobank=biobank,
                    binary_metric=args.binary_metric,
                    analysis_table_id="multitrait_prs_table",
                    keep_analyses=paired_standard_analysis_ids,
                    metric_kind=accuracy_metric_kind,
                    ref_model_biobank=accuracy_ref_model_biobank,
                    prediction_type="full",
                    dataset="test_data",
                    exclude_all_group=False,
                    add_training_biobank_to_model_name=True,
                    aggregate_single_prs=False,
                )
                relevant_metrics_df = _retain_disease_specific_prs(
                    relevant_metrics_df
                )
                relevant_metrics_df = relevant_metrics_df.loc[
                    (relevant_metrics_df["Evaluation Group"] == "All")
                    & relevant_metrics_df["Phenotype"].isin(
                        control_phenotypes_for_biobank
                    )
                    & relevant_metrics_df["Model Name"].isin(
                        {
                            f"MoEPRS ({bb_short})",
                            f"MultiPRS ({bb_short})",
                            "Disease-specific PRS",
                        }
                    )
                ].copy()
                relevant_metrics_df["Model Name"] = relevant_metrics_df[
                    "Model Name"
                ].replace(
                    {
                        f"MoEPRS ({bb_short})": (
                            f"MoEPRS (relevant PRSs; {bb_short})"
                        ),
                        f"MultiPRS ({bb_short})": (
                            f"MultiPRS (relevant PRSs; {bb_short})"
                        ),
                    }
                )
                relevant_metrics_df["Biobank"] = BIOBANK_NAME_MAP[biobank]
                metrics_dfs.append(relevant_metrics_df)

        metrics_dfs = pd.concat(metrics_dfs).reset_index()
        metrics_dfs = _drop_cartagene_sparse_trait_rows(metrics_dfs)
        metrics_dfs["Phenotype"] = metrics_dfs["Phenotype"].map(_shorten_disease_label)
        plot_phenotype_order = [_shorten_disease_label(p) for p in phenotype_order]

        accuracy_font_scale = (
            SECTION4_STANDARD_ACCURACY_FONT_SCALE
            if tab_name == "standard"
            else SECTION4_FONT_SCALE
        )
        accuracy_legend_fontsize = "medium" if tab_name == "standard" else None

        for plot_biobank in evaluation_biobanks:
            plot_bb_short = BIOBANK_NAME_MAP_SHORT[plot_biobank]
            plot_metrics_df = metrics_dfs.loc[
                metrics_dfs["Biobank"] == BIOBANK_NAME_MAP[plot_biobank]
            ].copy()
            if plot_metrics_df.empty:
                print(f"> Skipping {tab_name}/{plot_biobank}: no metrics to plot.")
                continue

            if tab_name == "control":
                plot_palette = control_palette
                plot_hue_order = [
                    h
                    for h in control_hue_order
                    if h in set(plot_metrics_df["Model Name"])
                ]
                test_models = [
                    (
                        f"MoEPRS (irrelevant PRSs; {plot_bb_short})",
                        f"MoEPRS (relevant PRSs; {plot_bb_short})",
                    ),
                    (
                        f"MultiPRS (irrelevant PRSs; {plot_bb_short})",
                        f"MultiPRS (relevant PRSs; {plot_bb_short})",
                    ),
                ]
                legend_title = "Model\n(PRS set; training biobank)"
            else:
                plot_palette = palette
                plot_hue_order = [
                    h for h in hue_order if h in set(plot_metrics_df["Model Name"])
                ]
                test_models = [
                    (
                        f"MoEPRS ({plot_bb_short})",
                        f"MultiPRS ({plot_bb_short})",
                    ),
                    (
                        f"MoEPRS ({plot_bb_short})",
                        "Disease-specific PRS",
                    ),
                ]
                legend_title = "Model Name\n(Training biobank)"

            present_phenotypes = set(plot_metrics_df["Phenotype"])
            with sns.plotting_context("paper", font_scale=accuracy_font_scale):
                g = plot_combined_accuracy_metrics(
                    plot_metrics_df,
                    output_f=(
                        "figures/section_4/"
                        f"accuracy_metrics_{tab_name}_{plot_biobank}.pdf"
                    ),
                    x="Phenotype",
                    metric=args.binary_metric,
                    palette=plot_palette,
                    order=[
                        p for p in plot_phenotype_order if p in present_phenotypes
                    ],
                    hue_order=plot_hue_order,
                    column=None,
                    row=None,
                    height=SECTION4_ACCURACY_PANEL_FIGSIZE[1],
                    aspect=SECTION4_ACCURACY_PANEL_FIGSIZE[0]
                    / SECTION4_ACCURACY_PANEL_FIGSIZE[1],
                    sharey=False,
                    test_models=test_models,
                    significance_symbols=["*", "+"],
                    x_tick_rotation=30,
                    legend_title=legend_title,
                    legend_fontsize=accuracy_legend_fontsize,
                    legend_title_fontsize=accuracy_legend_fontsize,
                    ylim=(0.5, 1.0) if args.binary_metric == "AUROC" else None,
                    title=BIOBANK_NAME_MAP[plot_biobank],
                )

    plot_minority_ancestry_accuracy_from_eval_metrics(
        moe_model_name=args.moe_model,
        analysis_ids=[
            a
            for a in ANALYSIS_TO_PHENOTYPE_MAP.keys()
            if ANALYSIS_TO_TABLE_MAP.get(a) == "multitrait_prs_table"
            and ANALYSIS_TO_PHENOTYPE_MAP.get(a, a) in phenotype_order
        ],
        biobanks=("ukbb",),
        dataset_by_biobank={"ukbb": "test_data"},
        train_biobank="test_biobank",
        metric=args.binary_metric,
        metric_kind=accuracy_metric_kind,
        ref_model_biobank=accuracy_ref_model_biobank,
        phenotype_order=phenotype_order,
        output_file=(
            "figures/section_4/"
            "accuracy_minority_ancestry_eval_metrics_all_mixed.pdf"
        ),
    )

    plot_minority_ancestry_accuracy_from_eval_metrics(
        moe_model_name=args.moe_model,
        analysis_ids=[
            a
            for a in ANALYSIS_TO_PHENOTYPE_MAP.keys()
            if ANALYSIS_TO_TABLE_MAP.get(a) == "control_multitrait_prs_table"
            and ANALYSIS_TO_PHENOTYPE_MAP.get(a, a) in phenotype_order
        ],
        biobanks=("ukbb",),
        dataset_by_biobank={"ukbb": "test_data"},
        train_biobank="test_biobank",
        metric=args.binary_metric,
        metric_kind=accuracy_metric_kind,
        ref_model_biobank=accuracy_ref_model_biobank,
        phenotype_order=phenotype_order,
        output_file=(
            "figures/section_4/"
            "accuracy_minority_ancestry_eval_metrics_control_mixed.pdf"
        ),
    )

    plot_disease_prs_mixing_weights_across_phenotypes(
        moe_model_name=args.moe_model,
        analysis_ids=[
            analysis_id
            for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP
            if ANALYSIS_TO_TABLE_MAP.get(analysis_id) == "multitrait_prs_table"
        ],
        test_biobank="ukbb",
        dataset="full_data",
        model_biobank="ukbb",
        model_dataset="train_data",
        fold=args.mixing_fold,
        phenotype_order=phenotype_order,
        max_strip_points=1200,
        random_state=42,
        figsize=SECTION4_HALF_PANEL_FIGSIZE,
        output_file=(
            "figures/section_4/"
            "mixing_weight_disease_prs_all_phenotypes_ukbb.pdf"
        ),
    )

    plot_mixing_quartile_metric_panels_across_phenotypes(
        moe_model_name=args.moe_model,
        analysis_ids=[
            analysis_id
            for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP
            if ANALYSIS_TO_TABLE_MAP.get(analysis_id) == "multitrait_prs_table"
        ],
        test_biobank="ukbb",
        dataset="test_data",
        model_biobank="ukbb",
        model_dataset="train_data",
        phenotype_order=phenotype_order,
        output_file=(
            "figures/section_4/"
            "mixing_weight_quartile_metric_panels_all_phenotypes_ukbb.pdf"
        ),
    )

    plot_mixing_quartile_metric_panels_across_phenotypes(
        moe_model_name=args.moe_model,
        analysis_ids=["CAD_MT", "HTN_MT", "T1D_MT", "T2D_MT", "GOUT_MT"],
        test_biobank="ukbb",
        dataset="test_data",
        model_biobank="ukbb",
        model_dataset="train_data",
        phenotype_order=phenotype_order,
        panel_order=[
            "Accuracy (AUROC)",
            "Mean age at recruitment",
            "Proportion male",
        ],
        figsize=SECTION4_HALF_TALL_PANEL_FIGSIZE,
        output_file=(
            "figures/section_4/"
            "mixing_weight_quartile_metric_panels_mini_ukbb.pdf"
        ),
    )

    control_analysis_ids = [
        analysis_id
        for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP
        if ANALYSIS_TO_TABLE_MAP.get(analysis_id)
        == "control_multitrait_prs_table"
    ]
    if len(control_analysis_ids) == 0:
        print(
            "> Skipping control mixing-weight figure: "
            "control_multitrait_prs_table is unavailable or empty."
        )
    else:
        paired_control_analysis_ids = []
        standard_analysis_ids = []
        for analysis_id in control_analysis_ids:
            if not analysis_id.endswith("_CTRL"):
                continue
            standard_analysis_id = analysis_id[:-5]
            if ANALYSIS_TO_TABLE_MAP.get(standard_analysis_id) != "multitrait_prs_table":
                continue
            paired_control_analysis_ids.append(analysis_id)
            standard_analysis_ids.append(standard_analysis_id)

        if len(paired_control_analysis_ids) == 0:
            raise ValueError(
                "No control analyses have matching standard multitrait analyses."
            )

        plot_disease_prs_mixing_weights_across_phenotypes(
            moe_model_name=args.moe_model,
            analysis_ids=paired_control_analysis_ids,
            comparison_analysis_ids=standard_analysis_ids,
            test_biobank="ukbb",
            dataset="full_data",
            model_biobank="ukbb",
            model_dataset="train_data",
            fold=args.mixing_fold,
            phenotype_order=phenotype_order,
            max_strip_points=1200,
            random_state=42,
            title=(
                "Standard vs control disease-specific PRS mixing weights across "
                "the full UKB cohort"
            ),
            output_file=(
                "figures/section_4/"
                "mixing_weight_disease_prs_control_phenotypes_ukbb.pdf"
            ),
        )

    # ---------------------------------------------------------------------------

    for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP.keys():
        if (
            ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            not in phenotype_order
            or ANALYSIS_TO_TABLE_MAP.get(analysis_id) != "multitrait_prs_table"
        ):
            continue

        for biobank in ("ukbb", "cartagene"):
            try:
                plot_disease_prs_age_sex_accuracy(
                    analysis_id=analysis_id,
                    test_biobank=biobank,
                    dataset="test_data",
                    metric="Liability_R2",
                    keep_ancestry=("EUR",),
                    model_biobank=biobank,
                    output_file=f"figures/section_4/accuracy_disease_prs_age_sex_{analysis_id}_{biobank}.png",
                )
            except Exception as e:
                print(f"Error: {analysis_id} | {biobank}")
                print(e)
                continue

    # ---------------- Plot PRS Mixture graphs ----------------

    print("> PRS Mixture Graphs")

    sns.set_context("paper", font_scale=2.)

    for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP.keys():
        if (
            ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            not in phenotype_order
            or ANALYSIS_TO_TABLE_MAP.get(analysis_id) not in analysis_tables.values()
        ):
            continue

        for biobank in ("ukbb",):
            try:
                data_path = _resolve_harmonized_dataset_path(
                    analysis_id,
                    biobank,
                    "full_data",
                    fold=args.mixing_fold,
                )
                model_path = _resolve_trained_model_path(
                    analysis_id,
                    biobank,
                    "train_data",
                    args.moe_model,
                    fold=args.mixing_fold,
                )
                p_dataset = PRSDataset.from_pickle(data_path)
                moe_model = MoEPRS.from_saved_model(model_path)
            except Exception as e:
                print(
                    f"> Skipping mixture graph for {analysis_id}/{biobank}: {e}",
                    file=sys.stderr,
                )
                continue

            # Filter to only include European samples:
            p_dataset.filter_samples(p_dataset.data["Ancestry"] == "EUR")

            # Generate the admixture graphs:
            plot_admixture_graphs(
                p_dataset,
                moe_model,
                title=f"PRS Mixture Graph for {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]} ({BIOBANK_NAME_MAP_SHORT[biobank]})",
                output_file=f"figures/section_4/mixture_graphs_{analysis_id}_{biobank}.png",
                subsample=True,
                agg_mechanism="sort",
                figsize=(SECTION4_ACCURACY_PANEL_FIGSIZE[0] / 3, 3.1),
            )
