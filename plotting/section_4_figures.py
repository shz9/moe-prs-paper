import argparse
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
from eval_utils import generate_predictions, subsample_to_prevalence
from evaluate_predictive_performance import evaluate_prs_models, stratified_evaluation
from moe import GroupMeanWeightedPRS, MoEPRS
from plot_pgs_admixture import plot_admixture_graphs
from plot_stratified_prediction_accuracy import extract_stratified_evaluation_metrics
from plot_utils import (
    ANALYSIS_TO_PHENOTYPE_MAP,
    ANALYSIS_TO_TABLE_MAP,
    BIOBANK_NAME_MAP,
    BIOBANK_NAME_MAP_SHORT,
    METRIC_NAME_MAP,
    MODEL_NAME_MAP,
    assign_models_consistent_colors,
)
from PRSDataset import PRSDataset
from section_2_figures import extract_accuracy_data_all_phenotypes

# -----------------------------------------------------------------------------------------


DISEASE_MATCH_FLAG_COL = "Is_Disease_Matched"


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

        matched = df.loc[df[DISEASE_MATCH_FLAG_COL].astype(bool)].copy()
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
    min_group_size=30,
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
        (e.g. "Liability_R2", "ROC_AUC", "PR_AUC").

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

    # For threshold_rules we evaluate each rule mask independently.
    rule_masks = (
        grouping["group_masks"]
        if partition_method == "threshold" and threshold_rules is not None
        else {}
    )

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

    if len(rule_masks) > 0:
        # Evaluate each rule mask as-is (independent groups).
        dat.set_backend("numpy")
        preds = generate_predictions(dat, trained_models)
        eval_frames = []

        all_df = evaluate_prs_models(
            dat,
            other_models=preds,
            metrics=requested_metrics,
            evaluate_base_models=True,
            min_group_size=min_group_size,
        )
        all_df["EvalCategory"] = "All"
        all_df["EvalGroup"] = "All"
        all_df["N"] = dat.N
        eval_frames.append(all_df)

        for group_label, msk in rule_masks.items():
            try:
                gdf = evaluate_prs_models(
                    dat,
                    other_models=preds,
                    mask=msk,
                    metrics=requested_metrics,
                    evaluate_base_models=True,
                    min_group_size=min_group_size,
                )
            except Exception:
                continue

            if gdf is None:
                continue

            gdf["EvalCategory"] = group_col
            gdf["EvalGroup"] = group_label
            gdf["N"] = int(np.sum(msk))
            eval_frames.append(gdf)

        if len(eval_frames) == 0:
            raise ValueError(
                "No valid evaluation groups were generated from threshold_rules."
            )

        eval_df = pd.concat(eval_frames, ignore_index=True)
    else:
        eval_df = stratified_evaluation(
            dat,
            trained_models=trained_models,
            metrics=requested_metrics,
            cat_group_cols=[group_col],
            min_group_size=min_group_size,
        )

    missing_metrics = [m for m in requested_metrics if m not in eval_df.columns]
    if missing_metrics:
        raise ValueError(
            f"Requested metric(s) not found in evaluation output: {missing_metrics}"
        )

    out_df = eval_df.copy()
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
    if len(rule_masks) > 0:
        for group_label, msk in rule_masks.items():
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
    metric="ROC_AUC",
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

        if metric_name == "ROC_AUC":
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
    min_group_size=30,
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
            f"figures/section_4/mixing_group_prevalence_subsampled_accuracy_"
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
    metrics = ["ROC_AUC", incremental_metric]

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
                    other_models=preds,
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

        sdf = pd.concat(eval_rows, ignore_index=True)
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

        if metric_name == "ROC_AUC":
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
            "ROC_AUC",
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
    output_file=None,
):
    """
    Plot prediction accuracy of the disease-specific PRS across sex and 3 age groups.
    Style is aligned with the LDL stratified-accuracy subpanel in section 3.
    The x-axis visually separates Sex vs Recruitment age strata.
    """

    if output_file is None:
        output_file = f"figures/section_4/disease_prs_age_sex_accuracy_{analysis_id}_{test_biobank}.png"

    if isinstance(keep_ancestry, str):
        keep_ancestry = [keep_ancestry]
    elif keep_ancestry is not None:
        keep_ancestry = list(keep_ancestry)

    eval_df = extract_stratified_evaluation_metrics(
        analysis_id=analysis_id,
        biobank=test_biobank,
        dataset=dataset,
        keep_ancestry=keep_ancestry,
        category=["SexG", "AgeGroup3"],
    )

    disease_prs = _get_disease_prs_name(analysis_id)
    plot_df = eval_df.loc[eval_df["PGS"] == disease_prs].copy()

    if plot_df.empty:
        available = sorted(eval_df["PGS"].dropna().unique().tolist())
        raise ValueError(
            f"Disease-specific PRS '{disease_prs}' not found for {analysis_id}. "
            f"Available PGS labels: {available}"
        )

    ordered_groups = ["Female", "Male", "Age<50", "Age 50–60", "Age>60"]
    display_label_map = {
        "Female": "Female",
        "Male": "Male",
        "Age<50": "<50",
        "Age 50–60": "50-60",
        "Age>60": ">60",
    }
    plot_df = plot_df.loc[plot_df["EvalGroup"].isin(ordered_groups)].copy()
    plot_df["EvalGroup"] = pd.Categorical(
        plot_df["EvalGroup"], categories=ordered_groups, ordered=True
    )
    plot_df = plot_df.sort_values("EvalGroup")

    prs_color = assign_models_consistent_colors([disease_prs]).get(
        disease_prs, "#4C78A8"
    )

    plt.figure(figsize=(5, 4))
    ax = sns.barplot(
        data=plot_df,
        x="EvalGroup",
        y=metric,
        color=prs_color,
        order=ordered_groups,
        errorbar=None,
    )

    metric_err_col = f"{metric}_err"
    if metric_err_col in plot_df.columns and plot_df[metric_err_col].notna().any():
        add_error_bars(ax, plot_df, x="EvalGroup", y=metric, order=ordered_groups)

    bb_short = BIOBANK_NAME_MAP_SHORT.get(test_biobank.lower(), test_biobank.upper())
    ax.set_title(
        f"{ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)} ({bb_short})"
    )
    ax.set_xlabel("")
    ax.set_xticklabels([display_label_map[g] for g in ordered_groups])
    # Separate Sex vs Recruitment-age groups.
    ax.axvline(1.5, color="grey", lw=1.0, ls="--", alpha=0.8, zorder=0)
    ax.text(
        0.2,
        -0.22,
        "Sex",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.text(
        0.7,
        -0.22,
        "Recruitment age",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.set_ylabel(METRIC_NAME_MAP.get(metric, metric.replace("_", " ")))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_minority_ancestry_accuracy_panels(
    moe_model_name,
    analysis_ids=None,
    biobanks=("ukbb", "cartagene"),
    dataset_by_biobank=None,
    disease_prs_name_map=None,
    metric="Nagelkerke_R2",
    min_group_size=30,
    phenotype_order=None,
    output_file=None,
):
    """
    Plot pooled non-EUR (Ancestry != EUR) performance across phenotypes and biobanks.
    Models shown: disease-specific PRS, MoEPRS, Weighted PRS, Weighted PRS (exc. disease).
    """

    if output_file is None:
        output_file = "figures/section_4/minority_ancestry_accuracy_metrics.eps"

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
                preds = generate_predictions(dat, trained_models)
                edf = evaluate_prs_models(
                    dat,
                    other_models=preds,
                    metrics=[metric],
                    evaluate_base_models=True,
                    min_group_size=min_group_size,
                )
            except Exception as e:
                print(e)
                continue

            if metric not in edf.columns:
                continue

            edf["PGS"] = edf["PGS"].map(lambda x: MODEL_NAME_MAP[analysis_id].get(x, x))

            keep_models = [
                moe_label,
                multiprs_label,
                disease_prs_name,
                weighted_label,
                weighted_excl_label,
            ]
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
            edf["Phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            edf["Biobank"] = BIOBANK_NAME_MAP.get(biobank, biobank.upper())

            cols = ["Model Name", "Phenotype", "Biobank", metric]
            if metric_err_col in edf.columns:
                cols.append(metric_err_col)
            plot_rows.append(edf[cols])

    if len(plot_rows) == 0:
        raise ValueError("No minority-ancestry evaluation results available to plot.")

    plot_df = pd.concat(plot_rows, ignore_index=True)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot figures of section 4 of manuscript"
    )

    parser.add_argument(
        "--moe-model",
        dest="moe_model",
        type=str,
        default="MoE-GS",
        help="The name of the MoE model to plot as reference.",
    )

    parser.add_argument(
        "--binary-metric",
        dest="binary_metric",
        type=str,
        choices=
        {"Liability_R2", "Nagelkerke_R2", "CoxSnell_R2","McFadden_R2",
            "Liability_Probit_R2", "Liability_Logit_R2"},
        default="Nagelkerke_R2",
        help="The metric to plot for binary phenotypes.",
    )

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)
    makedir("figures/section_4/")

    palette = {
        "MoEPRS (UKB)": "#375E97",
        "MultiPRS (UKB)": "#FFBB00",
        "Best Single Source PRS": "#BC80BD",
    }

    hue_order = ["MoEPRS (UKB)", "MultiPRS (UKB)", "Best Single Source PRS"]

    phenotype_order = [
        "Asthma",
        "Gout",
        "Coronary Artery Disease",
        "Heart Failure",
        "Atrial Fibrillation",
        "Hypertension",
        "Stroke",
        "Dementia",
        "Type 2 Diabetes",
        "Type 1 Diabetes",
    ]

    analysis_tables = {
        "standard": "multitrait_prs_table",
        "control": "control_multitrait_prs_table",
    }

    for tab_name, analysis_table in analysis_tables.items():
        metrics_dfs = []
        for biobank in ("ukbb", "cartagene"):
            bb_short = BIOBANK_NAME_MAP_SHORT[biobank]
            metrics_df = extract_accuracy_data_all_phenotypes(
                args.moe_model,
                biobank,
                binary_metric=args.binary_metric,
                analysis_table_id=analysis_table,
                dataset=["test_data", "full_data"][biobank == "cartagene"],
                exclude_all=False,
            )

            metrics_df = metrics_df.loc[metrics_df["Evaluation Group"] == "All"]
            metrics_df["Biobank"] = BIOBANK_NAME_MAP[biobank]
            metrics_dfs.append(metrics_df)

        metrics_dfs = pd.concat(metrics_dfs).reset_index()
        uniq_phenotypes = metrics_dfs["Phenotype"].unique()

        g = plot_combined_accuracy_metrics(
            metrics_dfs,
            output_f=f"figures/section_4/accuracy_metrics_{tab_name}.eps",
            x="Phenotype",
            palette=palette,
            order=[p for p in phenotype_order if p in uniq_phenotypes],
            hue_order=hue_order,
            column=None,
            row="Biobank",
            height=3,
            aspect=4,
            sharey=False,
            test_models=[
                ("MoEPRS (UKB)", "MultiPRS (UKB)"),
                ("MoEPRS (UKB)", "Best Single Source PRS"),
            ],
            significance_symbols=["*", "+"],
            x_tick_rotation=90,
        )

    plot_minority_ancestry_accuracy_panels(
        moe_model_name=args.moe_model,
        analysis_ids=[
            a
            for a in ANALYSIS_TO_PHENOTYPE_MAP.keys()
            if ANALYSIS_TO_TABLE_MAP.get(a) == "multitrait_prs_table"
            and ANALYSIS_TO_PHENOTYPE_MAP.get(a, a) in phenotype_order
        ],
        biobanks=("ukbb", "cartagene"),
        dataset_by_biobank={"ukbb": "test_data", "cartagene": "full_data"},
        metric=args.binary_metric,
        phenotype_order=phenotype_order,
        output_file="figures/section_4/minority_ancestry_accuracy_metrics.eps",
    )

    # ---------------- Plot explanatory graphs ----------------

    for analysis_id in [
        "CAD_MT",
        "CAD_MT_CTRL",
        "HTN_MT",
        "T2D_MT",
        "AF_MT",
        "ASTHMA_MT",
    ]:
        plot_binary_mixing_group_panels(
            moe_model_name=args.moe_model,
            analysis_id=analysis_id,
            test_biobank="ukbb",
            dataset="test_data",
            incremental_metric=args.binary_metric,
            partition_method="quartile",
            n_quantiles=4,
            output_file=f"figures/section_4/group_summary_{analysis_id}_ukbb.png",
        )

        plot_prevalence_subsampled_mixing_accuracy_panels(
            moe_model_name=args.moe_model,
            analysis_id=analysis_id,
            test_biobank="ukbb",
            incremental_metric=args.binary_metric,
            dataset="test_data",
            partition_method="quartile",
            n_quantiles=4,
            output_file=f"figures/section_4/group_accuracy_prevalence_{analysis_id}_ukbb.png",
        )

    for analysis_id in ["HF_MT"]:
        plot_binary_mixing_group_panels(
            moe_model_name=args.moe_model,
            analysis_id=analysis_id,
            test_biobank="ukbb",
            dataset="test_data",
            partition_method="threshold",
            threshold_rules=["<=0.05", "<=0.1", "<=0.25", "<=0.5", ">0.5"],
            output_file=f"figures/section_4/group_summary_{analysis_id}_threshold_ukbb.png",
        )

    # ---------------------------------------------------------------------------
    # Figure out what's going on with ASTHMA
    plot_binary_mixing_group_panels(
        moe_model_name=args.moe_model,
        analysis_id="ASTHMA_MT",
        test_biobank="ukbb",
        dataset="test_data",
        disease_prs_name="ALLERGY",
        partition_method="threshold",
        threshold_rules=[">0.1", ">0.25", ">0.5"],
        output_file="figures/section_4/group_summary_ASTHMA_ALLERGY_threshold_ukbb.png",
    )

    # ---------------------------------------------------------------------------
    # Figure out what's going on with stroke:
    plot_binary_mixing_group_panels(
        moe_model_name=args.moe_model,
        analysis_id="STR_MT",
        test_biobank="ukbb",
        dataset="test_data",
        disease_prs_name="STR_433",
        partition_method="threshold",
        threshold_rules=["<=0.05", "<=0.1", "<=0.25", "<=0.5", ">0.5"],
        output_file="figures/section_4/group_summary_STR433_threshold_ukbb.png",
    )

    # Figure out what's going on with STR_433
    plot_binary_mixing_group_panels(
        moe_model_name=args.moe_model,
        analysis_id="STR_MT",
        test_biobank="ukbb",
        dataset="test_data",
        disease_prs_name="STR_433.1",
        partition_method="threshold",
        threshold_rules=["<=0.1", ">0.1"],
        output_file="figures/section_4/group_summary_STR433_1_threshold_ukbb.png",
    )

    # ---------------------------------------------------------------------------

    for analysis_id in ["CAD_MT", "HTN_MT", "T2D_MT"]:
        for biobank in ("ukbb", "cartagene"):
            plot_disease_prs_age_sex_accuracy(
                analysis_id=analysis_id,
                test_biobank=biobank,
                dataset="test_data",
                metric=args.binary_metric,
                keep_ancestry=("EUR",),
                output_file=f"figures/section_4/disease_prs_age_sex_accuracy_{analysis_id}_{biobank}.png",
            )

    # ---------------- Plot PRS Mixture graphs ----------------

    for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP.keys():
        if (
            ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            not in phenotype_order
            or ANALYSIS_TO_TABLE_MAP.get(analysis_id) not in analysis_tables.values()
        ):
            continue

        for biobank in ("ukbb", "cartagene"):
            data_path = f"data/harmonized_data/{analysis_id}/{biobank}/test_data.pkl"
            model_path = f"data/trained_models/{analysis_id}/{biobank}/train_data/{args.moe_model}.pkl"

            try:
                p_dataset = PRSDataset.from_pickle(data_path)
            except Exception as e:
                continue

            # Filter to only include European samples:
            p_dataset.filter_samples(p_dataset.data["Ancestry"] == "EUR")

            moe_model = MoEPRS.from_saved_model(model_path)

            # Generate the admixture graphs:
            plot_admixture_graphs(
                p_dataset,
                moe_model,
                title=f"PRS Mixture Graph for {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]} ({BIOBANK_NAME_MAP_SHORT[biobank]})",
                output_file=f"figures/section_4/mixture_graphs_{analysis_id}_{biobank}.png",
                subsample=True,
                agg_mechanism="sort",
                figsize=(g.fig.get_size_inches()[0] // 3, 3.1),
            )
