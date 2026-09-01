import argparse
import glob
import os.path as osp
import sys
from functools import lru_cache
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from magenpy.utils.system_utils import makedir
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import jensenshannon

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from combined_accuracy_plots import plot_combined_accuracy_metrics
from error_bars import add_error_bars
from eval_utils import rowwise_cosine_similarity
from model_utils import subset_standard_scaler
from moe import MoEPRS
from plot_pgs_admixture import plot_admixture_graphs
from plot_stratified_prediction_accuracy import estimate_stratified_evaluation_metrics
from plot_utils import (
    ANALYSIS_TO_PHENOTYPE_MAP,
    ANALYSIS_TO_TABLE_MAP,
    BIOBANK_NAME_MAP_SHORT,
    METRIC_NAME_MAP,
    MODEL_NAME_MAP,
    assign_models_consistent_colors,
    extract_accuracy_data_all_phenotypes,
)
from PRSDataset import PRSDataset


ANCESTRY_PROBA_COLS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]
SECTION3_FONT_SCALE = 1.0
SECTION3_HALF_PANEL_FIGSIZE = (7.2, 3.2)
SECTION3_QUARTER_PANEL_FIGSIZE = (3.45, 3.2)
CONCORDANCE_FIGSIZE = SECTION3_HALF_PANEL_FIGSIZE

PHENOTYPE_LABEL_MAP = {
    "Standing Height": "Height",
    "Log Body Mass Index": "Log(BMI)",
    "Diastolic blood pressure": "DBP",
    "Systolic blood pressure": "SBP",
    "Type 2 Diabetes": "T2D",
    "Asthma": "Asthma",
    "Log triglycerides": "Log(TG)",
    "Log HDL Cholesterol": "Log(HDL-C)",
    "LDL Cholesterol": "LDL-C",
    "Total Cholesterol": "TC",
    "LDL Cholesterol (Adj.)": "LDL-C",
    "Total Cholesterol (Adj.)": "TC",
    "Diastolic blood pressure (Adj.)": "DBP",
    "Systolic blood pressure (Adj.)": "SBP",
}


def shorten_phenotype_label(label):
    return PHENOTYPE_LABEL_MAP.get(label, label.replace(" (Adj.)", ""))


def plot_age_and_sex_stratified_mixing_weights(
    moe_model_name,
    analysis_id,
    biobank="ukbb",
    stratified_model="EUR",
    figsize=(5, 5),
    fold="fold_1",
):
    dataset = PRSDataset.from_pickle(
        f"data/harmonized_data/{analysis_id}/{biobank}/full_data.pkl"
    )

    # Keep only European samples:
    dataset.filter_samples(dataset.data["Ancestry"] == "EUR")

    moe_model = MoEPRS.from_saved_model(
        f"data/trained_models/{analysis_id}/{biobank}/"
        f"{fold}/train_data/{moe_model_name}.pkl"
    )

    # Extract weight and sex data for the individuals:
    weights_df = pd.DataFrame(
        dataset.get_data_columns(["Age", "Sex"]),
        columns=["Age", "Sex"],
    )

    weights_df["Sex"] = np.array(["Female", "Male"])[
        weights_df["Sex"].values.astype(int)
    ]

    prs_col_names = [
        MODEL_NAME_MAP[analysis_id][prs_col] for prs_col in dataset.prs_cols
    ]
    weights_df[prs_col_names] = moe_model.predict_proba(dataset)

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=weights_df,
        x="Age",
        y=stratified_model,
        hue="Sex",
        palette={"Male": "#A1BE95", "Female": "#F98866"},
        alpha=0.7,
        ax=ax,
    )
    ax.set_xlabel("Age at recruitment")
    ax.set_ylabel(f"Mixing weight for {stratified_model} PRS")
    ax.set_title(
        f"Mixing weights for {stratified_model} PRS\n"
        f"{ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]} in samples of "
        f"European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})"
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(
        f"figures/section_3/mixing_weights_by_age_sex_{analysis_id}_{biobank}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def generate_stratified_metrics_figures(
    analysis_id,
    biobank="ukbb",
    keep_ancestry=("EUR",),
    category=("SexG", "AgeGroup3"),
    metric="Incremental_R2",
    figsize=(5, 5),
):
    # -----------------------------------------------------------------
    metrics_df = estimate_stratified_evaluation_metrics(
        analysis_id,
        biobank=biobank,
        keep_ancestry=keep_ancestry,
        category=category,
        metric=metric,
    )
    metrics_df = metrics_df.loc[metrics_df["PGS"].isin(keep_ancestry)]
    metrics_df = metrics_df.loc[metrics_df["EvalGroup"] != "All"]
    metrics_df.rename(columns={"PGS": "Stratified PRS"}, inplace=True)
    metrics_df = metrics_df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    g = sns.barplot(
        data=metrics_df,
        x="EvalGroup",
        y=metric,
        hue="Stratified PRS",
        palette=assign_models_consistent_colors(metrics_df["Stratified PRS"].unique()),
        order=["Female", "Male", "Age<50", "Age 50–60", "Age>60"],
        ax=ax,
    )

    add_error_bars(g, metrics_df, x="EvalGroup", y=metric, hue_order=["EUR"])

    ax.set_title(
        f"Prediction accuracy on {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]}\nin samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})"
    )

    ax.set_xlabel("Evaluation Group")
    ax.set_ylabel(METRIC_NAME_MAP.get(metric, metric))

    fig.tight_layout(pad=0.4)
    plt.savefig(
        f"figures/section_3/accuracy_stratified_{analysis_id}_{biobank}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close()


def plot_medication_use_figures(
    analysis_id,
    biobank="ukbb",
    metric="Incremental_R2",
    figsize=(5, 5),
):
    metrics_df = estimate_stratified_evaluation_metrics(
        analysis_id,
        biobank=biobank,
        keep_ancestry=["EUR"],
        category=["SexG", "AgeGroup3"],
        metric=metric,
    )
    metrics_df = metrics_df.loc[metrics_df["PGS"] == "EUR"]
    metrics_df = metrics_df.loc[metrics_df["EvalGroup"] != "All"]
    metrics_df.rename(columns={"PGS": "Stratified PGS"}, inplace=True)
    metrics_df = metrics_df.reset_index(drop=True)

    # Convert column to ordered categorical
    ordered_cats = ["Female", "Male", "Age<50", "Age 50–60", "Age>60"]
    metrics_df["EvalGroup"] = pd.Categorical(
        metrics_df["EvalGroup"], categories=ordered_cats, ordered=True
    )
    metrics_df = metrics_df.sort_values("EvalGroup")

    # -------------------------------------------
    # Extract the medication-use prevalence data:
    med_prev = pd.read_csv(f"data/misc/medication_prevalence_{biobank}.csv")

    if any([bp in analysis_id for bp in ("SBP", "DBP")]):
        med_name = "Blood pressure medication"
        med_prev = med_prev.loc[
            med_prev["Medication"] == "Blood pressure medication"
        ].copy()
    else:
        med_name = "Cholesterol medication"
        med_prev = med_prev.loc[
            med_prev["Medication"] == "Cholesterol lowering medication"
        ].copy()

    med_prev["Group"] = pd.Categorical(
        med_prev["Group"], categories=ordered_cats, ordered=True
    )
    med_prev = med_prev.sort_values("Group")

    # -------------------------------------------

    fig, ax1 = plt.subplots(figsize=figsize)
    width = 0.4

    x = np.arange(len(ordered_cats))
    y1 = metrics_df[metric].values
    y2 = med_prev["Proportion_Using_Medication"].values

    eur_color = assign_models_consistent_colors(["EUR"])["EUR"]
    bars1 = ax1.bar(x - width / 2, y1, width=width, color=eur_color, label="Quantity 1")
    ax1.set_ylabel(METRIC_NAME_MAP.get(metric, metric), color="#3C6B64")
    ax1.tick_params(axis="y", labelcolor="#3C6B64")

    # Add the appropriate tick positions to shift the error bars:
    ax1.set_xticks(x - width / 2)
    ax1.set_xticklabels(ordered_cats)

    add_error_bars(ax1, metrics_df, x="EvalGroup", y=metric)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        y2,
        width=width,
        color="#B56D7F",
        hatch="//",
        edgecolor="#DAB6BF",
        label="Quantity 2",
    )
    ax2.set_ylabel("Proportion of samples", color="#A03C56")
    ax2.tick_params(axis="y", labelcolor="#A03C56")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered_cats, rotation=30)
    ax1.set_xlabel("Evaluation Group")

    # Only include one bar per group in the legend
    custom_legend = [bars1[0], bars2[0]]
    labels = ["Prediction accuracy (EUR)", med_name + " use"]
    ax1.legend(custom_legend, labels, loc="upper left")

    ax1.set_ylim([0.0, np.max(y1) * 1.3])
    ax2.set_ylim([0.0, np.max(y2) * 1.3])
    plt.title(
        f"Prediction accuracy on {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]}\n"
        f"and prevalence of {med_name}\n"
        f"in samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})"
    )

    fig.tight_layout(pad=0.4)

    plt.savefig(
        f"figures/section_3/accuracy_medication_use_{analysis_id}_{biobank}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close()


def _fold_sort_key(fold):
    try:
        return (0, int(str(fold).removeprefix("fold_")))
    except ValueError:
        return (1, str(fold))


def _available_moe_folds(moe_model_name, biobank, ref_analysis):
    model_paths = glob.glob(
        f"data/trained_models/{ref_analysis}/{biobank}/fold_*/train_data/"
        f"{moe_model_name}.pkl"
    )
    return sorted(
        {
            osp.basename(osp.dirname(osp.dirname(model_path)))
            for model_path in model_paths
        },
        key=_fold_sort_key,
    )


@lru_cache(maxsize=32)
def _get_ref_moe_probability_tables(
    moe_model_name,
    biobank,
    ref_analysis,
    fold="fold_1",
):
    """
    Predict all available MoE mixing weights on the same reference dataset once.

    The section 3 concordance/similarity figures repeatedly compare the same
    model predictions on the same reference samples. Caching this avoids
    re-loading every model and re-running predict_proba for every pair.
    """

    ref_dataset = PRSDataset.from_pickle(
        f"data/harmonized_data/{ref_analysis}/{biobank}/full_data.pkl"
    )
    ancestry_proba = ref_dataset.data[ANCESTRY_PROBA_COLS].copy()
    ancestry = ref_dataset.data["Ancestry"].values.copy()

    model_entries = []
    for analysis_id, table_id in ANALYSIS_TO_TABLE_MAP.items():
        if table_id != "multi_ancestry_prs_table":
            continue

        model_f = (
            f"data/trained_models/{analysis_id}/{biobank}/{fold}/train_data/"
            f"{moe_model_name}.pkl"
        )
        if not osp.exists(model_f):
            continue

        moe_model = MoEPRS.from_saved_model(model_f)

        # For compatibility, update the scaler to only keep the gating model input features.
        moe_model.data_scaler = subset_standard_scaler(
            moe_model.data_scaler,
            [c for c in moe_model.gate_input_cols if c != "Sex"],
        )

        model_entries.append(
            {
                "analysis_id": analysis_id,
                "phenotype": ANALYSIS_TO_PHENOTYPE_MAP[analysis_id],
                "proba": pd.DataFrame(
                    moe_model.predict_proba(ref_dataset),
                    columns=[
                        MODEL_NAME_MAP.get(analysis_id, {}).get(c, c)
                        for c in moe_model.expert_cols
                    ],
                ),
            }
        )

    return ancestry_proba, ancestry, tuple(model_entries)


def _normalized_shared_arrays(left_proba, right_proba):
    shared_cols = sorted(set(left_proba.columns).intersection(right_proba.columns))
    if len(shared_cols) == 0:
        return None, None

    left_arr = left_proba[shared_cols].to_numpy(dtype=float, copy=True)
    right_arr = right_proba[shared_cols].to_numpy(dtype=float, copy=True)

    left_arr /= np.clip(left_arr.sum(axis=1).reshape(-1, 1), 1e-6, None)
    right_arr /= np.clip(right_arr.sum(axis=1).reshape(-1, 1), 1e-6, None)

    return left_arr, right_arr


def extract_mixing_weight_similarity_across_analyses(
    moe_model_name,
    biobank,
    ref_analysis="HEIGHT_MA",
    metric="cosine",  # "cosine" or "jsd"
    fold="fold_1",
):
    ancestry_proba, ancestry, model_entries = _get_ref_moe_probability_tables(
        moe_model_name, biobank, ref_analysis, fold
    )
    entries = list(model_entries) + [
        {
            "analysis_id": None,
            "phenotype": "Ancestry classifier",
            "proba": ancestry_proba,
        }
    ]

    sim_result = []

    for left, right in combinations(entries, 2):
        left_arr, right_arr = _normalized_shared_arrays(left["proba"], right["proba"])
        if left_arr is None:
            continue

        masks = {
            "All": np.arange(left_arr.shape[0]),
            "Non-European ancestry": ancestry != "EUR",
            "Unassigned ancestry (OTH)": ancestry == "OTH",
        }

        for msk, msk_val in masks.items():
            # Compute similarity:
            if metric == "cosine":
                similarity = rowwise_cosine_similarity(
                    left_arr[msk_val, :], right_arr[msk_val, :]
                )
            elif metric == "jsd":
                # jensenshannon returns Jensen-Shannon distance, so convert to similarity
                similarity = 1.0 - jensenshannon(
                    left_arr[msk_val, :], right_arr[msk_val, :], axis=1, base=2
                )
            else:
                raise ValueError("metric must be either 'cosine' or 'jsd'")

            sim_result.append(
                {
                    "Similarity": np.mean(similarity),
                    "Cohort": msk,
                    "Phenotype 1": left["phenotype"],
                    "Phenotype 2": right["phenotype"],
                }
            )

    return pd.DataFrame(
        sim_result,
        columns=["Similarity", "Cohort", "Phenotype 1", "Phenotype 2"],
    )


def extract_ancestry_classifier_concordance(
    moe_model_name,
    biobank,
    fold=None,
):
    """Compare each phenotype's fold models with ancestry on its own dataset."""
    sim_result = []

    for analysis_id, table_id in ANALYSIS_TO_TABLE_MAP.items():
        if table_id != "multi_ancestry_prs_table":
            continue

        dataset_f = (
            f"data/harmonized_data/{analysis_id}/{biobank}/full_data.pkl"
        )
        if not osp.exists(dataset_f):
            continue

        model_paths = glob.glob(
            f"data/trained_models/{analysis_id}/{biobank}/fold_*/train_data/"
            f"{moe_model_name}.pkl"
        )
        if fold is not None:
            model_paths = [
                model_path
                for model_path in model_paths
                if osp.basename(osp.dirname(osp.dirname(model_path))) == fold
            ]
        model_paths = sorted(
            model_paths,
            key=lambda path: _fold_sort_key(
                osp.basename(osp.dirname(osp.dirname(path)))
            ),
        )
        if len(model_paths) == 0:
            continue

        phenotype_dataset = PRSDataset.from_pickle(dataset_f)
        ancestry_proba = phenotype_dataset.data[ANCESTRY_PROBA_COLS].copy()
        ancestry = phenotype_dataset.data["Ancestry"].values.copy()
        masks = {
            "All": np.arange(len(ancestry)),
            "European": ancestry == "EUR",
            "Non-European": ancestry != "EUR",
        }

        for model_f in model_paths:
            model_fold = osp.basename(osp.dirname(osp.dirname(model_f)))
            moe_model = MoEPRS.from_saved_model(model_f)
            moe_model.data_scaler = subset_standard_scaler(
                moe_model.data_scaler,
                [c for c in moe_model.gate_input_cols if c != "Sex"],
            )
            model_proba = pd.DataFrame(
                moe_model.predict_proba(phenotype_dataset),
                columns=[
                    MODEL_NAME_MAP.get(analysis_id, {}).get(c, c)
                    for c in moe_model.expert_cols
                ],
            )
            model_arr, classifier_arr = _normalized_shared_arrays(
                model_proba, ancestry_proba
            )
            if model_arr is None:
                continue

            for cohort, msk in masks.items():
                similarity = rowwise_cosine_similarity(
                    model_arr[msk, :], classifier_arr[msk, :]
                )
                sim_result.append(
                    {
                        "analysis_id": analysis_id,
                        "Phenotype": ANALYSIS_TO_PHENOTYPE_MAP[analysis_id],
                        "Cohort": cohort,
                        "Similarity": np.mean(similarity),
                        "Biobank": BIOBANK_NAME_MAP_SHORT[biobank],
                        "Fold": model_fold,
                    }
                )

    return pd.DataFrame(
        sim_result,
        columns=[
            "analysis_id",
            "Phenotype",
            "Cohort",
            "Similarity",
            "Biobank",
            "Fold",
        ],
    )


def plot_ancestry_classifier_concordance(
    sim_df,
    output_path,
    phenotype_order,
    title=None,
):
    if len(sim_df) == 0:
        print(f"> Skipping ancestry classifier concordance plot: no data for {output_path}")
        return

    cohort_order = ["All", "European", "Non-European"]
    cohort_palette = {
        "All": "#9FBAD6",
        "European": "#5F7FA6",
        "Non-European": "#CD9395",
    }
    phenotype_order = [
        p for p in phenotype_order if p in set(sim_df["Phenotype"].dropna().unique())
    ]
    plot_df = sim_df.copy()
    plot_df["Phenotype Label"] = plot_df["Phenotype"].map(shorten_phenotype_label)
    phenotype_label_order = [shorten_phenotype_label(p) for p in phenotype_order]

    with sns.plotting_context("paper", font_scale=SECTION3_FONT_SCALE):
        fig, ax = plt.subplots(figsize=CONCORDANCE_FIGSIZE)

        sns.pointplot(
            data=plot_df,
            x="Phenotype Label",
            y="Similarity",
            hue="Cohort",
            order=phenotype_label_order,
            hue_order=cohort_order,
            palette=cohort_palette,
            dodge=0.35,
            markers="o",
            linestyles="",
            errorbar="se",
            capsize=0,
            ax=ax,
        )

        ax.set_ylabel("Mean cosine similarity")
        ax.set_xlabel("")
        ax.set_ylim(0.0, 1.03)
        ax.set_yticks(np.linspace(0.0, 1.0, 5))
        ax.tick_params(axis="x", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.legend(
            title="Test cohort",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            borderaxespad=0,
            fontsize="x-small",
            title_fontsize="x-small",
        )

        if title is not None:
            ax.set_title(title, pad=14)

        fig.subplots_adjust(left=0.13, right=0.78, bottom=0.24, top=0.80)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_medication_adjusted_ancestry_classifier_concordance(
    sim_df,
    adj_analysis_ids,
    output_path,
    title=None,
):
    if len(sim_df) == 0 or "analysis_id" not in sim_df.columns:
        print(
            "> Skipping medication-adjusted ancestry classifier concordance plot: "
            f"no data for {output_path}"
        )
        return

    plot_dfs = []
    phenotype_order = []

    for adj_analysis_id in adj_analysis_ids:
        base_analysis_id = adj_analysis_id.replace("ADJ_", "")
        base_label = ANALYSIS_TO_PHENOTYPE_MAP.get(base_analysis_id, base_analysis_id)
        phenotype_order.append(base_label)

        for analysis_id, adjustment in (
            (base_analysis_id, "Unadjusted"),
            (adj_analysis_id, "Medication-adjusted"),
        ):
            df = sim_df.loc[sim_df["analysis_id"] == analysis_id].copy()
            if len(df) == 0:
                continue
            df["Phenotype"] = base_label
            df["Adjustment"] = adjustment
            plot_dfs.append(df)

    if len(plot_dfs) == 0:
        print(
            f"> Skipping medication-adjusted ancestry classifier concordance plot: "
            f"no data for {output_path}"
        )
        return

    plot_df = pd.concat(plot_dfs, ignore_index=True)
    cohort_order = ["All", "European", "Non-European"]
    adjustment_order = ["Unadjusted", "Medication-adjusted"]
    cohort_palette = {
        "All": "#9FBAD6",
        "European": "#5F7FA6",
        "Non-European": "#CD9395",
    }
    adjustment_marker = {
        "Unadjusted": "o",
        "Medication-adjusted": "X",
    }
    phenotype_order = [
        p for p in phenotype_order if p in set(plot_df["Phenotype"].dropna().unique())
    ]
    phenotype_label_order = [shorten_phenotype_label(p) for p in phenotype_order]
    plot_df["Phenotype Label"] = plot_df["Phenotype"].map(shorten_phenotype_label)
    x_pos = {p: i for i, p in enumerate(phenotype_order)}
    cohort_offset = {
        cohort: offset
        for cohort, offset in zip(cohort_order, np.linspace(-0.24, 0.24, 3))
    }
    adjustment_offset = {
        adjustment: offset
        for adjustment, offset in zip(adjustment_order, (-0.055, 0.055))
    }
    plot_df["PlotX"] = plot_df.apply(
        lambda r: x_pos[r["Phenotype"]]
        + cohort_offset[r["Cohort"]]
        + adjustment_offset[r["Adjustment"]],
        axis=1,
    )
    plot_df = (
        plot_df.groupby(
            ["analysis_id", "Phenotype", "Cohort", "Adjustment", "PlotX"],
            as_index=False,
            observed=True,
        )
        .agg(
            Similarity=("Similarity", "mean"),
            Similarity_SE=("Similarity", "sem"),
        )
    )

    with sns.plotting_context("paper", font_scale=SECTION3_FONT_SCALE):
        fig, ax = plt.subplots(figsize=CONCORDANCE_FIGSIZE)

        sns.scatterplot(
            data=plot_df,
            x="PlotX",
            y="Similarity",
            hue="Cohort",
            style="Adjustment",
            hue_order=cohort_order,
            style_order=adjustment_order,
            palette=cohort_palette,
            markers=adjustment_marker,
            s=80,
            linewidth=1.6,
            alpha=0.95,
            ax=ax,
        )
        for cohort, cohort_df in plot_df.groupby("Cohort", observed=True):
            cohort_df = cohort_df.loc[cohort_df["Similarity_SE"].notna()]
            if cohort_df.empty:
                continue
            ax.errorbar(
                cohort_df["PlotX"],
                cohort_df["Similarity"],
                yerr=cohort_df["Similarity_SE"],
                fmt="none",
                ecolor=cohort_palette[cohort],
                elinewidth=1.0,
                capsize=0,
                zorder=2,
            )

        ax.set_ylabel("Mean cosine similarity")
        ax.set_xlabel("")
        ax.set_ylim(0.0, 1.03)
        ax.set_yticks(np.linspace(0.0, 1.0, 5))
        ax.set_xticks(range(len(phenotype_order)))
        ax.set_xticklabels(phenotype_label_order)
        ax.tick_params(axis="x", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.legend(
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            borderaxespad=0,
            fontsize="x-small",
            title_fontsize="x-small",
        )

        if title is not None:
            ax.set_title(title, pad=24)

        fig.subplots_adjust(left=0.13, right=0.74, bottom=0.24, top=0.70)

        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_triangular_similarity_matrix(
    df,
    output_path,
    order,
    phenotype1_col="Phenotype 1",
    phenotype2_col="Phenotype 2",
    similarity_col="Similarity",
    title=None,
    metric_name="Similarity",
    cmap="viridis",
    fill_value=np.nan,
    figsize=(10, 8),
    annot=False,
    fmt=".2f",
):
    """
    Plot a triangular similarity matrix from a long-form dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain phenotype1_col, phenotype2_col, and similarity_col.
    order : list
        Desired phenotype order for both rows and columns.
    phenotype1_col, phenotype2_col, similarity_col : str
        Column names in df.
    title : str or None
        Plot title.
    metric_name : str
        Colorbar label.
    cmap : str
        Colormap.
    fill_value : float
        Value to fill missing cells with before plotting. Use np.nan to keep gaps.
    figsize : tuple
        Figure size.
    annot : bool
        Whether to annotate cells.
    fmt : str
        Annotation format.

    Returns
    -------
    matrix : pandas.DataFrame
        Symmetric similarity matrix ordered by `order`.
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Heatmap axis.
    """
    # Build square matrix
    matrix = df.pivot(
        index=phenotype1_col, columns=phenotype2_col, values=similarity_col
    )

    # Reindex to requested order
    matrix = matrix.reindex(index=order, columns=order)

    # Mirror to make symmetric
    matrix = matrix.combine_first(matrix.T)

    # Reindex again in case combine_first changed ordering
    matrix = matrix.reindex(index=order, columns=order)

    # Fill missing values if requested
    if not (isinstance(fill_value, float) and np.isnan(fill_value)):
        matrix = matrix.fillna(fill_value)

    # Create triangular mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Add a colorbar axis aligned to the heatmap axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)

    # Plot
    sns.heatmap(
        matrix,
        mask=mask,
        cmap=cmap,
        square=True,
        annot=annot,
        fmt=fmt,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": metric_name},
    )

    ax.set_ylabel("Mixing weight source")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Set the ancestry weights label to bold
    if len(ax.get_xticklabels()) > 0:
        ax.get_xticklabels()[-1].set_fontweight("bold")
    if len(ax.get_yticklabels()) > 0:
        ax.get_yticklabels()[-1].set_fontweight("bold")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return matrix, fig, ax


# -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot figures of section 3 of manuscript"
    )

    parser.add_argument(
        "--moe-model",
        dest="moe_model",
        type=str,
        default="MoE-GS",
        help="The name of the MoE model to plot as reference.",
    )

    parser.add_argument(
        "--similarity-metric",
        dest="sim_metric",
        type=str,
        default="cosine",
        choices={"cosine", "jsd"},
        help="The similarity metric for the mixing weights.",
    )

    parser.add_argument(
        "--mixing-weight-fold",
        dest="mixing_weight_fold",
        type=str,
        default="fold_1",
        help=(
            "Fold-trained model used for descriptive mixing-weight plots "
            "(default: fold_1). Concordance plots use every available fold."
        ),
    )

    args = parser.parse_args()

    sns.set_context("paper", font_scale=SECTION3_FONT_SCALE)
    makedir("figures/section_3/")

    # ----------------------------------------------------------

    phenotype_order = [
        "Standing Height",
        "Log Body Mass Index",
        "Diastolic blood pressure",
        "Systolic blood pressure",
        "Type 2 Diabetes",
        "Asthma",
        "Log triglycerides",
        "Log HDL Cholesterol",
        "LDL Cholesterol",
        "Total Cholesterol",
    ]

    adj_pheno_analyses = ["LDL_ADJ_MA", "TC_ADJ_MA", "DBP_ADJ_MA", "SBP_ADJ_MA"]
    phenotype_order_adj = [ANALYSIS_TO_PHENOTYPE_MAP[p] for p in adj_pheno_analyses]
    phenotype_order_all = phenotype_order + phenotype_order_adj

    metric_names = {"cosine": "Cosine similarity", "jsd": "Jensen-Shannon similarity"}

    figure_width = 15

    # ----------------------------------------------------------
    print(">>> Section 3 Figures <<<")

    plot_similarity_matrices = False

    # Extract and plot the similarity matrix across phenotypes:
    for biobank in ("ukbb", "cartagene"):
        if plot_similarity_matrices:
            concordance_folds = _available_moe_folds(
                args.moe_model,
                biobank,
                ref_analysis="HEIGHT_MA",
            )
            if len(concordance_folds) == 0:
                concordance_folds = [args.mixing_weight_fold]
            sim_data = pd.concat(
                [
                    extract_mixing_weight_similarity_across_analyses(
                        args.moe_model,
                        biobank,
                        ref_analysis="HEIGHT_MA",
                        metric=args.sim_metric,
                        fold=fold,
                    ).assign(Fold=fold)
                    for fold in concordance_folds
                ],
                ignore_index=True,
            )

        ancestry_classifier_sim = extract_ancestry_classifier_concordance(
            args.moe_model,
            biobank,
        )
        plot_ancestry_classifier_concordance(
            ancestry_classifier_sim,
            f"figures/section_3/ancestry_classifier_concordance_all_{biobank}.pdf",
            phenotype_order,
            title=(
                "Concordance between ancestry classifier and\n"
                f"MoEPRS mixing weights on {BIOBANK_NAME_MAP_SHORT[biobank]} samples"
            ),
        )
        plot_medication_adjusted_ancestry_classifier_concordance(
            ancestry_classifier_sim,
            adj_pheno_analyses,
            f"figures/section_3/ancestry_classifier_concordance_med_adj_phenotypes_{biobank}.pdf",
            title=(
                "Concordance between ancestry classifier and MoEPRS\nmixing weights for "
                f"medication-adjusted phenotypes ({BIOBANK_NAME_MAP_SHORT[biobank]} samples)"
            ),
        )

        """
        for cohort in sim_data["Cohort"].unique():
            plot_triangular_similarity_matrix(
                sim_data.loc[sim_data["Cohort"] == cohort].copy(),
                f"figures/section_3/similarity_matrix_{args.sim_metric}_{cohort}_all_{biobank}.pdf",
                [shorten_phenotype_label(p) for p in phenotype_order]
                + ["Ancestry classifier"],
                title=f"Mixing weight similarity for ancestry-stratified PRS\nacrosss {len(phenotype_order)} phenotypes in {BIOBANK_NAME_MAP_SHORT[biobank]}\n{cohort} samples",
                metric_name=f"Mean {metric_names[args.sim_metric]}",
                figsize=(figure_width // 2, figure_width // 2),
            )
        # ------------------------------------------------------------------------------------

        sim_data_adj = sim_data.loc[
            sim_data["Phenotype 1"].isin(phenotype_order_adj + ["Ancestry classifier"])
            & sim_data["Phenotype 2"].isin(
                phenotype_order_adj + ["Ancestry classifier"]
            )
        ].copy()

        sim_data_adj["Phenotype 1"] = sim_data_adj["Phenotype 1"].str.replace(
            " (Adj.)", ""
        )
        sim_data_adj["Phenotype 2"] = sim_data_adj["Phenotype 2"].str.replace(
            " (Adj.)", ""
        )

        phenotype_order_adj_mod = [
            p.replace(" (Adj.)", "") for p in phenotype_order_adj
        ]

        sns.set_context("paper", font_scale=1.0)

        for cohort in sim_data["Cohort"].unique():
            title = (
                f"Mixing weight similarity for ancestry-stratified PRS\n"
                f"across {len(phenotype_order_adj_mod)} "
                f"$\\mathbf{{medication-adjusted}}$ phenotypes in "
                f"{BIOBANK_NAME_MAP_SHORT[biobank]}\n"
                f"{cohort} samples"
            )
            plot_triangular_similarity_matrix(
                sim_data_adj.loc[sim_data_adj["Cohort"] == cohort].copy(),
                f"figures/section_3/similarity_matrix_{args.sim_metric}_{cohort}_adj_phenotypes_{biobank}.pdf",
                [shorten_phenotype_label(p) for p in phenotype_order_adj_mod]
                + ["Ancestry classifier"],
                title=title,
                metric_name=f"Mean {metric_names[args.sim_metric]}",
                figsize=(figure_width // 3, figure_width // 3),
            )

        sns.set_context("paper", font_scale=1.25)

        # ------------------------------------------------------------------------------------

        for cohort in sim_data["Cohort"].unique():
            plot_triangular_similarity_matrix(
                sim_data.loc[sim_data["Cohort"] == cohort].copy(),
                f"figures/section_3/similarity_matrix_{args.sim_metric}_{cohort}_all_phenotypes_{biobank}.pdf",
                [shorten_phenotype_label(p) for p in phenotype_order_all]
                + ["Ancestry classifier"],
                title=f"Mixing weight similarity for ancestry-stratified PRS\nacrosss {len(phenotype_order_all)} phenotypes in {BIOBANK_NAME_MAP_SHORT[biobank]}\n{cohort} samples",
                metric_name=f"Mean {metric_names[args.sim_metric]}",
                figsize=(figure_width // 2, figure_width // 2),
            )
    """
    # ----------------------------------------------------------
    # Plot mixture graphs for the medication-adjusted phenotypes:

    sns.set_context("paper", font_scale=1.5)

    for analysis_id in adj_pheno_analyses:
        for biobank in ("ukbb", "cartagene"):
            data_path = f"data/harmonized_data/{analysis_id}/{biobank}/full_data.pkl"
            model_path = (
                f"data/trained_models/{analysis_id}/{biobank}/"
                f"{args.mixing_weight_fold}/train_data/{args.moe_model}.pkl"
            )

            p_dataset = PRSDataset.from_pickle(data_path)
            moe_model = MoEPRS.from_saved_model(model_path)

            # Generate the admixture graphs:
            plot_admixture_graphs(
                p_dataset,
                moe_model,
                group_col="Ancestry",
                title=f"PRS Mixture Graph for {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]} ({BIOBANK_NAME_MAP_SHORT[biobank]})",
                output_file=f"figures/section_3/mixture_graphs_{analysis_id}_{biobank}.png",
                subsample=True,
                agg_mechanism="sort",
                figsize=(figure_width, 3.1),
            )

    # ----------------------------------------------------------
    # Plot accuracy metrics for medication-adjusted phenotypes:
    hue_order = [
        "MoEPRS (UKB)",
        "MoEPRS (CaG)",
        "MultiPRS (UKB)",
        "MultiPRS (CaG)",
        "Ancestry-weighted PRS",
        "Best Single Source PRS",
    ]

    palette = {
        "MoEPRS (UKB)": "#375E97",
        "MoEPRS (CaG)": "#8CA8D8",
        "MultiPRS (UKB)": "#FFBB00",
        "MultiPRS (CaG)": "#FFE066",
        "Best Single Source PRS": "#BC80BD",
        "Ancestry-weighted PRS": "#66C2A5",
    }

    for biobank in ("ukbb", "cartagene"):
        bb_short = BIOBANK_NAME_MAP_SHORT[biobank]
        metrics_df = extract_accuracy_data_all_phenotypes(
            args.moe_model,
            biobank,
            keep_analyses=["LDL_ADJ_MA", "TC_ADJ_MA", "DBP_ADJ_MA", "SBP_ADJ_MA"],
            metric_kind="incremental_vs_ref",
            ref_model_biobank="test_biobank",
            add_training_biobank_to_model_name=True,
        )

        metrics_df["Evaluation Group"] = (
            "Samples of "
            + metrics_df["Evaluation Group"].map(
                {"EUR": "European", "non-EUR": "minority (non-EUR)"}
            )
            + f" ancestry in {BIOBANK_NAME_MAP_SHORT[biobank]}"
        )
        metrics_df["Phenotype"] = metrics_df["Phenotype"].map(shorten_phenotype_label)
        plot_phenotype_order_adj = [
            shorten_phenotype_label(p) for p in phenotype_order_adj
        ]

        with sns.plotting_context("paper", font_scale=SECTION3_FONT_SCALE):
            g = plot_combined_accuracy_metrics(
                metrics_df,
                output_f=f"figures/section_3/accuracy_metrics_med_adj_all_{biobank}.pdf",
                x="Phenotype",
                palette=palette,
                order=plot_phenotype_order_adj,
                hue_order=hue_order,
                column=None,
                row="Evaluation Group",
                height=SECTION3_HALF_PANEL_FIGSIZE[1] / 2,
                aspect=SECTION3_HALF_PANEL_FIGSIZE[0]
                / (SECTION3_HALF_PANEL_FIGSIZE[1] / 2),
                sharey=True,
                test_models=[
                    (f"MoEPRS ({bb_short})", f"MultiPRS ({bb_short})"),
                    (f"MoEPRS ({bb_short})", "Best Single Source PRS"),
                    (f"MoEPRS ({bb_short})", "Ancestry-weighted PRS"),
                ],
                significance_symbols=["*", "+", "°"],
                x_tick_rotation=30,
                legend_title="Model Name\n(Training biobank)",
                legend_fontsize="medium",
                legend_title_fontsize="medium",
            )

    # ----------------------------------------------------------
    #
    sns.set_context("paper", font_scale=SECTION3_FONT_SCALE)

    # Generate figures with stratified metrics:
    for adj_pheno in adj_pheno_analyses:
        for biobank in ("ukbb", "cartagene"):
            # plots for unadjusted phenotypes:
            generate_stratified_metrics_figures(
                adj_pheno.replace("ADJ_", ""),
                biobank=biobank,
                metric="Incremental_R2",
                figsize=SECTION3_QUARTER_PANEL_FIGSIZE,
            )
            plot_medication_use_figures(
                adj_pheno.replace("ADJ_", ""),
                biobank=biobank,
                metric="Incremental_R2",
                figsize=SECTION3_QUARTER_PANEL_FIGSIZE,
            )

            # plots for unadjusted phenotypes:
            generate_stratified_metrics_figures(
                adj_pheno,
                biobank=biobank,
                metric="Incremental_R2",
                figsize=SECTION3_QUARTER_PANEL_FIGSIZE,
            )

            # Plot the gating model weights:
            plot_age_and_sex_stratified_mixing_weights(
                args.moe_model,
                adj_pheno,
                biobank=biobank,
                figsize=SECTION3_QUARTER_PANEL_FIGSIZE,
                fold=args.mixing_weight_fold,
            )
            plot_age_and_sex_stratified_mixing_weights(
                args.moe_model,
                adj_pheno.replace("ADJ_", ""),
                biobank=biobank,
                figsize=SECTION3_QUARTER_PANEL_FIGSIZE,
                fold=args.mixing_weight_fold,
            )
