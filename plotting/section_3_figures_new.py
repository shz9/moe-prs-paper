import argparse
import glob
import os.path as osp
import sys
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
from plot_stratified_prediction_accuracy import extract_stratified_evaluation_metrics
from plot_utils import (
    ANALYSIS_TO_TABLE_MAP,
    ANALYSIS_TO_PHENOTYPE_MAP,
    BIOBANK_NAME_MAP_SHORT,
    MODEL_NAME_MAP,
    assign_models_consistent_colors,
)
from PRSDataset import PRSDataset
from section_2_figures import extract_accuracy_data_all_phenotypes


def plot_age_and_sex_stratified_mixing_weights(
    moe_model_name,
    analysis_id,
    biobank="ukbb",
    stratified_model="EUR",
):
    dataset = PRSDataset.from_pickle(
        f"data/harmonized_data/{analysis_id}/{biobank}/test_data.pkl"
    )

    # Keep only European samples:
    dataset.filter_samples(dataset.data["Ancestry"] == "EUR")

    moe_model = MoEPRS.from_saved_model(
        f"data/trained_models/{analysis_id}/{biobank}/train_data/{moe_model_name}.pkl"
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

    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        data=weights_df,
        x="Age",
        y=stratified_model,
        hue="Sex",
        palette={"Male": "#A1BE95", "Female": "#F98866"},
        alpha=0.7,
    )
    plt.xlabel("Age at recruitment")
    plt.ylabel(f"Mixing weight for {stratified_model} PRS")
    plt.title(
        f"Mixing weights for {stratified_model} PRS for {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]}\nin samples of "
        f"European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})"
    )
    plt.tight_layout()
    plt.savefig(f"figures/section_3_new/weights_{analysis_id}_{biobank}.png", dpi=300)
    plt.close()


def generate_stratified_metrics_figures(
    analysis_id,
    biobank="ukbb",
    keep_ancestry=("EUR",),
    category=("SexG", "AgeGroup3"),
):
    # -----------------------------------------------------------------
    metrics_df = extract_stratified_evaluation_metrics(
        analysis_id,
        biobank=biobank,
        keep_ancestry=keep_ancestry,
        category=category,
    )
    metrics_df = metrics_df.loc[metrics_df["PGS"].isin(keep_ancestry)]
    metrics_df = metrics_df.loc[metrics_df["EvalGroup"] != "All"]
    metrics_df.rename(columns={"PGS": "Stratified PRS"}, inplace=True)
    metrics_df = metrics_df.reset_index(drop=True)

    plt.figure(figsize=(5, 5))
    g = sns.barplot(
        metrics_df,
        x="EvalGroup",
        y="Incremental_R2",
        hue="Stratified PRS",
        palette=assign_models_consistent_colors(metrics_df["Stratified PRS"].unique()),
        order=["Female", "Male", "Age<50", "Age 50–60", "Age>60"],
    )

    add_error_bars(g, metrics_df, x="EvalGroup", y="Incremental_R2", hue_order=["EUR"])

    plt.title(
        f"Prediction accuracy on {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]}\nin samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})"
    )

    plt.xlabel("Evaluation Group")
    plt.ylabel("Incremental $R^2$")

    plt.tight_layout()
    plt.savefig(
        f"figures/section_3_new/stratified_accuracy_{analysis_id}_{biobank}.eps"
    )
    plt.close()


def plot_medication_use_figures(analysis_id, biobank="ukbb"):
    metrics_df = extract_stratified_evaluation_metrics(
        analysis_id,
        biobank=biobank,
        keep_ancestry=["EUR"],
        category=["SexG", "AgeGroup3"],
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

    fig, ax1 = plt.subplots(figsize=(5, 5))
    width = 0.4

    x = np.arange(len(ordered_cats))
    y1 = metrics_df["Incremental_R2"].values
    y2 = med_prev["Proportion_Using_Medication"].values

    eur_color = assign_models_consistent_colors(["EUR"])["EUR"]
    bars1 = ax1.bar(x - width / 2, y1, width=width, color=eur_color, label="Quantity 1")
    ax1.set_ylabel("Incremental $R^2$", color="#3C6B64")
    ax1.tick_params(axis="y", labelcolor="#3C6B64")

    # Add the appropriate tick positions to shift the error bars:
    ax1.set_xticks(x - width / 2)
    ax1.set_xticklabels(ordered_cats)

    add_error_bars(ax1, metrics_df, x="EvalGroup", y="Incremental_R2")

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

    plt.tight_layout()

    plt.savefig(
        f"figures/section_3_new/medication_use_accuracy_{analysis_id}_{biobank}.eps"
    )
    plt.close()


def extract_mixing_weight_similarity_across_analyses(
    moe_model_name,
    biobank,
    ref_analysis="HEIGHT_MA",
    metric="cosine",  # "cosine" or "jsd"
):
    unique_analysis = []
    for analysis_id, table_id in ANALYSIS_TO_TABLE_MAP.items():
        if table_id != "multi_ancestry_prs_table":
            continue
        model_f = (
            f"data/trained_models/{analysis_id}/{biobank}/train_data/{moe_model_name}.pkl"
        )
        if osp.exists(model_f):
            unique_analysis.append(model_f)

    unique_analysis.append("Ancestry classifier")

    ref_dataset = PRSDataset.from_pickle(
        f"data/harmonized_data/{ref_analysis}/{biobank}/full_data.pkl"
    )

    sim_result = []

    for models in combinations(unique_analysis, 2):
        proba = []
        phenotypes = []

        # Load the probability predictions from the models:
        for m in models:
            if m == "Ancestry classifier":
                proba.append(
                    ref_dataset.data[["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]].copy()
                )

                phenotypes.append("Ancestry classifier")

            else:
                moe_model = MoEPRS.from_saved_model(m)

                # For compatability, update the scaler to only keep the gating model input features:
                moe_model.data_scaler = subset_standard_scaler(
                    moe_model.data_scaler,
                    [c for c in moe_model.gate_input_cols if c != "Sex"],
                )

                analysis_id = m.split("/")[2]
                phenotypes.append(ANALYSIS_TO_PHENOTYPE_MAP[analysis_id])

                proba.append(
                    pd.DataFrame(
                        moe_model.predict_proba(ref_dataset),
                        columns=[
                            MODEL_NAME_MAP[analysis_id][c]
                            for c in moe_model.expert_cols
                        ],
                    )
                )

        # Keep shared columns across the two models:
        shared_cols = list(
            set(list(proba[0].columns)).intersection(set(list(proba[1].columns)))
        )

        for i in range(len(proba)):
            proba[i] = proba[i][shared_cols].values
            proba[i] /= np.clip(proba[i].sum(axis=1).reshape(-1, 1), 1e-6, None)

        masks = {
            "All": np.arange(proba[0].shape[0]),
            "Non-European ancestry": ref_dataset.data["Ancestry"].values != "EUR",
            "Unassigned ancestry (OTH)": ref_dataset.data["Ancestry"].values == "OTH",
        }

        for msk, msk_val in masks.items():
            # Compute similarity:
            if metric == "cosine":
                similarity = rowwise_cosine_similarity(
                    proba[0][msk_val, :], proba[1][msk_val, :]
                )
            elif metric == "jsd":
                # jensenshannon returns Jensen-Shannon distance, so convert to similarity
                similarity = 1.0 - jensenshannon(
                    proba[0][msk_val, :], proba[1][msk_val, :], axis=1, base=2
                )
            else:
                raise ValueError("metric must be either 'cosine' or 'jsd'")

            sim_result.append(
                {
                    "Similarity": np.mean(similarity),
                    "Cohort": msk,
                    "Phenotype 1": phenotypes[0],
                    "Phenotype 2": phenotypes[1],
                }
            )

    return pd.DataFrame(sim_result)


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

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.25)
    makedir("figures/section_3_new/")

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

    metric_names = {"cosine": "Cosine similarity", "jsd": "Jensen-Shannon similarity"}

    figure_width = 15

    # ----------------------------------------------------------

    # Extract and plot the similarity matrix across phenotypes:
    for biobank in ("ukbb", "cartagene"):
        sim_data = extract_mixing_weight_similarity_across_analyses(
            args.moe_model,
            biobank,
            ref_analysis="HEIGHT_MA",
            metric=args.sim_metric,
        )

        for cohort in sim_data["Cohort"].unique():
            plot_triangular_similarity_matrix(
                sim_data.loc[sim_data["Cohort"] == cohort].copy(),
                f"figures/section_3_new/similarity_matrix_{biobank}_{args.sim_metric}_{cohort}.eps",
                phenotype_order + ["Ancestry classifier"],
                title=f"Mixing weight similarity for ancestry-stratified PRS\nacrosss {len(phenotype_order)} phenotypes in {BIOBANK_NAME_MAP_SHORT[biobank]}\n{cohort} samples",
                metric_name=f"Mean {metric_names[args.sim_metric]}",
                figsize=(figure_width // 2, figure_width // 2),
            )

        # ------------------------------------------------------------------------------------
        phenotype_order_adj = [ANALYSIS_TO_PHENOTYPE_MAP[p] for p in adj_pheno_analyses]

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
                f"figures/section_3_new/similarity_matrix_{biobank}_{args.sim_metric}_{cohort}_ADJ_phenotypes.eps",
                phenotype_order_adj_mod + ["Ancestry classifier"],
                title=title,
                metric_name=f"Mean {metric_names[args.sim_metric]}",
                figsize=(figure_width // 3, figure_width // 3),
            )

        sns.set_context("paper", font_scale=1.25)

        # ------------------------------------------------------------------------------------

        phenotype_order_all = phenotype_order + phenotype_order_adj

        for cohort in sim_data["Cohort"].unique():
            plot_triangular_similarity_matrix(
                sim_data.loc[sim_data["Cohort"] == cohort].copy(),
                f"figures/section_3_new/similarity_matrix_{biobank}_{args.sim_metric}_{cohort}_all_phenotypes.eps",
                phenotype_order_all + ["Ancestry classifier"],
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
            data_path = f"data/harmonized_data/{analysis_id}/{biobank}/test_data.pkl"
            model_path = f"data/trained_models/{analysis_id}/{biobank}/train_data/{args.moe_model}.pkl"

            p_dataset = PRSDataset.from_pickle(data_path)
            moe_model = MoEPRS.from_saved_model(model_path)

            # Generate the admixture graphs:
            plot_admixture_graphs(
                p_dataset,
                moe_model,
                group_col="Ancestry",
                title=f"PRS Mixture Graph for {ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]} ({BIOBANK_NAME_MAP_SHORT[biobank]})",
                output_file=f"figures/section_3_new/mixture_graphs_{analysis_id}_{biobank}.png",
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
        "Best Single Source PRS",
        "Ancestry-weighted PRS",
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
        )

        metrics_df["Evaluation Group"] = (
            "Samples of "
            + metrics_df["Evaluation Group"].map(
                {"EUR": "European", "non-EUR": "minority (non-EUR)"}
            )
            + f" ancestry in {BIOBANK_NAME_MAP_SHORT[biobank]}"
        )

        g = plot_combined_accuracy_metrics(
            metrics_df,
            output_f=f"figures/section_3_new/accuracy_metrics_med_adj_{biobank}.eps",
            x="Phenotype",
            palette=palette,
            order=phenotype_order_adj,
            hue_order=hue_order,
            column=None,
            row="Evaluation Group",
            height=3,
            aspect=4,
            sharey=True,
            test_models=[
                (f"MoEPRS ({bb_short})", f"MultiPRS ({bb_short})"),
                (f"MoEPRS ({bb_short})", "Best Single Source PRS"),
            ],
            significance_symbols=["*", "+"],
            x_tick_rotation=90,
        )

    # ----------------------------------------------------------
    #
    sns.set_context("paper", font_scale=1.25)

    # Generate figures with stratified metrics:
    for adj_pheno in adj_pheno_analyses:
        for biobank in ("ukbb", "cartagene"):
            # plots for unadjusted phenotypes:
            generate_stratified_metrics_figures(
                adj_pheno.replace("ADJ_", ""), biobank=biobank
            )
            plot_medication_use_figures(adj_pheno.replace("ADJ_", ""), biobank=biobank)

            # plots for unadjusted phenotypes:
            generate_stratified_metrics_figures(adj_pheno, biobank=biobank)

            # Plot the gating model weights:
            plot_age_and_sex_stratified_mixing_weights(
                args.moe_model,
                adj_pheno,
                biobank=biobank,
            )
    """
