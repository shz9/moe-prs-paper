import argparse
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from magenpy.utils.system_utils import makedir
from viprs.eval.continuous_metrics import (
    incremental_r2,
    pearson_r,
)

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from combined_accuracy_plots import plot_combined_accuracy_metrics
from moe import MoEPRS
from plot_pgs_admixture import plot_admixture_graphs
from plot_predictive_performance import postprocess_metrics_df
from plot_utils import (
    BIOBANK_NAME_MAP_SHORT,
    MODEL_NAME_MAP,
    PHENOTYPE_NAME_MAP,
    assign_ancestry_consistent_colors,
    assign_models_consistent_colors,
    read_eval_metrics,
    transform_eval_metrics,
)
from PRSDataset import PRSDataset


def plot_mixing_weight_comparison_quantitative(biobank="ukbb"):
    bbk_name = {"ukbb": "UKB", "cartagene": "CaG"}

    test_dat = PRSDataset.from_pickle(
        f"data/harmonized_data/HEIGHT/{biobank}/test_data.pkl"
    )

    ukb_model = MoEPRS.from_saved_model(
        f"data/trained_models/HEIGHT/ukbb/train_data/{args.moe_model}.pkl"
    )
    cag_model = MoEPRS.from_saved_model(
        f"data/trained_models/HEIGHT/cartagene/train_data/{args.moe_model}.pkl"
    )

    prob_ukb = ukb_model.predict_proba(test_dat)
    prob_cag = cag_model.predict_proba(test_dat)

    # Step 1: Extract mean weights data:
    mean_weighs_ukb = pd.DataFrame(
        {
            "Mean Mixing Weight": prob_ukb.mean(axis=0),
            "Stratified PRS": [
                MODEL_NAME_MAP[test_dat.prs_cols[i]] for i in range(prob_ukb.shape[1])
            ],
        }
    )
    mean_weighs_ukb["Training data"] = "UKB"

    mean_weighs_cag = pd.DataFrame(
        {
            "Mean Mixing Weight": prob_cag.mean(axis=0),
            "Stratified PRS": [
                MODEL_NAME_MAP[test_dat.prs_cols[i]] for i in range(prob_cag.shape[1])
            ],
        }
    )
    mean_weighs_cag["Training data"] = "CaG"

    mean_weights = pd.concat([mean_weighs_ukb, mean_weighs_cag], ignore_index=True)

    # Step 2: Extract Pearson correlation data:

    corr_data = []

    for i in range(prob_ukb.shape[1]):
        corr_data.append(
            {
                "Stratified PRS": MODEL_NAME_MAP[test_dat.prs_cols[i]],
                "Pearson $R$": np.corrcoef(prob_ukb[:, i], prob_cag[:, i])[0, 1],
            }
        )

    corr_data = pd.DataFrame(corr_data)

    # Step 3: Plot the data:

    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 2.9), sharey=True, gridspec_kw={"width_ratios": [1, 1.3]}
    )
    axes = axes.flatten()

    if biobank == "ukbb":
        x_order = list(
            mean_weighs_ukb.sort_values("Mean Mixing Weight")["Stratified PRS"].values
        )[::-1]
    else:
        x_order = list(
            mean_weighs_cag.sort_values("Mean Mixing Weight")["Stratified PRS"].values
        )[::-1]

    sns.barplot(
        data=mean_weights,
        y="Stratified PRS",
        x="Mean Mixing Weight",
        ax=axes[0],
        order=x_order,
        palette={"UKB": "#005f6f", "CaG": "#64daf3"},
        hue="Training data",
        orient="h",
    )
    axes[0].set_title("Mean mixing weights per PRS")

    sns.barplot(
        data=corr_data,
        y="Stratified PRS",
        x="Pearson $R$",
        ax=axes[1],
        order=x_order,
        hue="Stratified PRS",
        palette=assign_models_consistent_colors(corr_data["Stratified PRS"].unique()),
        orient="h",
    )
    axes[1].set_title("Cross-biobank correlation\nof mixing weights per PRS")

    mean_corr = corr_data["Pearson $R$"].mean()

    axes[1].axvline(mean_corr, color="gray", linestyle="--", linewidth=1)

    axes[1].text(
        y=axes[1].get_ylim()[0] - 0.05,
        x=mean_corr + 0.05,
        s=f"Mean={mean_corr:.2}",
        ha="left",
        va="bottom",
        fontsize="smaller",
        color="black",
        rotation=0,
    )

    axes[0].set_xscale("log")
    axes[1].set_xticks(
        np.arange(np.floor(corr_data["Pearson $R$"].min() * 4) / 4, 1.25, 0.25)
    )

    fig.suptitle(
        f"Concordance between UKB- and CaG-trained\n$MoEPRS$ "
        f"models on independent test samples in {bbk_name[biobank]}",
        y=0.9,
    )

    plt.tight_layout()
    plt.savefig(
        f"figures/section_2/mixing_weight_quant_comparison_{biobank}.png", dpi=400
    )
    plt.close()


def plot_mixing_weight_comparison(biobank="ukbb"):
    bbk_name = {"ukbb": "UKB", "cartagene": "CaG"}

    test_dat = PRSDataset.from_pickle(
        f"data/harmonized_data/HEIGHT/{biobank}/test_data.pkl"
    )

    ukb_model = MoEPRS.from_saved_model(
        f"data/trained_models/HEIGHT/ukbb/train_data/{args.moe_model}.pkl"
    )
    cag_model = MoEPRS.from_saved_model(
        f"data/trained_models/HEIGHT/cartagene/train_data/{args.moe_model}.pkl"
    )

    prob_ukb = ukb_model.predict_proba(test_dat)
    prob_cag = cag_model.predict_proba(test_dat)

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    handles, labels = [], []

    # Plot using seaborn.scatterplot with hue
    for i in range(prob_ukb.shape[1]):
        model_name = MODEL_NAME_MAP[test_dat.prs_cols[i]]
        x_axis = f"P({model_name}) - UKB-trained MoE"
        y_axis = f"P({model_name}) - CaG-trained MoE"

        df = pd.DataFrame(
            {
                x_axis: prob_ukb[:, i],
                y_axis: prob_cag[:, i],
                "Ancestry": test_dat.data["Ancestry"],
            }
        )

        ax = axes[i]

        # For the first subplot, extract handles/labels for the legend
        if i == 0:
            sns.scatterplot(
                data=df,
                x=x_axis,
                y=y_axis,
                hue="Ancestry",
                palette=assign_ancestry_consistent_colors(df["Ancestry"].unique()),
                ax=ax,
                legend="full",
                alpha=0.5,
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()  # Remove legend from subplot after extracting
        else:
            sns.scatterplot(
                data=df,
                x=x_axis,
                y=y_axis,
                hue="Ancestry",
                palette=assign_ancestry_consistent_colors(df["Ancestry"].unique()),
                ax=ax,
                legend=False,
                alpha=0.5,
            )

        ax.set_title(
            f"Comparison of mixing weights for {model_name} PRS\nPearson R={np.corrcoef(df[x_axis], df[y_axis])[0, 1]:.2}"
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, 1.0)

    # Use sixth subplot for shared legend
    axes[5].axis("off")
    legend = axes[5].legend(
        handles=handles,
        labels=labels,
        title="Sample ancestry",
        loc="center",
        frameon=False,
    )

    # Adjust the alpha for the legend handles:
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    fig.suptitle(
        f"Mixing weights for ancestry-stratified PRSs for Standing Height in {bbk_name[biobank]}"
    )

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"figures/section_2/mixing_weight_comparison_{biobank}.png", dpi=400)
    plt.close()


def extract_accuracy_data(
    moe_model_name,
    phenotype="HEIGHT",
    test_biobank="ukbb",
    metric="Incremental_R2",
    dataset="test_data",
    evaluation_category="Ancestry",
):
    # Extract accuracy metrics:
    f = f"data/evaluation/{phenotype}/{test_biobank}/{dataset}.csv"
    df = transform_eval_metrics(read_eval_metrics(f))

    df = df.loc[
        (df["Model Category"] != "MoE")
        | df["Model Name"].isin(
            [
                f"{moe_model_name} (ukbb)",
                f"{moe_model_name} (cartagene)",
            ]
        )
    ]

    df = df.loc[
        df["Model Category"].isin(
            ["MoE", "MultiPRS", "AncestryWeightedPRS"]
        )  # <- Ensemble model categories
        | (df["Training biobank"] == test_biobank.upper())
    ]

    # Rename the ensemble models for clarity:
    for m, m_new in {
        moe_model_name: "MoEPRS",
        "MultiPRS": "MultiPRS",
        "AncestryWeightedPRS": "AncestryWeightedPRS",
    }.items():
        df["Model Name"] = df["Model Name"].str.replace(
            f"{m} (ukbb)", f"{m_new} (UKB)", regex=False
        )
        df["Model Name"] = df["Model Name"].str.replace(
            f"{m} (cartagene)", f"{m_new} (CaG)", regex=False
        )

    dfs = postprocess_metrics_df(
        df,
        metric,
        category=evaluation_category,
        min_sample_size=50,
        aggregate_single_prs=True,
        include_cohort_matched=True,
    )

    dfs["Phenotype"] = PHENOTYPE_NAME_MAP[phenotype]
    dfs["Phenotype"] += f" ({BIOBANK_NAME_MAP_SHORT[test_biobank]})"

    return dfs


def plot_performance_on_ancestry_group(biobank="ukbb", ancestry="AMR"):
    dataset = PRSDataset.from_pickle(
        f"data/harmonized_data/HEIGHT/{biobank}/train_data.pkl"
    )
    dataset.standardize_data()
    dataset.filter_samples(dataset.data["Ancestry"] == ancestry)

    metrics = {}

    # Compute metrics:
    for prs in dataset.prs_cols:
        inc_r2 = incremental_r2(
            dataset.data["HEIGHT"].values,
            dataset.data[prs].values,
            covariates=pd.DataFrame(dataset.get_covariates()),
        )
        p_corr = pearson_r(dataset.data["HEIGHT"].values, dataset.data[prs].values)

        metrics[MODEL_NAME_MAP[prs]] = (
            f"{MODEL_NAME_MAP[prs]} (Incremental $R^2$= {inc_r2:.2}; Pearson R={p_corr:.2})"
        )

    melt_df = pd.melt(
        dataset.data[["IID", "HEIGHT"] + dataset.prs_ids],
        id_vars=["IID", "HEIGHT"],
        var_name="Stratified PGS",
        value_name="Polygenic Score",
    )
    melt_df["Stratified PRS"] = melt_df["Stratified PGS"].map(MODEL_NAME_MAP)

    plt.figure(figsize=(8, 6))

    ax = sns.scatterplot(
        data=melt_df,
        x="Polygenic Score",
        y="HEIGHT",
        hue="Stratified PRS",
        palette=assign_models_consistent_colors(melt_df["Stratified PRS"].unique()),
        alpha=0.5,
    )

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles=handles,
        labels=[metrics[label] for label in labels],
        title="Stratified PRS",
        fontsize="x-small",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        borderaxespad=0.0,
    )

    # Adjust the alpha for the legend handles:
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    ax.set_ylabel("Standing Height")
    bbk_name = {"ukbb": "UKB", "cartagene": "CaG"}[biobank]

    ax.set_title(
        f"Performance of stratified PRS for\nStanding Height on {ancestry} samples in {bbk_name}"
    )

    plt.tight_layout()
    plt.savefig(f"figures/section_2/perf_{ancestry}_{biobank}.png", dpi=400)
    plt.close()


# -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Figure 2 of manuscript")

    parser.add_argument(
        "--moe-model",
        dest="moe_model",
        type=str,
        default="MoE-GS",
        help="The name of the MoE model to plot as reference.",
    )

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)
    makedir("figures/section_2/")

    # ---------------- Plot accuracy subpanels ----------------

    ukb_data = extract_accuracy_data("ukbb")
    cag_data = extract_accuracy_data("cartagene")

    all_data = pd.concat([ukb_data, cag_data], axis=0).reset_index(drop=True)

    g = plot_combined_accuracy_metrics(
        all_data,
        output_f="figures/section_2/accuracy_subpanels.eps",
        col_order=["Standing Height (UKB)", "Standing Height (CaG)"],
        hue_order=[
            "MoEPRS (UKB)",
            "MoEPRS (CaG)",
            "MultiPRS (UKB)",
            "MultiPRS (CaG)",
            "Best Single Source PRS",
            "Ancestry-matched PRS",
        ],
        height=4.5,
        aspect=1.25,
        palette={
            "MoEPRS (UKB)": "#375E97",
            "MoEPRS (CaG)": "#8CA8D8",
            "MultiPRS (UKB)": "#FFBB00",
            "MultiPRS (CaG)": "#FFE066",
            "Best Single Source PRS": "#BC80BD",
            "Ancestry-matched PRS": "#66C2A5",
        },
        test_models=("MoEPRS (UKB)", "MultiPRS (UKB)"),
    )

    # ---------------- Plot accuracy subpanels ----------------

    ukb_data = extract_accuracy_data("ukbb", evaluation_category="Coarse Ancestry")
    cag_data = extract_accuracy_data("cartagene", evaluation_category="Coarse Ancestry")

    all_data = pd.concat([ukb_data, cag_data], axis=0).reset_index(drop=True)

    g = plot_combined_accuracy_metrics(
        all_data,
        output_f="figures/section_2/accuracy_subpanels_coarse_ancestry.eps",
        col_order=["Standing Height (UKB)", "Standing Height (CaG)"],
        hue_order=[
            "MoEPRS (UKB)",
            "MoEPRS (CaG)",
            "MultiPRS (UKB)",
            "MultiPRS (CaG)",
            "Best Single Source PRS",
            "Ancestry-matched PRS",
        ],
        height=4.5,
        aspect=1.25,
        palette={
            "MoEPRS (UKB)": "#375E97",
            "MoEPRS (CaG)": "#8CA8D8",
            "MultiPRS (UKB)": "#FFBB00",
            "MultiPRS (CaG)": "#FFE066",
            "Best Single Source PRS": "#BC80BD",
            "Ancestry-matched PRS": "#66C2A5",
        },
        test_models=("MoEPRS (UKB)", "MultiPRS (UKB)"),
    )

    # ----------------------------------------------------------------
    # Evaluate on training data (sanity check):

    ukb_data = extract_accuracy_data("ukbb", dataset="train_data")
    cag_data = extract_accuracy_data("cartagene", dataset="train_data")

    all_data = pd.concat([ukb_data, cag_data], axis=0).reset_index(drop=True)

    g = plot_combined_accuracy_metrics(
        all_data,
        output_f="figures/section_2/accuracy_subpanels_train.eps",
        col_order=["Standing Height (UKB)", "Standing Height (CaG)"],
        hue_order=[
            "MoEPRS (UKB)",
            "MoEPRS (CaG)",
            "MultiPRS (UKB)",
            "MultiPRS (CaG)",
            "Best Single Source PRS",
            "Ancestry-matched PRS",
        ],
        height=4.5,
        aspect=1.25,
        palette={
            "MoEPRS (UKB)": "#375E97",
            "MoEPRS (CaG)": "#8CA8D8",
            "MultiPRS (UKB)": "#FFBB00",
            "MultiPRS (CaG)": "#FFE066",
            "Best Single Source PRS": "#BC80BD",
            "Ancestry-matched PRS": "#66C2A5",
        },
        test_models=("MoEPRS (UKB)", "MultiPRS (UKB)"),
    )

    # ---------------- Plot accuracy subpanels ----------------
    # Plot Pearson correlation metrics
    ukb_data = extract_accuracy_data("ukbb", metric="CORR")
    cag_data = extract_accuracy_data("cartagene", metric="CORR")

    all_data = pd.concat([ukb_data, cag_data], axis=0).reset_index(drop=True)

    g = plot_combined_accuracy_metrics(
        all_data,
        output_f="figures/section_2/accuracy_subpanels_corr_test.eps",
        col_order=["Standing Height (UKB)", "Standing Height (CaG)"],
        hue_order=[
            "MoEPRS (UKB)",
            "MoEPRS (CaG)",
            "MultiPRS (UKB)",
            "MultiPRS (CaG)",
            "Best Single Source PRS",
            "Ancestry-matched PRS",
        ],
        metric="CORR",
        height=4.5,
        aspect=1.25,
        palette={
            "MoEPRS (UKB)": "#375E97",
            "MoEPRS (CaG)": "#8CA8D8",
            "MultiPRS (UKB)": "#FFBB00",
            "MultiPRS (CaG)": "#FFE066",
            "Best Single Source PRS": "#BC80BD",
            "Ancestry-matched PRS": "#66C2A5",
        },
    )

    ukb_data = extract_accuracy_data("ukbb", metric="CORR", dataset="train_data")
    cag_data = extract_accuracy_data("cartagene", metric="CORR", dataset="train_data")

    all_data = pd.concat([ukb_data, cag_data], axis=0).reset_index(drop=True)

    g = plot_combined_accuracy_metrics(
        all_data,
        output_f="figures/section_2/accuracy_subpanels_corr_train.eps",
        col_order=["Standing Height (UKB)", "Standing Height (CaG)"],
        hue_order=[
            "MoEPRS (UKB)",
            "MoEPRS (CaG)",
            "MultiPRS (UKB)",
            "MultiPRS (CaG)",
            "Best Single Source PRS",
            "Ancestry-matched PRS",
        ],
        metric="CORR",
        height=4.5,
        aspect=1.25,
        palette={
            "MoEPRS (UKB)": "#375E97",
            "MoEPRS (CaG)": "#8CA8D8",
            "MultiPRS (UKB)": "#FFBB00",
            "MultiPRS (CaG)": "#FFE066",
            "Best Single Source PRS": "#BC80BD",
            "Ancestry-matched PRS": "#66C2A5",
        },
    )

    # ---------------- Plot PRS Mixture graphs ----------------

    for biobank in ("ukbb", "cartagene"):
        data_path = f"data/harmonized_data/HEIGHT/{biobank}/test_data.pkl"
        model_path = (
            f"data/trained_models/HEIGHT/{biobank}/train_data/{args.moe_model}.pkl"
        )

        p_dataset = PRSDataset.from_pickle(data_path)
        moe_model = MoEPRS.from_saved_model(model_path)

        biobank_name = {"ukbb": "UKB", "cartagene": "CaG"}[biobank]

        # Generate the admixture graphs:
        plot_admixture_graphs(
            p_dataset,
            moe_model,
            group_col="Ancestry",
            title=f"PRS Mixture Graph for Standing Height ({biobank_name})",
            output_file=f"figures/section_2/admixture_graphs_{biobank}.png",
            subsample_within_groups=True,
            agg_mechanism="sort",
            figsize=(g.fig.get_size_inches()[0], 3.1),
        )

    # ---------------- Plot mixing weight comparison ----------------

    sns.set_context("paper", font_scale=1.25)
    plot_mixing_weight_comparison("ukbb")
    plot_mixing_weight_comparison("cartagene")

    sns.set_context("paper", font_scale=1.1)

    plot_mixing_weight_comparison_quantitative()
    plot_mixing_weight_comparison_quantitative("cartagene")

    # ---------------- Plot fine-grained admixture graphs ----------------

    # Plot the fine-grained admixture graphs for the MoE model:

    # First case: OTH ancestry group in UKB:
    data_path = "data/harmonized_data/HEIGHT/ukbb/full_data.pkl"
    model_path = f"data/trained_models/HEIGHT/ukbb/train_data/{args.moe_model}.pkl"

    p_dataset = PRSDataset.from_pickle(data_path)

    umap_cluster_map = {
        "14 ENG-EAS-MIX": "Mixed EAS",
        "22 ENG-CAR-WAB": "Mixed Caribbean",
        "24 ENG-BRI-OTH": "White (Other)",
        "25 ENG-AFR-CAR-MIX": "Mixed\nAfro-Caribbean",
        "4 ENG-BRI-AOW": "White/Jewish",
        "5 LEV": "Levant",
    }

    # Filter the samples to only include those with OTH ancestry AND those that
    # belong to the assigned clusters above:
    p_dataset.filter_samples(
        (p_dataset.data["Ancestry"] == "OTH")
        & (p_dataset.data["UMAP_Cluster"].isin(list(umap_cluster_map.keys())))
    )

    p_dataset.data["Fine-scale genetic cluster (UMAP+HDBSCAN)"] = p_dataset.data[
        "UMAP_Cluster"
    ]

    p_dataset.data["Fine-scale genetic cluster (UMAP+HDBSCAN)"] = p_dataset.data[
        "Fine-scale genetic cluster (UMAP+HDBSCAN)"
    ].map(umap_cluster_map)

    moe_model = MoEPRS.from_saved_model(model_path)

    sns.set_context("paper", font_scale=1.2)

    plot_admixture_graphs(
        p_dataset,
        moe_model,
        group_col="Fine-scale genetic cluster (UMAP+HDBSCAN)",
        title="PRS Mixture Graph\nUKB samples with unassigned ancestry (OTH)",
        output_file="figures/section_2/admixture_graphs_ukbb_OTH.png",
        subsample_within_groups=True,
        agg_mechanism="sort",
        sorted_groups=[
            "Levant",
            "White/Jewish",
            "White (Other)",
            "Mixed\nAfro-Caribbean",
            "Mixed Caribbean",
            "Mixed EAS",
        ],
        min_group_size=30,
        figsize=(0.5 * g.fig.get_size_inches()[0], 2.7),
        drop_legend=True,
        tick_rotation=0,
    )

    # ---------------------------------------------------
    # Second case: MID ancestry group in cartagene:
    data_path = "data/harmonized_data/HEIGHT/cartagene/test_data.pkl"
    model_path = f"data/trained_models/HEIGHT/cartagene/train_data/{args.moe_model}.pkl"

    p_dataset = PRSDataset.from_pickle(data_path)
    # Filter the samples to only include those with MID ancestry:
    p_dataset.filter_samples(p_dataset.data["Ancestry"] == "MID")
    p_dataset.data["Fine-scale genetic cluster (UMAP+HDBSCAN)"] = p_dataset.data[
        "UMAP_Cluster"
    ].map(lambda x: {"4-NAF": "North Africa", "5-MIE": "Levant"}.get(x, x))
    moe_model = MoEPRS.from_saved_model(model_path)

    sns.set_context("paper", font_scale=1.25)

    plot_admixture_graphs(
        p_dataset,
        moe_model,
        group_col="Fine-scale genetic cluster (UMAP+HDBSCAN)",
        title="PRS Mixture Graph for Standing Height (CaG; Ancestry=MID)",
        output_file="figures/section_2/admixture_graphs_cartagene_MID.png",
        subsample_within_groups=True,
        agg_mechanism="sort",
        sorted_groups=["North Africa", "Levant"],
        min_group_size=30,
        figsize=(0.75 * g.fig.get_size_inches()[0], 3.1),
        tick_rotation=0,
    )

    # Third case: MID ancestry group in UKB:
    data_path = "data/harmonized_data/HEIGHT/ukbb/full_data.pkl"
    model_path = f"data/trained_models/HEIGHT/ukbb/train_data/{args.moe_model}.pkl"

    p_dataset = PRSDataset.from_pickle(data_path)

    umap_mid_clusters = {
        "18 AFR": "Africa",
        "23 HAFR": "Horn of Africa",
        "5 LEV": "Levant",
        "6 NAF": "North Africa",
    }

    # Filter the samples to only include those with MID ancestry:
    p_dataset.filter_samples(
        (p_dataset.data["Ancestry"] == "MID")
        & (p_dataset.data["UMAP_Cluster"].isin(list(umap_mid_clusters.keys())))
    )
    p_dataset.data["Fine-scale genetic cluster (UMAP+HDBSCAN)"] = p_dataset.data[
        "UMAP_Cluster"
    ].map(umap_mid_clusters)
    moe_model = MoEPRS.from_saved_model(model_path)

    sns.set_context("paper", font_scale=1.25)

    plot_admixture_graphs(
        p_dataset,
        moe_model,
        group_col="Fine-scale genetic cluster (UMAP+HDBSCAN)",
        title="PRS Mixture Graph for Standing Height (UKB; Ancestry=MID)",
        output_file="figures/section_2/admixture_graphs_ukbb_MID.png",
        subsample_within_groups=True,
        sorted_groups=["Africa", "Horn of Africa", "North Africa", "Levant"],
        agg_mechanism="sort",
        min_group_size=30,
        tick_rotation=0,
        figsize=(0.75 * g.fig.get_size_inches()[0], 3.1),
    )
