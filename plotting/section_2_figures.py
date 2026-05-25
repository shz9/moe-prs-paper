import argparse
import glob
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from magenpy.utils.system_utils import makedir
from scipy.spatial.distance import jensenshannon

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from combined_accuracy_plots import plot_combined_accuracy_metrics
from eval_utils import rowwise_cosine_similarity
from moe import MoEPRS
from plot_pgs_admixture import plot_admixture_graphs
from plot_predictive_performance import postprocess_metrics_df
from plot_utils import (
    ANALYSIS_TO_PHENOTYPE_MAP,
    ANALYSIS_TO_TABLE_MAP,
    BIOBANK_NAME_MAP_SHORT,
    read_transform_eval_metrics,
)
from PRSDataset import PRSDataset


def extract_accuracy_data_all_phenotypes(
    moe_model_name,
    biobank,
    dataset="test_data",
    analysis_table_id="multi_ancestry_prs_table",
    binary_metric="Nagelkerke_R2",
    keep_analyses=None,
    exclude_analyses=None,
    exclude_all=True,
):
    analysis_results = []

    for d in glob.glob(f"data/harmonized_data/*/{biobank}"):
        analysis_id = d.split("/")[-2]
        if ANALYSIS_TO_TABLE_MAP.get(analysis_id) != analysis_table_id:
            continue

        if keep_analyses is not None:
            if analysis_id not in keep_analyses:
                continue

        if exclude_analyses is not None:
            if analysis_id in exclude_analyses:
                continue

        analysis_results.append(
            extract_accuracy_data(
                moe_model_name,
                analysis_id,
                biobank,
                dataset=dataset,
                exclude_all=exclude_all,
            )
        )

    df = pd.concat(analysis_results).reset_index(drop=True)

    # Simplify the ancestry weighted model:
    if biobank == "ukbb":
        df = df.loc[df["Model Name"] != "Ancestry-weighted PRS (CaG)"]
        df["Model Name"] = df["Model Name"].replace(
            "Ancestry-weighted PRS (UKB)", "Ancestry-weighted PRS"
        )
    else:
        df = df.loc[df["Model Name"] != "Ancestry-weighted PRS (UKB)"]
        df["Model Name"] = df["Model Name"].replace(
            "Ancestry-weighted PRS (CaG)", "Ancestry-weighted PRS"
        )

    return df


def extract_mixing_weight_similarity(
    moe_model_name,
    analysis_id,
    biobank,
    metric="cosine",  # "cosine" or "jsd"
):
    test_dat = PRSDataset.from_pickle(
        f"data/harmonized_data/{analysis_id}/{biobank}/test_data.pkl"
    )

    ukb_model = MoEPRS.from_saved_model(
        f"data/trained_models/{analysis_id}/ukbb/train_data/{moe_model_name}.pkl"
    )
    cag_model = MoEPRS.from_saved_model(
        f"data/trained_models/{analysis_id}/cartagene/train_data/{moe_model_name}.pkl"
    )

    prob_ukb = np.asarray(ukb_model.predict_proba(test_dat), dtype=float)
    prob_cag = np.asarray(cag_model.predict_proba(test_dat), dtype=float)

    if prob_ukb.shape != prob_cag.shape:
        raise ValueError(
            f"Shape mismatch: prob_ukb has shape {prob_ukb.shape}, "
            f"prob_cag has shape {prob_cag.shape}"
        )

    # Generate masks for different subsets of the data:

    masks = {
        "All": np.arange(prob_ukb.shape[0]),
        "EUR": test_dat.data["Ancestry"].values == "EUR",
        "non-EUR": test_dat.data["Ancestry"].values != "EUR",
    }

    sim_result = []

    for msk, msk_val in masks.items():
        if metric == "cosine":
            similarity = rowwise_cosine_similarity(
                prob_ukb[msk_val, :], prob_cag[msk_val, :]
            )
        elif metric == "jsd":
            # jensenshannon returns Jensen-Shannon distance, so convert to similarity
            similarity = 1.0 - jensenshannon(
                prob_ukb[msk_val, :], prob_cag[msk_val, :], axis=1, base=2
            )
        else:
            raise ValueError("metric must be either 'cosine' or 'jsd'")

        sim_result.append(
            {
                "Cohort": msk,
                "Similarity": np.mean(similarity),
                "Phenotype": ANALYSIS_TO_PHENOTYPE_MAP[analysis_id],
            }
        )

    return pd.DataFrame(sim_result)


def extract_accuracy_data(
    moe_model_name,
    analysis_id,
    test_biobank,
    metric="Incremental_R2",
    dataset="test_data",
    evaluation_category="Coarse Ancestry",
    exclude_all=True,
):
    # Extract accuracy metrics:
    f = f"data/evaluation/{analysis_id}/{test_biobank}/{dataset}.csv"
    df = read_transform_eval_metrics(f)

    df = df.loc[
        (df["model_category"] != "MoE")
        | df["model_name"].isin(
            [
                f"{moe_model_name}",
            ]
        )
    ]

    df = df.loc[
        df["model_category"].isin(
            ["MoE", "MultiPRS", "AncestryWeightedPRS"]
        )  # <- Ensemble model categories
        | (df["train_biobank"] == test_biobank.upper())
    ]

    # Rename the ensemble models for clarity:
    df["model_name"] = df["model_name"].replace(
        {
            moe_model_name: "MoEPRS",
            "MultiPRS": "MultiPRS",
            "AncestryWeightedPRS": "Ancestry-weighted PRS",
        }
    )

    # Correction for binary phenotypes:
    if metric == "Incremental_R2" and "Nagelkerke_R2" in set(df["metric"].unique()):
        metric = "Nagelkerke_R2"

    dfs = postprocess_metrics_df(
        df,
        metric,
        category=evaluation_category,
        min_sample_size=50,
        aggregate_single_prs=True,
    )

    if exclude_all:
        dfs = dfs.loc[dfs["Evaluation Group"] != "All"]

    dfs["Phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]
    # dfs["Phenotype"] += f" ({BIOBANK_NAME_MAP_SHORT[test_biobank]})"

    return dfs


# -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot figures of section 2 of manuscript"
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

    parser.add_argument(
        "--similarity-metric",
        dest="sim_metric",
        type=str,
        default="cosine",
        choices={"cosine", "jsd"},
        help="The similarity metric for the mixing weights.",
    )

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)
    makedir("figures/section_2/")

    palette = {
        "MoEPRS (UKB)": "#375E97",
        "MoEPRS (CaG)": "#8CA8D8",
        "MultiPRS (UKB)": "#FFBB00",
        "MultiPRS (CaG)": "#FFE066",
        "Best Single Source PRS": "#BC80BD",
        "Ancestry-weighted PRS": "#66C2A5",
    }

    metric_name = {"cosine": "Cosine similarity", "jsd": "Jensen-Shannon similarity"}

    hue_order = [
        "MoEPRS (UKB)",
        "MoEPRS (CaG)",
        "MultiPRS (UKB)",
        "MultiPRS (CaG)",
        "Best Single Source PRS",
        "Ancestry-weighted PRS",
    ]

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

    for biobank in ("ukbb", "cartagene"):
        bb_short = BIOBANK_NAME_MAP_SHORT[biobank]
        metrics_df = extract_accuracy_data_all_phenotypes(
            args.moe_model,
            biobank,
            binary_metric=args.binary_metric,
            exclude_analyses=["LDL_ADJ_MA", "TC_ADJ_MA", "DBP_ADJ_MA", "SBP_ADJ_MA"],
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
            output_f=f"figures/section_2/{biobank}_metrics.eps",
            x="Phenotype",
            palette=palette,
            order=phenotype_order,
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

    # = = = = = = = = = = = = = = = = = = = = = = = =

    for biobank in ("ukbb", "cartagene"):
        bb_short = BIOBANK_NAME_MAP_SHORT[biobank]
        weight_similarity = []

        for d in glob.glob(f"data/harmonized_data/*/{biobank}"):
            analysis_id = d.split("/")[-2]
            if ANALYSIS_TO_TABLE_MAP.get(analysis_id) != "multi_ancestry_prs_table":
                continue

            if "_ADJ_" in analysis_id:
                continue

            weight_similarity.append(
                extract_mixing_weight_similarity(
                    args.moe_model, analysis_id, biobank, metric=args.sim_metric
                )
            )

        weight_similarity = pd.concat(weight_similarity)

        fig, ax = plt.subplots(figsize=(6.5, 2.9))

        sns.barplot(
            data=weight_similarity,
            x="Phenotype",
            y="Similarity",
            hue="Cohort",
            order=phenotype_order,
            hue_order=["All", "EUR", "non-EUR"],
            palette={
                "All": "#9FBAD6",  # light, desaturated blue
                "EUR": "#5F7FA6",  # muted mid-blue
                "non-EUR": "#CD9395",  # keep your terracotta
            },
            ax=ax,
        )

        ax.set_ylabel(f"Mean {metric_name[args.sim_metric]}")
        ax.tick_params(axis="x", labelrotation=90)

        ax.legend(
            title="Test cohort",
            bbox_to_anchor=(1.05, 0.5),  # (x, y)
            loc="upper left",
            borderaxespad=0,
        )

        plt.title(
            f"Concordance of mixing weights between\nCaG- and UKB-trained MoEPRS on {bb_short} samples"
        )
        plt.savefig(
            f"figures/section_2/mixing_weight_sim_{biobank}_{args.sim_metric}.eps",
            bbox_inches="tight",
        )
        plt.close()

    # ---------------- Plot PRS Mixture graphs ----------------

    for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP.keys():
        if (
            ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
            not in phenotype_order
            or ANALYSIS_TO_TABLE_MAP.get(analysis_id) == "multitrait_prs_table"
        ):
            continue

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
                output_file=f"figures/section_2/mixture_graphs_{analysis_id}_{biobank}.png",
                subsample=True,
                agg_mechanism="sort",
                figsize=(g.fig.get_size_inches()[0], 3.1),
            )

    # ---------------- Plot fine-grained admixture graphs ----------------

    # Plot the fine-grained admixture graphs for the MoE model:

    # First case: OTH ancestry group in UKB:
    data_path = "data/harmonized_data/HEIGHT_MA/ukbb/full_data.pkl"
    model_path = f"data/trained_models/HEIGHT_MA/ukbb/train_data/{args.moe_model}.pkl"

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
        subsample=True,
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
    data_path = "data/harmonized_data/HEIGHT_MA/cartagene/test_data.pkl"
    model_path = (
        f"data/trained_models/HEIGHT_MA/cartagene/train_data/{args.moe_model}.pkl"
    )

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
        subsample=True,
        agg_mechanism="sort",
        sorted_groups=["North Africa", "Levant"],
        min_group_size=30,
        figsize=(0.75 * g.fig.get_size_inches()[0], 3.1),
        tick_rotation=0,
    )

    # Third case: MID ancestry group in UKB:
    data_path = "data/harmonized_data/HEIGHT_MA/ukbb/full_data.pkl"
    model_path = f"data/trained_models/HEIGHT_MA/ukbb/train_data/{args.moe_model}.pkl"

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
        subsample=True,
        sorted_groups=["Africa", "Horn of Africa", "North Africa", "Levant"],
        agg_mechanism="sort",
        min_group_size=30,
        tick_rotation=0,
        figsize=(0.75 * g.fig.get_size_inches()[0], 3.1),
    )
