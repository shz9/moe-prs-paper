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
from eval_utils import DEFAULT_MIN_GROUP_SIZE, rowwise_cosine_similarity
from moe import MoEPRS
from plot_pgs_admixture import plot_admixture_graphs
from plot_utils import (
    ANALYSIS_TO_PHENOTYPE_MAP,
    ANALYSIS_TO_TABLE_MAP,
    BIOBANK_NAME_MAP_SHORT,
    extract_accuracy_data_all_phenotypes,
)
from PRSDataset import PRSDataset


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
}


def shorten_phenotype_label(label):
    return PHENOTYPE_LABEL_MAP.get(label, label.replace(" (Adj.)", ""))


def _fold_sort_key(path):
    fold_name = next(
        (part for part in osp.normpath(path).split(osp.sep) if part.startswith("fold_")),
        "fold_0",
    )
    try:
        return int(fold_name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return fold_name


def _fold_model_paths(analysis_id, train_biobank, moe_model_name):
    paths = sorted(
        glob.glob(
            f"data/trained_models/{analysis_id}/{train_biobank}/"
            f"fold_*/train_data/{moe_model_name}.pkl"
        ),
        key=_fold_sort_key,
    )
    return {
        osp.basename(osp.dirname(osp.dirname(path))): path for path in paths
    }


def load_fold_moe(
    moe_model_name,
    analysis_id,
    train_biobank,
    fold="fold_1",
):
    """Load one fold-trained MoE model for descriptive admixture plots."""
    model_path = (
        f"data/trained_models/{analysis_id}/{train_biobank}/"
        f"{fold}/train_data/{moe_model_name}.pkl"
    )
    if not osp.exists(model_path):
        raise FileNotFoundError(
            f"Fold model not found for {analysis_id}/{train_biobank}/{fold}: "
            f"{model_path}"
        )
    return MoEPRS.from_saved_model(model_path)


def extract_mixing_weight_similarity(
    moe_model_name,
    analysis_id,
    biobank,
    metric="cosine",  # "cosine" or "jsd"
):
    test_dat = PRSDataset.from_pickle(
        f"data/harmonized_data/{analysis_id}/{biobank}/full_data.pkl"
    )

    ukb_model_paths = _fold_model_paths(analysis_id, "ukbb", moe_model_name)
    cag_model_paths = _fold_model_paths(analysis_id, "cartagene", moe_model_name)
    common_folds = sorted(
        set(ukb_model_paths).intersection(cag_model_paths),
        key=_fold_sort_key,
    )
    if not common_folds:
        raise FileNotFoundError(
            f"No matched UKBB/CARTaGENE fold models found for {analysis_id}."
        )

    # Generate masks for different subsets of the data:

    masks = {
        "All": np.arange(test_dat.N),
        "EUR": test_dat.data["Ancestry"].values == "EUR",
        "non-EUR": test_dat.data["Ancestry"].values != "EUR",
    }

    sim_result = []

    for fold in common_folds:
        ukb_model = MoEPRS.from_saved_model(ukb_model_paths[fold])
        cag_model = MoEPRS.from_saved_model(cag_model_paths[fold])
        prob_ukb = np.asarray(ukb_model.predict_proba(test_dat), dtype=float)
        prob_cag = np.asarray(cag_model.predict_proba(test_dat), dtype=float)

        if prob_ukb.shape != prob_cag.shape:
            raise ValueError(
                f"Shape mismatch in {fold}: prob_ukb has shape {prob_ukb.shape}, "
                f"prob_cag has shape {prob_cag.shape}"
            )

        for cohort, cohort_mask in masks.items():
            if metric == "cosine":
                similarity = rowwise_cosine_similarity(
                    prob_ukb[cohort_mask, :], prob_cag[cohort_mask, :]
                )
            elif metric == "jsd":
                # jensenshannon returns distance, so convert to similarity.
                similarity = 1.0 - jensenshannon(
                    prob_ukb[cohort_mask, :],
                    prob_cag[cohort_mask, :],
                    axis=1,
                    base=2,
                )
            else:
                raise ValueError("metric must be either 'cosine' or 'jsd'")

            sim_result.append(
                {
                    "Fold": fold,
                    "Cohort": cohort,
                    "Similarity": np.mean(similarity),
                    "Phenotype": ANALYSIS_TO_PHENOTYPE_MAP[analysis_id],
                }
            )

    return pd.DataFrame(sim_result)


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
        choices={
            "Liability_R2",
            "Nagelkerke_R2",
            "CoxSnell_R2",
            "McFadden_R2",
            "Liability_Probit_R2",
            "Liability_Logit_R2",
        },
        default="Liability_R2",
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

    parser.add_argument(
        "--admixture-fold",
        dest="admixture_fold",
        type=str,
        default="fold_1",
        help=(
            "Fold-trained model used for descriptive admixture plots "
            "(default: fold_1)."
        ),
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

    metric_name = {"cosine": "cosine similarity", "jsd": "Jensen-Shannon similarity"}

    hue_order = [
        "MoEPRS (UKB)",
        "MoEPRS (CaG)",
        "MultiPRS (UKB)",
        "MultiPRS (CaG)",
        "Ancestry-weighted PRS",
        "Best Single Source PRS",
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

    print(">>> Section 2 figures <<<")

    print("> Plotting accuracy metrics...")

    for biobank in ("ukbb", "cartagene"):
        bb_short = BIOBANK_NAME_MAP_SHORT[biobank]
        metrics_df = extract_accuracy_data_all_phenotypes(
            args.moe_model,
            biobank,
            metric_kind="incremental_vs_ref",
            ref_model_biobank="test_biobank",
            binary_metric=args.binary_metric,
            exclude_analyses=["LDL_ADJ_MA", "TC_ADJ_MA", "DBP_ADJ_MA", "SBP_ADJ_MA"],
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
        plot_phenotype_order = [shorten_phenotype_label(p) for p in phenotype_order]

        g = plot_combined_accuracy_metrics(
            metrics_df,
            output_f=f"figures/section_2/accuracy_metrics_all_{biobank}.pdf",
            x="Phenotype",
            palette=palette,
            order=plot_phenotype_order,
            hue_order=hue_order,
            column=None,
            row="Evaluation Group",
            height=3,
            aspect=4,
            sharey=True,
            test_models=[
                (f"MoEPRS ({bb_short})", f"MultiPRS ({bb_short})"),
                (f"MoEPRS ({bb_short})", "Best Single Source PRS"),
                (f"MoEPRS ({bb_short})", "Ancestry-weighted PRS"),
            ],
            significance_symbols=["*", "+", "°"],
            x_tick_rotation=30,
            legend_title="Model Name\n(Training biobank)",
        )
        if g.legend is not None:
            g.legend.get_title().set_fontsize("small")

    # = = = = = = = = = = = = = = = = = = = = = = = =

    print("> Plotting mixing weight similarity...")

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
        weight_similarity["Phenotype"] = weight_similarity["Phenotype"].map(
            shorten_phenotype_label
        )
        plot_phenotype_order = [shorten_phenotype_label(p) for p in phenotype_order]

        fig, ax = plt.subplots(figsize=(6.5, 2.9))

        sns.pointplot(
            data=weight_similarity,
            x="Phenotype",
            y="Similarity",
            hue="Cohort",
            order=plot_phenotype_order,
            hue_order=["All", "EUR", "non-EUR"],
            palette={
                "All": "#9FBAD6",  # light, desaturated blue
                "EUR": "#5F7FA6",  # muted mid-blue
                "non-EUR": "#CD9395",  # keep your terracotta
            },
            dodge=0.35,
            markers="o",
            linestyles="",
            errorbar="se",
            capsize=0,
            ax=ax,
        )

        ax.set_ylabel(f"Mean {metric_name[args.sim_metric]}")
        ax.set_ylim(0.0, 1.03)
        ax.set_yticks(np.linspace(0.0, 1.0, 5))
        ax.tick_params(axis="x", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")

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
            f"figures/section_2/mixing_weight_similarity_{args.sim_metric}_all_{biobank}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    # ---------------- Plot PRS Mixture graphs ----------------

    print("> Plotting PRS mixture graphs...")

    for analysis_id in ANALYSIS_TO_PHENOTYPE_MAP.keys():
        if ANALYSIS_TO_PHENOTYPE_MAP.get(
            analysis_id, analysis_id
        ) not in phenotype_order or ANALYSIS_TO_TABLE_MAP.get(analysis_id) in (
            "multitrait_prs_table",
            "control_multitrait_prs_table",
        ):
            continue

        for biobank in ("ukbb", "cartagene"):
            data_path = f"data/harmonized_data/{analysis_id}/{biobank}/full_data.pkl"

            p_dataset = PRSDataset.from_pickle(data_path)
            moe_model = load_fold_moe(
                args.moe_model,
                analysis_id,
                biobank,
                fold=args.admixture_fold,
            )

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

    print("> Plotting fine-grained admixture graphs...")

    # Plot the fine-grained admixture graphs for the MoE model:

    # First case: OTH ancestry group in UKB:
    data_path = "data/harmonized_data/HEIGHT_MA/ukbb/full_data.pkl"

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

    moe_model = load_fold_moe(
        args.moe_model,
        "HEIGHT_MA",
        "ukbb",
        fold=args.admixture_fold,
    )

    sns.set_context("paper", font_scale=1.2)

    plot_admixture_graphs(
        p_dataset,
        moe_model,
        group_col="Fine-scale genetic cluster (UMAP+HDBSCAN)",
        title="PRS Mixture Graph\nUKB samples with unassigned ancestry (OTH)",
        output_file="figures/section_2/admixture_graphs_OTH_ukbb.png",
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
        min_group_size=DEFAULT_MIN_GROUP_SIZE,
        figsize=(0.5 * g.fig.get_size_inches()[0], 2.7),
        drop_legend=True,
        tick_rotation=0,
    )

    # ---------------------------------------------------
    # Second case: MID ancestry group in cartagene:

    print("> Plotting MID ancestry group in cartagene...")

    data_path = "data/harmonized_data/HEIGHT_MA/cartagene/full_data.pkl"

    p_dataset = PRSDataset.from_pickle(data_path)
    # Filter the samples to only include those with MID ancestry:
    p_dataset.filter_samples(p_dataset.data["Ancestry"] == "MID")
    p_dataset.data["Fine-scale genetic cluster (UMAP+HDBSCAN)"] = p_dataset.data[
        "UMAP_Cluster"
    ].map(lambda x: {"4-NAF": "North Africa", "5-MIE": "Levant"}.get(x, x))
    moe_model = load_fold_moe(
        args.moe_model,
        "HEIGHT_MA",
        "cartagene",
        fold=args.admixture_fold,
    )

    sns.set_context("paper", font_scale=1.25)

    plot_admixture_graphs(
        p_dataset,
        moe_model,
        group_col="Fine-scale genetic cluster (UMAP+HDBSCAN)",
        title="PRS Mixture Graph for Standing Height (CaG; Ancestry=MID)",
        output_file="figures/section_2/admixture_graphs_MID_cartagene.png",
        subsample=True,
        agg_mechanism="sort",
        sorted_groups=["North Africa", "Levant"],
        min_group_size=DEFAULT_MIN_GROUP_SIZE,
        figsize=(0.75 * g.fig.get_size_inches()[0], 3.1),
        tick_rotation=0,
    )

    # Third case: MID ancestry group in UKB:
    data_path = "data/harmonized_data/HEIGHT_MA/ukbb/full_data.pkl"

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
    moe_model = load_fold_moe(
        args.moe_model,
        "HEIGHT_MA",
        "ukbb",
        fold=args.admixture_fold,
    )

    sns.set_context("paper", font_scale=1.25)

    plot_admixture_graphs(
        p_dataset,
        moe_model,
        group_col="Fine-scale genetic cluster (UMAP+HDBSCAN)",
        title="PRS Mixture Graph for Standing Height (UKB; Ancestry=MID)",
        output_file="figures/section_2/admixture_graphs_MID_ukbb.png",
        subsample=True,
        sorted_groups=["Africa", "Horn of Africa", "North Africa", "Levant"],
        agg_mechanism="sort",
        min_group_size=DEFAULT_MIN_GROUP_SIZE,
        tick_rotation=0,
        figsize=(0.75 * g.fig.get_size_inches()[0], 3.1),
    )
