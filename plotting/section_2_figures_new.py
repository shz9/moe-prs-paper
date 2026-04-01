import argparse
import glob
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
    ANALYSIS_TO_PHENOTYPE_MAP,
    BIOBANK_NAME_MAP_SHORT,
    MODEL_NAME_MAP,
    assign_ancestry_consistent_colors,
    assign_models_consistent_colors,
    read_eval_metrics,
    transform_eval_metrics,
)
from PRSDataset import PRSDataset


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
        "AncestryWeightedPRS": "Ancestry-weighted PRS",
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

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)
    makedir("figures/section_2_new/")

    palette = {
        "MoEPRS (UKB)": "#375E97",
        "MoEPRS (CaG)": "#8CA8D8",
        "MultiPRS (UKB)": "#FFBB00",
        "MultiPRS (CaG)": "#FFE066",
        "Best Single Source PRS": "#BC80BD",
        "Ancestry-weighted PRS": "#66C2A5",
    }

    hue_order = [
        "MoEPRS (UKB)",
        "MoEPRS (CaG)",
        "MultiPRS (UKB)",
        "MultiPRS (CaG)",
        "Best Single Source PRS",
        "Ancestry-weighted PRS",
    ]

    for biobank in ("ukbb", "cartagene"):
        bb_short = BIOBANK_NAME_MAP_SHORT[biobank]
        analysis_results = []

        for d in glob.glob(f"data/harmonized_data/*_MA/{biobank}"):
            analysis_id = d.split("/")[-2]

            if "_ADJ_" in analysis_id:
                continue

            if any([p in analysis_id for p in ("ASTHMA", "T2D")]):
                metric = "Liability_R2"
            else:
                metric = "Incremental_R2"

            analysis_results.append(
                extract_accuracy_data(
                    args.moe_model, analysis_id, biobank, metric=metric
                )
            )

            if metric == "Liability_R2":
                analysis_results[-1]["Incremental_R2"] = analysis_results[-1][
                    "Liability_R2"
                ]
                analysis_results[-1]["Incremental_R2_err"] = analysis_results[-1][
                    "Liability_R2_err"
                ]

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

        df["Evaluation Group"] += f" samples in {bb_short}"

        plot_combined_accuracy_metrics(
            df,
            output_f=f"figures/section_2_new/{biobank}_metrics.eps",
            x="Phenotype",
            palette=palette,
            order=[
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
            ],
            hue_order=hue_order,
            column=None,
            row="Evaluation Group",
            height=3,
            aspect=3,
            sharey=True,
            test_models=[
                (f"MoEPRS ({bb_short})", f"MultiPRS ({bb_short})"),
                (f"MoEPRS ({bb_short})", "Best Single Source PRS"),
            ],
            significance_symbols=["*", "+"],
            x_tick_rotation=90,
        )
