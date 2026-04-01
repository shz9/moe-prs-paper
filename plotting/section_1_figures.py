import argparse
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from combined_accuracy_plots import plot_combined_accuracy_metrics
from evaluate_predictive_performance import stratified_evaluation
from moe import MoEPRS
from plot_predictive_performance import postprocess_metrics_df
from plot_utils import (
    ANALYSIS_TO_PHENOTYPE_MAP,
    BIOBANK_NAME_MAP_SHORT,
    GROUP_MAP,
    assign_ancestry_consistent_colors,
    read_eval_metrics,
    sort_groups,
    transform_eval_metrics,
)
from PRSDataset import PRSDataset


def assign_ancestry_consistent_markers(groups, markers=None):
    """
    Assign consistent markers to the ancestry groups for plotting.
    :param groups: A list of ancestry group names
    :param markers: A dictionary of group names and markers
    :return: A dictionary of group names and markers
    """
    if markers is None:
        markers = {
            "AFR": "o",
            "AMR": "s",
            "EAS": "^",
            "EUR": "D",
            "CSA": "v",
            "MID": "*",
            "OTH": "X",
        }

    return {k: markers[k] for k in groups if k in markers}


def plot_gate_mixing_weights_colored_by_ancestry(weights_df, output_f, order=None):
    g = sns.relplot(
        data=weights_df,
        x="Age",
        y="P(Male_PGS)",
        col="Phenotype",  # Creates one subplot per phenotype
        col_order=order,
        hue="Ancestry",  # Color by Sex
        hue_order=sort_groups(weights_df["Ancestry"].unique()),
        style="Sex",  # Marker style by Ancestry
        kind="scatter",
        alpha=0.3,
        height=5,
        aspect=1,
        palette=assign_ancestry_consistent_colors(weights_df["Ancestry"].unique()),
        markers={
            "Male": "o",
            "Female": "X",
        },
        facet_kws={"sharex": True, "sharey": True},
    )

    # Set the alpha of legend handles to 1 (full opacity)
    for lh in g.legend.legend_handles:
        lh.set_alpha(1)

    # Remove the "Phenotype = " prefix from the title:
    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("Phenotype = "):
            ax.set_title(title.replace("Phenotype = ", ""))

    g.set_axis_labels(
        x_var="Age at recruitment", y_var="Mixing weight for male PRS\nP(Male_PRS)"
    )

    plt.savefig(output_f, bbox_inches="tight", dpi=400)
    plt.close()


def plot_gate_mixing_weights_colored_by_sex(weights_df, output_f, order=None):
    g = sns.relplot(
        data=weights_df,
        x="Age",
        y="P(Male_PGS)",
        col="Phenotype",  # Creates one subplot per phenotype
        col_order=order,
        hue="Sex",  # Color by Sex
        hue_order=["Female", "Male"],
        style="Ancestry",  # Marker style by Ancestry
        kind="scatter",
        alpha=0.3,
        height=5,
        aspect=1,
        palette={
            "Male": "#A1BE95",
            "Female": "#F98866",
        },
        markers=assign_ancestry_consistent_markers(weights_df["Ancestry"].unique()),
        style_order=sort_groups(weights_df["Ancestry"].unique()),
        facet_kws={"sharex": True, "sharey": True},
    )

    # Set the alpha of legend handles to 1 (full opacity)
    for lh in g.legend.legend_handles:
        lh.set_alpha(1)

    # Remove the "Phenotype = " prefix from the title:
    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("Phenotype = "):
            ax.set_title(title.replace("Phenotype = ", ""))

    g.set_axis_labels(
        x_var="Age at recruitment", y_var="Mixing weight for male PRS\nP(Male_PRS)"
    )

    plt.savefig(output_f, bbox_inches="tight", dpi=400)
    plt.close()


def extract_weights_data(biobank="ukbb"):
    dfs = []

    for pheno in phenotypes:
        # Extract expert weights from model for same dataset:
        try:
            dataset = PRSDataset.from_pickle(
                f"data/harmonized_data/{pheno}/{biobank}/test_data.pkl"
            )
            moe_model = MoEPRS.from_saved_model(
                f"data/trained_models/{pheno}/{biobank}/train_data/{args.moe_model}.pkl"
            )
        except Exception as e:
            print(e)
            continue

        w_df = pd.DataFrame(
            np.array(["Female", "Male"])[dataset.get_data_columns("Sex").astype(int)],
            columns=["Sex"],
        )
        w_df[["Age", "Ancestry"]] = dataset.get_data_columns(["Age", "Ancestry"])

        prs_col_names = []
        for prs_col in dataset.prs_cols:
            if prs_col.endswith("_F"):
                prs_col_names.append("P(Female_PGS)")
            else:
                prs_col_names.append("P(Male_PGS)")

        w_df[prs_col_names] = moe_model.predict_proba(dataset)
        w_df["Phenotype"] = (
            phenotypes[pheno] + {"ukbb": " (UKB)", "cartagene": " (CaG)"}[biobank]
        )

        dfs.append(w_df)

    return pd.concat(dfs, axis=0).reset_index(drop=True)


def extract_stratified_evaluation_metrics(
    pheno, test_biobank, keep_ancestry=None, exclude_ancestry=None, category="Sex+Age"
):
    if isinstance(keep_ancestry, str):
        keep_ancestry = [keep_ancestry]

    if isinstance(exclude_ancestry, str):
        exclude_ancestry = [exclude_ancestry]

    dat = PRSDataset.from_pickle(
        f"data/harmonized_data/{pheno}/{test_biobank}/full_data.pkl"
    )

    # Apply filters:
    if keep_ancestry is not None:
        dat.filter_samples(dat.data["Ancestry"].isin(keep_ancestry))
    elif exclude_ancestry is not None:
        dat.filter_samples(~dat.data["Ancestry"].isin(exclude_ancestry))

    dat.data["Ancestry+Sex"] = (
        dat.data["Ancestry"]
        + "-"
        + dat.data["Sex"].astype(int).astype(str).map(GROUP_MAP)
    )
    dat.data["Sex+Age"] = (
        dat.data["Sex"].astype(int).astype(str).map(GROUP_MAP)
        + "\n("
        + np.array(["Age<=55", "Age>55"]).take(
            dat.get_data_columns("Age").flatten() > 55
        )
        + ")"
    )

    eval_df = stratified_evaluation(
        dat, trained_models=None, cat_group_cols=category, min_group_size=20
    )

    # Remove the "All" category:
    eval_df = eval_df.loc[eval_df["EvalGroup"] != "All"]

    uniq_pgs = eval_df["PGS"].unique()
    male_pgs = [m for m in uniq_pgs if m.endswith("_M")][0]
    female_pgs = [m for m in uniq_pgs if m.endswith("_F")][0]

    eval_df = eval_df.loc[eval_df["PGS"].isin([male_pgs, female_pgs])]

    tr_df = eval_df.pivot(
        index="EvalGroup", columns="PGS", values="Incremental_R2"
    ).reset_index()
    tr_df["Ratio"] = tr_df[male_pgs] / tr_df[female_pgs]

    return tr_df


def plot_relative_stratified_evaluation(
    phenotype: str, cohort_specs: list[dict], output_path: str
):
    """
    Plot stratified evaluation metrics for a given phenotype.

    Parameters
    ----------
    phenotype : str
        Phenotype code passed to extract_stratified_evaluation_metrics.
    cohort_specs : list[dict]
        Each dict should contain:
          - "dataset": e.g. "ukbb" or "cartagene"
          - "label": cohort label for plotting
          - either "keep_ancestry": list[str] or "exclude_ancestry": list[str]
    output_path : str
        Path to save the PDF.
    title_label : str
        Text used in the plot title.
    """
    dfs = []

    for spec in cohort_specs:
        kwargs = {}
        if "keep_ancestry" in spec:
            kwargs["keep_ancestry"] = spec["keep_ancestry"]
        if "exclude_ancestry" in spec:
            kwargs["exclude_ancestry"] = spec["exclude_ancestry"]

        df = extract_stratified_evaluation_metrics(
            phenotype,
            spec["dataset"],
            **kwargs,
        )
        df["Cohort"] = spec["label"]
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    g = sns.catplot(
        data=combined_df,
        kind="bar",
        x="EvalGroup",
        y="Ratio",
        row="Cohort",
        height=2.0,
        aspect=2.5,
        sharey=False,
        hue="EvalGroup",
        palette=["#F98866", "#F98866", "#A1BE95", "#A1BE95"],
    )

    for ax in g.axes.flatten():
        ax.axhline(y=1.0, color="#878787", linestyle=":")

    g.set_axis_labels(x_var="", y_var="")
    g.fig.supylabel(
        "Relative Incremental $R^2$\n(Male PRS/Female PRS Ratio)",
        multialignment="center",
    )
    g.fig.supxlabel("Evaluation Group", multialignment="center")
    g.fig.suptitle(
        f"Stratified Relative Prediction Accuracy ({ANALYSIS_TO_PHENOTYPE_MAP[phenotype]})",
        multialignment="center",
    )

    g.fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close()


def extract_accuracy_data(
    test_biobank="ukbb", train_biobank="ukbb", restrict_to_same_biobank=True
):
    dfs = []

    for pheno in phenotypes:
        # Extract accuracy metrics:
        f = f"data/evaluation/{pheno}/{test_biobank}/test_data.csv"
        try:
            df = transform_eval_metrics(read_eval_metrics(f))
        except Exception as e:
            print(e)
            continue

        df = df.loc[
            (df["Model Category"] != "MoE")
            | df["Model Name"].isin(
                [  # f'MoE-CFG ({args.biobank})',
                    f"{args.moe_model} ({train_biobank})"
                ]
            )
        ]

        df["Model Name"] = df["Model Name"].str.replace(
            f" ({train_biobank})", "", regex=False
        )
        df["Model Name"] = df["Model Name"].str.replace(
            f"{args.moe_model}", "MoEPRS", regex=False
        )

        if restrict_to_same_biobank:
            df = df.loc[df["Training biobank"] == df["Test biobank"]]

        df = postprocess_metrics_df(
            df,
            "Incremental_R2",
            category="Sex",
            aggregate_single_prs=False,
            include_cohort_matched=False,
        )

        dfs.append(df)

    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    dfs["Phenotype"] += {"ukbb": " (UKB)", "cartagene": " (CaG)"}[test_biobank]

    return dfs


def plot_phenotypic_variance(pheno, biobank="ukbb"):
    dataset = PRSDataset.from_pickle(
        f"data/harmonized_data/{pheno}/{biobank}/full_data.pkl"
    )

    dataset.data["SexG"] = dataset.data["Sex"].astype(int).astype(str).map(GROUP_MAP)
    dataset.data["AgeGroup2"] = np.array(["Age<=55", "Age>55"]).take(
        dataset.get_data_columns("Age").flatten() > 55
    )
    dataset.data["Sex+Age"] = (
        dataset.data["SexG"].values + " (" + dataset.data["AgeGroup2"].values + ")"
    )

    pheno_col = dataset.phenotype_col

    summary = (
        dataset.data.groupby(["Ancestry", "Sex+Age"])[pheno_col]
        .agg(["var", "count"])
        .reset_index()
    )

    unique_ancestries = sort_groups(summary["Ancestry"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    g = sns.swarmplot(
        data=summary,
        x="Ancestry",
        y="var",
        hue="Sex+Age",
        dodge=True,
        order=unique_ancestries,
        palette={
            "Male (Age<=55)": "#A1BE95",
            "Male (Age>55)": "#5B7F61",
            "Female (Age<=55)": "#F98866",
            "Female (Age>55)": "#B23C17",
        },
    )

    for i, x in enumerate(unique_ancestries):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.2, zorder=0)

    ax.axhline(
        np.var(dataset.data[pheno_col]),
        ls="--",
        color="silver",
        label="Overall variance",
    )
    ax.axhline(
        np.var(dataset.data.loc[dataset.data["SexG"] == "Female", pheno_col]),
        ls="--",
        color="#F98866",
        label="Female variance",
    )
    ax.axhline(
        np.var(dataset.data.loc[dataset.data["SexG"] == "Male", pheno_col]),
        ls="--",
        color="#A1BE95",
        label="Male variance",
    )

    sns.scatterplot(
        ax=ax,
        data=dataset.data.groupby("Ancestry")[pheno_col].agg(["var"]).reset_index(),
        x="Ancestry",
        y="var",
        marker="D",
        color="#b19cd9",
        s=36,
        label="Per-ancestry variance",
    )

    ax.set_ylabel(f"Variance of {phenotypes[pheno]}")
    ax.set_title(
        f"Variance of {phenotypes[pheno]} across ancestry, age, and sex ({BIOBANK_NAME_MAP_SHORT[biobank]})"
    )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.savefig(
        f"figures/section_1/phenotypic_variance_{pheno}_{biobank}.pdf",
        bbox_inches="tight",
        dpi=400,
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot figures related to section 1 of manuscript"
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
    makedir("figures/section_1/")

    phenotypes = {
        "LOG_TST_SEX": "Log Testosterone",
        "URT_SEX": "Urate",
        "LOG_CRTN_SEX": "Log Creatinine",
        "WHR_SEX": "Waist-hip ratio",
    }

    palette = {
        "Male PRS": "#A1BE95",
        "Female PRS": "#F98866",
        "MoEPRS": "#375E97",
        "MultiPRS": "#FFBB00",
    }

    hue_order = ["MoEPRS", "MultiPRS", "Female PRS", "Male PRS"]
    phenotype_order = ["Waist-hip ratio", "Log Testosterone", "Log Creatinine", "Urate"]

    ukbb_metrics_dfs = extract_accuracy_data()

    ukbb_metrics_dfs["Model Name"] = ukbb_metrics_dfs["Model Name"] + np.where(
        ukbb_metrics_dfs["Model Category"].isin(["MoE", "MultiPRS"]), "", " PRS"
    )

    ukbb_w_dfs = extract_weights_data()

    ukb_col_order = [p + " (UKB)" for p in phenotype_order]

    plot_combined_accuracy_metrics(
        ukbb_metrics_dfs,
        "figures/section_1/ukb_accuracy_subpanels.pdf",
        column="Phenotype",
        col_order=ukb_col_order,
        palette=palette,
        hue_order=hue_order,
        test_models=("MoEPRS", "MultiPRS"),
    )

    plot_gate_mixing_weights_colored_by_sex(
        ukbb_w_dfs,
        "figures/section_1/ukb_weights.png",
        order=ukb_col_order,
    )

    plot_gate_mixing_weights_colored_by_ancestry(
        ukbb_w_dfs,
        "figures/section_1/ukb_weights_ancestry_colored.png",
        order=ukb_col_order,
    )

    cartagene_metrics_dfs = extract_accuracy_data(
        test_biobank="cartagene", train_biobank="cartagene"
    )

    cartagene_metrics_dfs["Model Name"] = cartagene_metrics_dfs[
        "Model Name"
    ] + np.where(
        cartagene_metrics_dfs["Model Category"].isin(["MoE", "MultiPRS"]), "", " PRS"
    )

    cartagene_w_dfs = extract_weights_data(biobank="cartagene")

    # Exclude testosterone:
    cag_col_order = [p + " (CaG)" for p in phenotype_order if "Testosterone" not in p]

    plot_combined_accuracy_metrics(
        cartagene_metrics_dfs,
        "figures/section_1/cartagene_accuracy_subpanels.pdf",
        column="Phenotype",
        col_order=cag_col_order,
        palette=palette,
        hue_order=hue_order,
        test_models=("MoEPRS", "MultiPRS"),
    )

    plot_gate_mixing_weights_colored_by_sex(
        cartagene_w_dfs,
        "figures/section_1/cartagene_weights.png",
        order=cag_col_order,
    )

    plot_gate_mixing_weights_colored_by_ancestry(
        cartagene_w_dfs,
        "figures/section_1/cartagene_weights_ancestry_colored.png",
        order=cag_col_order,
    )

    sns.set_context("paper", font_scale=1.25)

    plot_relative_stratified_evaluation(
        phenotype="LOG_CRTN_SEX",
        output_path="figures/section_1/stratified_creatinine_accuracy.pdf",
        cohort_specs=[
            {
                "dataset": "ukbb",
                "label": "MID Samples in UKB",
                "keep_ancestry": ["MID"],
            },
            {
                "dataset": "ukbb",
                "label": "AFR Samples in UKB",
                "keep_ancestry": ["AFR"],
            },
            {
                "dataset": "ukbb",
                "label": "CSA Samples in UKB",
                "keep_ancestry": ["CSA"],
            },
            {
                "dataset": "cartagene",
                "label": "Non-European Samples in CaG",
                "exclude_ancestry": ["EUR"],
            },
            {
                "dataset": "cartagene",
                "label": "European Samples in CaG",
                "keep_ancestry": ["EUR"],
            },
        ],
    )

    plot_relative_stratified_evaluation(
        phenotype="URT_SEX",
        output_path="figures/section_1/stratified_urate_accuracy.pdf",
        cohort_specs=[
            {
                "dataset": "ukbb",
                "label": "EAS Samples in UKB",
                "keep_ancestry": ["EAS"],
            },
            {
                "dataset": "ukbb",
                "label": "AFR Samples in UKB",
                "keep_ancestry": ["AFR"],
            },
            {
                "dataset": "ukbb",
                "label": "CSA Samples in UKB",
                "keep_ancestry": ["CSA"],
            },
            {
                "dataset": "cartagene",
                "label": "Non-European Samples in CaG",
                "exclude_ancestry": ["EUR"],
            },
            {
                "dataset": "cartagene",
                "label": "European Samples in CaG",
                "keep_ancestry": ["EUR"],
            },
        ],
    )

    plot_relative_stratified_evaluation(
        phenotype="LOG_TST_SEX",
        output_path="figures/section_1/stratified_testosterone_accuracy.pdf",
        cohort_specs=[
            {
                "dataset": "ukbb",
                "label": "EAS Samples in UKB",
                "keep_ancestry": ["EAS"],
            },
            {
                "dataset": "ukbb",
                "label": "AFR Samples in UKB",
                "keep_ancestry": ["AFR"],
            },
            {
                "dataset": "ukbb",
                "label": "CSA Samples in UKB",
                "keep_ancestry": ["CSA"],
            },
        ],
    )

    plot_phenotypic_variance("LOG_CRTN_SEX", biobank="ukbb")
    plot_phenotypic_variance("URT_SEX", biobank="ukbb")
    plot_phenotypic_variance("WHR_SEX", biobank="ukbb")
    plot_phenotypic_variance("LOG_CRTN_SEX", biobank="cartagene")
    plot_phenotypic_variance("URT_SEX", biobank="cartagene")
    plot_phenotypic_variance("WHR_SEX", biobank="cartagene")
