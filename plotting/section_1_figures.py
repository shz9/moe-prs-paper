import argparse
import glob
import os.path as osp
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from baseline_models import AttributePartitionedPRS, MultiPRS
from combined_accuracy_plots import plot_combined_accuracy_metrics
from eval_utils import DEFAULT_MIN_GROUP_SIZE
from evaluate_predictive_performance import stratified_evaluation
from moe import MoEPRS
from plot_predictive_performance import postprocess_metrics_df
from plot_utils import (
    ANALYSIS_TO_PHENOTYPE_MAP,
    BIOBANK_NAME_MAP_SHORT,
    MODEL_NAME_MAP,
    SEX_LABEL_MAP,
    aggregate_cross_validation_metrics,
    assign_ancestry_consistent_colors,
    read_transform_eval_metrics,
    sort_groups,
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
        col="phenotype",  # Creates one subplot per phenotype
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

    # Remove the "phenotype = " prefix from the title:
    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("phenotype = "):
            ax.set_title(title.replace("phenotype = ", ""))

    g.set_axis_labels(
        x_var="Age at recruitment", y_var="Mixing weight for Male PRS"
    )

    plt.savefig(output_f, bbox_inches="tight", dpi=400)
    plt.close()


def plot_gate_mixing_weights_colored_by_sex(
    weights_df, output_f, x="Age", x_label="Age at recruitment", order=None
):
    g = sns.relplot(
        data=weights_df,
        x=x,
        y="P(Male_PGS)",
        col="phenotype",  # Creates one subplot per phenotype
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

    # Remove the "phenotype = " prefix from the title:
    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("phenotype = "):
            ax.set_title(title.replace("phenotype = ", ""))

    g.set_axis_labels(
        x_var=x or x_label, y_var="Mixing weight for Male PRS"
    )

    plt.savefig(output_f, bbox_inches="tight", dpi=400)
    plt.close()


def _plot_gate_mixing_weights_boxpoints(
    plot_df,
    output_f,
    x,
    x_label,
    x_order,
    order=None,
    show_points_for_all=True,
    height=5,
    aspect=1,
):
    sex_order = ["Female", "Male"]
    sex_palette = {
        "Female": "#F98866",
        "Male": "#A1BE95",
    }

    g = sns.FacetGrid(
        plot_df,
        col="phenotype",
        col_order=order,
        height=height,
        aspect=aspect,
        sharex=True,
        sharey=True,
    )

    def stripplot_points(data, **kwargs):
        if not show_points_for_all:
            data = data.loc[data[x] != "All"]
        if len(data) == 0:
            return
        sns.stripplot(data=data, **kwargs)

    g.map_dataframe(
        stripplot_points,
        x=x,
        y="P(Male_PGS)",
        hue="Sex",
        hue_order=sex_order,
        order=x_order,
        dodge=True,
        jitter=0.22,
        alpha=0.15,
        size=1.3,
        linewidth=0,
        palette=sex_palette,
    )

    # Draw transparent summary boxes on top of the points.
    g.map_dataframe(
        sns.boxplot,
        x=x,
        y="P(Male_PGS)",
        hue="Sex",
        hue_order=sex_order,
        order=x_order,
        dodge=True,
        showfliers=False,
        width=0.8,
        linewidth=1.3,
        palette=sex_palette,
        boxprops={"zorder": 3},
        whiskerprops={"zorder": 3},
        capprops={"zorder": 3},
        medianprops={"linewidth": 1.3, "zorder": 4},
    )

    g.add_legend(title="Sex")
    for lh in g.legend.legend_handles:
        try:
            lh.set_alpha(1)
        except Exception:
            pass

    # Remove the "phenotype = " prefix from the title:
    for ax in g.axes.flat:
        boxes = []
        for patch in ax.patches:
            edge_color = patch.get_facecolor()
            vertices = patch.get_path().vertices
            x_min = vertices[:, 0].min()
            x_max = vertices[:, 0].max()
            y_min = vertices[:, 1].min()
            y_max = vertices[:, 1].max()
            boxes.append(
                {
                    "color": edge_color,
                    "x_min": x_min,
                    "x_max": x_max,
                    "x_center": 0.5 * (x_min + x_max),
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )
            patch.set_facecolor("none")
            patch.set_edgecolor(edge_color)
            patch.set_linewidth(1.3)

        for line in ax.lines:
            if not boxes:
                break

            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)
            if x_data.size == 0 or y_data.size == 0:
                continue

            line_x_center = 0.5 * (np.nanmin(x_data) + np.nanmax(x_data))
            box = min(boxes, key=lambda b: abs(line_x_center - b["x_center"]))

            line.set_color(box["color"])
            line.set_linewidth(1.3)

        title = ax.get_title()
        if title.startswith("phenotype = "):
            ax.set_title(title.replace("phenotype = ", ""))

        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="x", labelrotation=20)

    g.set_xlabels("")
    g.set_ylabels("Mixing weight for Male PRS")
    g.fig.subplots_adjust(bottom=0.25)
    g.fig.supxlabel(x_label, y=0.04, fontsize="medium")

    plt.savefig(output_f, bbox_inches="tight", dpi=400)
    plt.close()


def plot_gate_mixing_weights_categorical(weights_df, output_f, order=None):
    plot_df = []
    for label, msk in (
        ("All", np.ones(len(weights_df), dtype=bool)),
        ("Europeans", weights_df["Ancestry"].astype(str).values == "EUR"),
        ("Non-Europeans", weights_df["Ancestry"].astype(str).values != "EUR"),
    ):
        df = weights_df.loc[msk].copy()
        if len(df) == 0:
            continue
        df["Coarse Ancestry"] = label
        plot_df.append(df)

    plot_df = pd.concat(plot_df, axis=0, ignore_index=True)
    _plot_gate_mixing_weights_boxpoints(
        plot_df,
        output_f,
        x="Coarse Ancestry",
        x_label="Coarse ancestry group",
        x_order=["All", "Europeans", "Non-Europeans"],
        order=order,
        show_points_for_all=False,
        height=5,
        aspect=1,
    )


def plot_gate_mixing_weights_continental_ancestry(weights_df, output_f, order=None):
    plot_df = weights_df.loc[weights_df["Ancestry"].astype(str) != "All"].copy()
    ancestry_order = [
        a for a in sort_groups(plot_df["Ancestry"].dropna().unique()) if a != "All"
    ]
    _plot_gate_mixing_weights_boxpoints(
        plot_df,
        output_f,
        x="Ancestry",
        x_label="Continental ancestry",
        x_order=ancestry_order,
        order=order,
        show_points_for_all=True,
        height=5,
        aspect=1,
    )


def _fold_sort_key(path):
    fold_name = next(
        (part for part in osp.normpath(path).split(osp.sep) if part.startswith("fold_")),
        osp.basename(osp.dirname(path)),
    )
    try:
        return int(fold_name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return fold_name


def _evaluation_fold_specs(pheno, test_biobank, train_biobank):
    """Return (fold, dataset path, model directory) evaluation tuples."""
    if test_biobank == train_biobank:
        dataset_paths = sorted(
            glob.glob(
                f"data/harmonized_data/{pheno}/{test_biobank}/"
                "fold_*/test_data.pkl"
            ),
            key=_fold_sort_key,
        )
        return [
            (
                osp.basename(osp.dirname(dataset_path)),
                dataset_path,
                (
                    f"data/trained_models/{pheno}/{train_biobank}/"
                    f"{osp.basename(osp.dirname(dataset_path))}/train_data"
                ),
            )
            for dataset_path in dataset_paths
        ]

    # External validation: every training fold is evaluated on the same full
    # held-out cohort. In particular, CARTaGENE is never reduced to fold test data.
    dataset_path = f"data/harmonized_data/{pheno}/{test_biobank}/full_data.pkl"
    model_paths = sorted(
        glob.glob(
            f"data/trained_models/{pheno}/{train_biobank}/"
            f"fold_*/train_data/{args.moe_model}.pkl"
        ),
        key=_fold_sort_key,
    )
    return [
        (
            osp.basename(osp.dirname(osp.dirname(model_path))),
            dataset_path,
            osp.dirname(model_path),
        )
        for model_path in model_paths
    ]


def _weights_dataframe(dataset, probabilities, phenotype_label):
    w_df = pd.DataFrame(
        np.array(["Female", "Male"])[dataset.get_data_columns("Sex").astype(int)],
        columns=["Sex"],
    )
    w_df[["Age", "Ancestry", "PC1", "PC2", "PC4", "PC5"]] = (
        dataset.get_data_columns(["Age", "Ancestry", "PC1", "PC2", "PC4", "PC5"])
    )

    prs_col_names = [
        "P(Female_PGS)" if prs_col.endswith("_F") else "P(Male_PGS)"
        for prs_col in dataset.prs_cols
    ]
    w_df[prs_col_names] = probabilities
    w_df["phenotype"] = phenotype_label
    return w_df


def extract_weights_data(
    biobank="ukbb", train_biobank=None, reference_fold="fold_1"
):
    """Extract full-cohort gate weights from one reference-fold model.

    Mixing-weight figures are descriptive illustrations of a fitted gate, so use
    one coherent model rather than pooling weights from separately fitted fold
    models. Predict that model's weights on the full evaluation biobank.
    """
    if train_biobank is None:
        train_biobank = biobank

    reference_fold = str(reference_fold)
    if not reference_fold.startswith("fold_"):
        reference_fold = f"fold_{reference_fold}"

    dfs = []

    for pheno in phenotypes:
        phenotype_label = (
            phenotypes[pheno] + " (" + BIOBANK_NAME_MAP_SHORT[biobank] + ")"
        )
        dataset_path = f"data/harmonized_data/{pheno}/{biobank}/full_data.pkl"
        model_path = (
            f"data/trained_models/{pheno}/{train_biobank}/{reference_fold}/"
            f"train_data/{args.moe_model}.pkl"
        )

        try:
            dataset = PRSDataset.from_pickle(dataset_path)
            moe_model = MoEPRS.from_saved_model(model_path)
            probabilities = moe_model.predict_proba(dataset)
        except Exception as e:
            print(
                f"> Skipping mixing weights for {pheno} ({biobank}) using "
                f"{train_biobank}/{reference_fold}: {e}"
            )
            continue

        weights_df = _weights_dataframe(dataset, probabilities, phenotype_label)
        weights_df["model_fold"] = reference_fold
        dfs.append(weights_df)

    if not dfs:
        raise FileNotFoundError(
            f"No full-data mixing-weight inputs found for {biobank} using the "
            f"{train_biobank}-trained {reference_fold} model."
        )

    return pd.concat(dfs, axis=0, ignore_index=True)


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
        + dat.data["Sex"].astype(int).astype(str).map(SEX_LABEL_MAP)
    )
    dat.data["Sex+Age"] = (
        dat.data["Sex"].astype(int).astype(str).map(SEX_LABEL_MAP)
        + "\n("
        + np.array(["Age<=55", "Age>55"]).take(
            dat.get_data_columns("Age").flatten() > 55
        )
        + ")"
    )

    eval_df = stratified_evaluation(
        dat,
        trained_models=None,
        cat_group_cols=category,
        metrics=["Incremental_R2"],
        min_group_size=DEFAULT_MIN_GROUP_SIZE,
    )

    eval_df = eval_df.loc[
        (eval_df["eval_group"] != "All")
        & (eval_df["metric"] == "Incremental_R2")
        & (eval_df["metric_kind"] == "base")
    ].copy()
    eval_df.rename(
        columns={
            "model_name": "PGS",
            "eval_group": "EvalGroup",
            "value": "Incremental_R2",
        },
        inplace=True,
    )

    uniq_pgs = eval_df["PGS"].dropna().unique()
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


def extract_sex_prs_ancestry_stratified_accuracy(pheno, test_biobank):
    dataset_path = f"data/harmonized_data/{pheno}/{test_biobank}/full_data.pkl"
    dat = PRSDataset.from_pickle(dataset_path)

    dat.data["SexLabel"] = dat.data["Sex"].astype(int).astype(str).map(SEX_LABEL_MAP)
    dat.data["CoarseAncestry"] = np.where(
        dat.data["Ancestry"] == "EUR", "European", "Non-European"
    )
    dat.data["ContinentalAncestry"] = dat.data["Ancestry"]
    dat.data["Sex+CoarseAncestry"] = (
        dat.data["SexLabel"] + "|" + dat.data["CoarseAncestry"]
    )
    dat.data["Sex+ContinentalAncestry"] = (
        dat.data["SexLabel"] + "|" + dat.data["ContinentalAncestry"]
    )

    eval_df = stratified_evaluation(
        dat,
        trained_models=None,
        cat_group_cols=[
            "SexLabel",
            "Sex+CoarseAncestry",
            "Sex+ContinentalAncestry",
        ],
        metrics=["Incremental_R2"],
        min_group_size=DEFAULT_MIN_GROUP_SIZE,
    )

    eval_df = eval_df.loc[
        (eval_df["metric"] == "Incremental_R2")
        & (eval_df["metric_kind"] == "base")
    ].copy()

    all_df = eval_df.loc[
        (eval_df["eval_category"] == "SexLabel")
        & (eval_df["eval_group"].isin(["Female", "Male"]))
    ].copy()
    all_df["Sex"] = all_df["eval_group"]
    all_df["eval_category"] = "CoarseAncestry"
    all_df["eval_group"] = "All"

    coarse_df = eval_df.loc[
        (eval_df["eval_category"] == "Sex+CoarseAncestry")
    ].copy()
    coarse_df[["Sex", "eval_group"]] = coarse_df["eval_group"].str.split(
        "|", expand=True, regex=False
    )
    coarse_df = coarse_df.loc[
        coarse_df["eval_group"].isin(["European", "Non-European"])
    ].copy()
    coarse_df["eval_category"] = "CoarseAncestry"

    continental_df = eval_df.loc[
        (eval_df["eval_category"] == "Sex+ContinentalAncestry")
    ].copy()
    continental_df[["Sex", "eval_group"]] = continental_df["eval_group"].str.split(
        "|", expand=True, regex=False
    )
    continental_df = continental_df.loc[
        continental_df["eval_group"].isin(["AFR", "CSA", "MID", "EAS"])
    ].copy()
    continental_df["eval_category"] = "ContinentalAncestry"

    plot_df = pd.concat([all_df, coarse_df, continental_df], ignore_index=True)
    plot_df["Model Name"] = np.select(
        [
            plot_df["model_name"].astype(str).str.endswith("_F"),
            plot_df["model_name"].astype(str).str.endswith("_M"),
        ],
        ["Female PRS", "Male PRS"],
        default=None,
    )
    plot_df = plot_df.loc[plot_df["Model Name"].notnull()].copy()

    plot_df["Panel"] = plot_df["eval_category"].replace(
        {
            "CoarseAncestry": "Coarse ancestry",
            "ContinentalAncestry": "Continental ancestry",
        }
    )
    plot_df.rename(
        columns={"eval_group": "Evaluation Group", "value": "Incremental_R2"},
        inplace=True,
    )

    return plot_df[
        ["Sex", "Panel", "Evaluation Group", "Model Name", "Incremental_R2"]
    ]


def plot_sex_prs_ancestry_stratified_accuracy(phenotype, test_biobank, output_path):
    dataset_path = f"data/harmonized_data/{phenotype}/{test_biobank}/full_data.pkl"
    if not osp.exists(dataset_path):
        print(
            f"> Skipping ancestry-stratified accuracy for {phenotype} "
            f"({test_biobank}): dataset not found."
        )
        return

    try:
        plot_df = extract_sex_prs_ancestry_stratified_accuracy(
            phenotype, test_biobank
        )
    except Exception as e:
        print(
            f"> Skipping ancestry-stratified accuracy for {phenotype} "
            f"({test_biobank}): {e}"
        )
        return

    if len(plot_df) == 0:
        print(
            f"> Skipping ancestry-stratified accuracy for {phenotype} "
            f"({test_biobank}): no plottable metrics."
        )
        return

    sex_order = ["Female", "Male"]
    group_order = ["All", "European", "Non-European", "AFR", "CSA", "MID", "EAS"]
    model_order = ["Female PRS", "Male PRS"]
    model_palette = {"Female PRS": "#F98866", "Male PRS": "#A1BE95"}
    model_offsets = {"Female PRS": -0.12, "Male PRS": 0.12}

    fig, axes = plt.subplots(
        len(sex_order),
        1,
        figsize=(7.6, 5.0),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).ravel()

    for row_idx, sex in enumerate(sex_order):
        ax = axes[row_idx]
        sub_df = plot_df.loc[plot_df["Sex"] == sex].copy()
        x_pos = np.arange(len(group_order))

        if len(sub_df) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#777777",
            )
        else:
            for group_idx, group in enumerate(group_order):
                group_df = sub_df.loc[sub_df["Evaluation Group"] == group]
                values = {}
                for model in model_order:
                    model_values = group_df.loc[
                        group_df["Model Name"] == model, "Incremental_R2"
                    ]
                    if len(model_values) == 0:
                        continue
                    values[model] = float(model_values.iloc[0])
                    ax.scatter(
                        group_idx + model_offsets[model],
                        values[model],
                        color=model_palette[model],
                        edgecolor="white",
                        linewidth=0.6,
                        s=36,
                        zorder=3,
                    )

                if len(values) == len(model_order):
                    ax.plot(
                        [
                            group_idx + model_offsets["Female PRS"],
                            group_idx + model_offsets["Male PRS"],
                        ],
                        [values["Female PRS"], values["Male PRS"]],
                        color="#9A9A9A",
                        linewidth=0.8,
                        alpha=0.7,
                        zorder=2,
                    )

        ax.axhline(0.0, color="#B8B8B8", linewidth=0.8, linestyle=":")
        ax.axvline(2.5, color="#D0D0D0", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_order, rotation=20)
        ax.tick_params(axis="x", labelbottom=(row_idx == len(sex_order) - 1))
        ax.set_title(sex)
        ax.set_ylabel("")
        ax.set_xlabel("")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=model,
            markerfacecolor=model_palette[model],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=6,
        )
        for model in model_order
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.supylabel("Incremental $R^2$")
    fig.supxlabel("Evaluation Group", y=0.02)
    fig.suptitle(
        f"Sex-specific PRS accuracy by ancestry ({ANALYSIS_TO_PHENOTYPE_MAP[phenotype]}, {BIOBANK_NAME_MAP_SHORT[test_biobank]})",
        y=0.98,
    )
    fig.subplots_adjust(left=0.12, right=0.82, top=0.88, bottom=0.24, hspace=0.28)

    plt.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close()


def extract_female_age_prs_ancestry_stratified_accuracy(pheno, test_biobank):
    dataset_path = f"data/harmonized_data/{pheno}/{test_biobank}/full_data.pkl"
    dat = PRSDataset.from_pickle(dataset_path)

    dat.data["SexLabel"] = dat.data["Sex"].astype(int).astype(str).map(SEX_LABEL_MAP)
    dat.filter_samples(dat.data["SexLabel"] == "Female")

    dat.data["AgeGroup2"] = np.array(["Age<=55", "Age>55"]).take(
        dat.get_data_columns("Age").flatten() > 55
    )
    dat.data["CoarseAncestry"] = np.where(
        dat.data["Ancestry"] == "EUR", "European", "Non-European"
    )
    dat.data["ContinentalAncestry"] = dat.data["Ancestry"]
    dat.data["Age+CoarseAncestry"] = (
        dat.data["AgeGroup2"] + "|" + dat.data["CoarseAncestry"]
    )
    dat.data["Age+ContinentalAncestry"] = (
        dat.data["AgeGroup2"] + "|" + dat.data["ContinentalAncestry"]
    )

    eval_df = stratified_evaluation(
        dat,
        trained_models=None,
        cat_group_cols=[
            "AgeGroup2",
            "Age+CoarseAncestry",
            "Age+ContinentalAncestry",
        ],
        metrics=["Incremental_R2"],
        min_group_size=DEFAULT_MIN_GROUP_SIZE,
    )

    eval_df = eval_df.loc[
        (eval_df["metric"] == "Incremental_R2")
        & (eval_df["metric_kind"] == "base")
    ].copy()

    all_df = eval_df.loc[
        (eval_df["eval_category"] == "AgeGroup2")
        & (eval_df["eval_group"].isin(["Age<=55", "Age>55"]))
    ].copy()
    all_df["Age Group"] = all_df["eval_group"]
    all_df["eval_category"] = "CoarseAncestry"
    all_df["eval_group"] = "All"

    coarse_df = eval_df.loc[
        eval_df["eval_category"] == "Age+CoarseAncestry"
    ].copy()
    coarse_df[["Age Group", "eval_group"]] = coarse_df["eval_group"].str.split(
        "|", expand=True, regex=False
    )
    coarse_df = coarse_df.loc[
        coarse_df["eval_group"].isin(["European", "Non-European"])
    ].copy()
    coarse_df["eval_category"] = "CoarseAncestry"

    continental_df = eval_df.loc[
        eval_df["eval_category"] == "Age+ContinentalAncestry"
    ].copy()
    continental_df[["Age Group", "eval_group"]] = continental_df[
        "eval_group"
    ].str.split("|", expand=True, regex=False)
    continental_df = continental_df.loc[
        continental_df["eval_group"].isin(["AFR", "CSA", "MID", "EAS"])
    ].copy()
    continental_df["eval_category"] = "ContinentalAncestry"

    plot_df = pd.concat([all_df, coarse_df, continental_df], ignore_index=True)
    plot_df["Model Name"] = np.select(
        [
            plot_df["model_name"].astype(str).str.endswith("_F"),
            plot_df["model_name"].astype(str).str.endswith("_M"),
        ],
        ["Female PRS", "Male PRS"],
        default=None,
    )
    plot_df = plot_df.loc[plot_df["Model Name"].notnull()].copy()

    plot_df.rename(
        columns={"eval_group": "Evaluation Group", "value": "Incremental_R2"},
        inplace=True,
    )

    return plot_df[
        ["Age Group", "Evaluation Group", "Model Name", "Incremental_R2"]
    ]


def plot_female_age_prs_ancestry_stratified_accuracy(
    phenotype, test_biobank, output_path
):
    dataset_path = f"data/harmonized_data/{phenotype}/{test_biobank}/full_data.pkl"
    if not osp.exists(dataset_path):
        print(
            f"> Skipping female age ancestry-stratified accuracy for {phenotype} "
            f"({test_biobank}): dataset not found."
        )
        return

    try:
        plot_df = extract_female_age_prs_ancestry_stratified_accuracy(
            phenotype, test_biobank
        )
    except Exception as e:
        print(
            f"> Skipping female age ancestry-stratified accuracy for {phenotype} "
            f"({test_biobank}): {e}"
        )
        return

    if len(plot_df) == 0:
        print(
            f"> Skipping female age ancestry-stratified accuracy for {phenotype} "
            f"({test_biobank}): no plottable metrics."
        )
        return

    age_order = ["Age<=55", "Age>55"]
    age_title = {"Age<=55": "Females <= 55", "Age>55": "Females > 55"}
    group_order = ["All", "European", "Non-European", "AFR", "CSA", "MID", "EAS"]
    model_order = ["Female PRS", "Male PRS"]
    model_palette = {"Female PRS": "#F98866", "Male PRS": "#A1BE95"}
    model_offsets = {"Female PRS": -0.12, "Male PRS": 0.12}

    fig, axes = plt.subplots(
        len(age_order),
        1,
        figsize=(7.6, 5.0),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).ravel()

    for row_idx, age_group in enumerate(age_order):
        ax = axes[row_idx]
        sub_df = plot_df.loc[plot_df["Age Group"] == age_group].copy()
        x_pos = np.arange(len(group_order))

        if len(sub_df) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#777777",
            )
        else:
            for group_idx, group in enumerate(group_order):
                group_df = sub_df.loc[sub_df["Evaluation Group"] == group]
                values = {}
                for model in model_order:
                    model_values = group_df.loc[
                        group_df["Model Name"] == model, "Incremental_R2"
                    ]
                    if len(model_values) == 0:
                        continue
                    values[model] = float(model_values.iloc[0])
                    ax.scatter(
                        group_idx + model_offsets[model],
                        values[model],
                        color=model_palette[model],
                        edgecolor="white",
                        linewidth=0.6,
                        s=36,
                        zorder=3,
                    )

                if len(values) == len(model_order):
                    ax.plot(
                        [
                            group_idx + model_offsets["Female PRS"],
                            group_idx + model_offsets["Male PRS"],
                        ],
                        [values["Female PRS"], values["Male PRS"]],
                        color="#9A9A9A",
                        linewidth=0.8,
                        alpha=0.7,
                        zorder=2,
                    )

        ax.axhline(0.0, color="#B8B8B8", linewidth=0.8, linestyle=":")
        ax.axvline(2.5, color="#D0D0D0", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_order, rotation=20)
        ax.tick_params(axis="x", labelbottom=(row_idx == len(age_order) - 1))
        ax.set_title(age_title[age_group])
        ax.set_ylabel("")
        ax.set_xlabel("")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=model,
            markerfacecolor=model_palette[model],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=6,
        )
        for model in model_order
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.supylabel("Incremental $R^2$")
    fig.supxlabel("Evaluation Group", y=0.02)
    fig.suptitle(
        f"Female age-stratified PRS accuracy by ancestry ({ANALYSIS_TO_PHENOTYPE_MAP[phenotype]}, {BIOBANK_NAME_MAP_SHORT[test_biobank]})",
        y=0.98,
    )
    fig.subplots_adjust(left=0.12, right=0.82, top=0.88, bottom=0.24, hspace=0.28)

    plt.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close()


def extract_accuracy_data(
    test_biobank="ukbb",
    train_biobank="ukbb",
    restrict_to_same_biobank=True,
    dataset=None,
):
    if dataset is None:
        dataset = "test_data"

    dfs = []

    for pheno in phenotypes:
        # Extract accuracy metrics:
        f = f"data/evaluation/{pheno}/{test_biobank}/{dataset}.csv"
        try:
            df = read_transform_eval_metrics(f)
        except Exception as e:
            print(e)
            continue

        df = df.loc[
            (df["model_category"] != "MoE")
            | df["model_name"].isin(
                [  # f'MoE-CFG ({args.biobank})',
                    f"{args.moe_model}"
                ]
            )
        ]
        df["model_name"] = df["model_name"].str.replace(
            f"{args.moe_model}", "MoEPRS", regex=False
        )

        if train_biobank is not None:
            df = df.loc[df["train_biobank"] == train_biobank]
        elif restrict_to_same_biobank:
            df = df.loc[df["train_biobank"] == df["test_biobank"]]

        df = postprocess_metrics_df(
            df,
            "Incremental_R2",
            category="Sex",
            aggregate_single_prs=False,
            add_training_biobank_to_model_name=False,
        )

        df["Model Name"] = df["Model Name"].replace("SexMatchedPRS", "Sex-matched PRS")
        df["Model Name"] = df["Model Name"].replace("Male", "Male PRS")
        df["Model Name"] = df["Model Name"].replace("Female", "Female PRS")

        # Remove sex-matched PRS from evaluation groups other than "All"
        df = df.loc[
            ~(
                (df["Model Name"] == "Sex-matched PRS")
                & (df["Evaluation Group"] != "All")
            )
        ]

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No fold-aware evaluation metrics found for {test_biobank}/{dataset}."
        )

    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    dfs["phenotype"] += " (" + BIOBANK_NAME_MAP_SHORT[test_biobank] + ")"

    return dfs


def extract_non_eur_accuracy_data(test_biobank="ukbb", train_biobank=None):
    if train_biobank is None:
        train_biobank = test_biobank

    dfs = []

    for pheno in phenotypes:
        fold_specs = _evaluation_fold_specs(pheno, test_biobank, train_biobank)
        if not fold_specs:
            print(
                f"> Skipping non-EUR accuracy for {pheno} ({test_biobank}): "
                f"no {train_biobank} fold models/datasets found."
            )
            continue

        fold_dfs = []
        for fold, dataset_path, model_root in fold_specs:
            try:
                dat = PRSDataset.from_pickle(dataset_path)
            except Exception as e:
                print(f"> Skipping {pheno} {fold} non-EUR data: {e}")
                continue

            dat.filter_samples(dat.data["Ancestry"] != "EUR")
            dat.data["SexG"] = (
                dat.data["Sex"].astype(int).astype(str).map(SEX_LABEL_MAP)
            )

            trained_models = {}
            for model_name, model_class, filename in (
                ("MoEPRS", MoEPRS, f"{args.moe_model}.pkl"),
                ("MultiPRS", MultiPRS, "MultiPRS.pkl"),
                ("SexMatchedPRS", AttributePartitionedPRS, "SexMatchedPRS.pkl"),
            ):
                try:
                    trained_models[model_name] = model_class.from_saved_model(
                        f"{model_root}/{filename}"
                    )
                except Exception as e:
                    print(f"> Skipping {pheno} {fold} {model_name}: {e}")

            if not trained_models:
                continue

            try:
                fold_df = stratified_evaluation(
                    dat,
                    trained_models=trained_models,
                    cat_group_cols=["SexG"],
                    metrics=["Incremental_R2"],
                    min_group_size=DEFAULT_MIN_GROUP_SIZE,
                )
            except Exception as e:
                print(f"> Skipping {pheno} {fold} non-EUR evaluation: {e}")
                continue

            fold_df["test_fold"] = fold
            fold_df["train_fold"] = fold
            fold_df["evaluation_scope"] = (
                "held_out_fold"
                if test_biobank == train_biobank
                else "external_full"
            )
            fold_dfs.append(fold_df)

        if not fold_dfs:
            continue

        df = aggregate_cross_validation_metrics(
            pd.concat(fold_dfs, axis=0, ignore_index=True)
        )

        df["analysis_id"] = pheno
        df["test_biobank"] = test_biobank
        df["phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP.get(pheno, pheno)
        df["model_name"] = df["model_name"].map(
            lambda x: MODEL_NAME_MAP.get(pheno, {}).get(x, x)
        )

        df = postprocess_metrics_df(
            df,
            "Incremental_R2",
            category="SexG",
            aggregate_single_prs=False,
            add_training_biobank_to_model_name=False,
            min_sample_size=20,
        )

        df["Model Name"] = df["Model Name"].replace("Male", "Male PRS")
        df["Model Name"] = df["Model Name"].replace("Female", "Female PRS")
        df["Model Name"] = df["Model Name"].replace("SexMatchedPRS", "Sex-matched PRS")

        # Keep behavior consistent with the standard subpanel figure.
        df = df.loc[
            ~(
                (df["Model Name"] == "Sex-matched PRS")
                & (df["Evaluation Group"] != "All")
            )
        ]

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No fold-aware non-EUR evaluations found for {test_biobank} "
            f"using {train_biobank}-trained models."
        )

    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    dfs["phenotype"] += f" ({BIOBANK_NAME_MAP_SHORT[test_biobank]})"

    return dfs


def plot_phenotypic_variance(pheno, biobank="ukbb"):
    dataset = PRSDataset.from_pickle(
        f"data/harmonized_data/{pheno}/{biobank}/full_data.pkl"
    )

    dataset.data["SexG"] = (
        dataset.data["Sex"].astype(int).astype(str).map(SEX_LABEL_MAP)
    )
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

    parser.add_argument(
        "--mixing-weight-fold",
        dest="mixing_weight_fold",
        type=str,
        default="fold_1",
        help=(
            "Reference model fold used for all mixing-weight figures. "
            "The selected model is evaluated on the full dataset (default: fold_1)."
        ),
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
        "Sex-matched PRS": "#66C2A5",
    }

    hue_order = ["Sex-matched PRS", "MoEPRS", "MultiPRS", "Female PRS", "Male PRS"]
    phenotype_order = ["Waist-hip ratio", "Log Testosterone", "Log Creatinine", "Urate"]

    print(">>> Section 1 Figures <<<")

    ukbb_metrics_dfs = extract_accuracy_data()
    ukbb_w_dfs = extract_weights_data(reference_fold=args.mixing_weight_fold)

    ukb_col_order = [p + " (UKB)" for p in phenotype_order]
    ukb_urate_order = ["Urate (UKB)"]
    ukbb_urate_w_dfs = ukbb_w_dfs.loc[
        ukbb_w_dfs["phenotype"].isin(ukb_urate_order)
    ].copy()

    plot_combined_accuracy_metrics(
        ukbb_metrics_dfs,
        "figures/section_1/accuracy_subpanels_all_ukbb.pdf",
        column="phenotype",
        col_order=ukb_col_order,
        palette=palette,
        hue_order=hue_order,
        test_models=[("MoEPRS", "MultiPRS"), ("MoEPRS", "Sex-matched PRS")],
        significance_symbols=["*", "+"],
    )

    ukbb_non_eur_metrics_dfs = extract_non_eur_accuracy_data(test_biobank="ukbb")
    plot_combined_accuracy_metrics(
        ukbb_non_eur_metrics_dfs,
        "figures/section_1/accuracy_subpanels_non_eur_all_ukbb.pdf",
        column="phenotype",
        col_order=ukb_col_order,
        palette=palette,
        hue_order=hue_order,
        test_models=[("MoEPRS", "MultiPRS"), ("MoEPRS", "Sex-matched PRS")],
        significance_symbols=["*", "+"],
    )

    plot_gate_mixing_weights_colored_by_sex(
        ukbb_urate_w_dfs,
        "figures/section_1/mixing_weights_by_sex_urate_ukbb.png",
        order=ukb_urate_order,
    )

    plot_gate_mixing_weights_colored_by_ancestry(
        ukbb_urate_w_dfs,
        "figures/section_1/mixing_weights_by_ancestry_urate_ukbb.png",
        order=ukb_urate_order,
    )

    plot_gate_mixing_weights_categorical(
        ukbb_w_dfs,
        "figures/section_1/mixing_weights_categorical_all_ukbb.png",
        order=ukb_col_order,
    )

    plot_gate_mixing_weights_continental_ancestry(
        ukbb_w_dfs,
        "figures/section_1/mixing_weights_continental_ancestry_all_ukbb.png",
        order=ukb_col_order,
    )

    cartagene_metrics_dfs = extract_accuracy_data(
        test_biobank="cartagene",
        train_biobank="cartagene",
        restrict_to_same_biobank=True,
        dataset="test_data",
    )
    cartagene_w_dfs = extract_weights_data(
        biobank="cartagene",
        train_biobank="cartagene",
        reference_fold=args.mixing_weight_fold,
    )

    # Exclude testosterone:
    cag_col_order = [p + " (CaG)" for p in phenotype_order if "Testosterone" not in p]
    cag_urate_order = ["Urate (CaG)"]
    cartagene_urate_w_dfs = cartagene_w_dfs.loc[
        cartagene_w_dfs["phenotype"].isin(cag_urate_order)
    ].copy()

    plot_combined_accuracy_metrics(
        cartagene_metrics_dfs,
        "figures/section_1/accuracy_subpanels_all_cartagene.pdf",
        column="phenotype",
        col_order=cag_col_order,
        palette=palette,
        hue_order=hue_order,
        test_models=[("MoEPRS", "MultiPRS"), ("MoEPRS", "Sex-matched PRS")],
        significance_symbols=["*", "+"],
    )

    cartagene_non_eur_metrics_dfs = extract_non_eur_accuracy_data(
        test_biobank="cartagene", train_biobank="cartagene"
    )
    plot_combined_accuracy_metrics(
        cartagene_non_eur_metrics_dfs,
        "figures/section_1/accuracy_subpanels_non_eur_all_cartagene.pdf",
        column="phenotype",
        col_order=cag_col_order,
        palette=palette,
        hue_order=hue_order,
        test_models=[("MoEPRS", "MultiPRS"), ("MoEPRS", "Sex-matched PRS")],
        significance_symbols=["*", "+"],
    )

    plot_gate_mixing_weights_colored_by_sex(
        cartagene_urate_w_dfs,
        "figures/section_1/mixing_weights_by_sex_urate_cartagene.png",
        order=cag_urate_order,
    )

    plot_gate_mixing_weights_colored_by_ancestry(
        cartagene_urate_w_dfs,
        "figures/section_1/mixing_weights_by_ancestry_urate_cartagene.png",
        order=cag_urate_order,
    )

    plot_gate_mixing_weights_categorical(
        cartagene_w_dfs,
        "figures/section_1/mixing_weights_categorical_all_cartagene.png",
        order=cag_col_order,
    )

    plot_gate_mixing_weights_continental_ancestry(
        cartagene_w_dfs,
        "figures/section_1/mixing_weights_continental_ancestry_all_cartagene.png",
        order=cag_col_order,
    )

    sns.set_context("paper", font_scale=1.25)

    """
    plot_relative_stratified_evaluation(
        phenotype="LOG_CRTN_SEX",
        output_path="figures/section_1/accuracy_stratified_ratio_creatinine_mixed.pdf",
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
    """

    plot_female_age_prs_ancestry_stratified_accuracy(
        "URT_SEX",
        test_biobank="ukbb",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_female_age_urate_ukbb.pdf",
    )
    plot_female_age_prs_ancestry_stratified_accuracy(
        "URT_SEX",
        test_biobank="cartagene",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_female_age_urate_cartagene.pdf",
    )

    """
    plot_relative_stratified_evaluation(
        phenotype="WHR_SEX",
        output_path="figures/section_1/accuracy_stratified_ratio_whr_mixed.pdf",
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
        output_path="figures/section_1/accuracy_stratified_ratio_testosterone_ukbb.pdf",
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
    """

    plot_sex_prs_ancestry_stratified_accuracy(
        "LOG_CRTN_SEX",
        test_biobank="ukbb",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_creatinine_ukbb.pdf",
    )
    plot_sex_prs_ancestry_stratified_accuracy(
        "LOG_CRTN_SEX",
        test_biobank="cartagene",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_creatinine_cartagene.pdf",
    )
    plot_sex_prs_ancestry_stratified_accuracy(
        "URT_SEX",
        test_biobank="ukbb",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_urate_ukbb.pdf",
    )
    plot_sex_prs_ancestry_stratified_accuracy(
        "URT_SEX",
        test_biobank="cartagene",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_urate_cartagene.pdf",
    )
    plot_sex_prs_ancestry_stratified_accuracy(
        "WHR_SEX",
        test_biobank="ukbb",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_whr_ukbb.pdf",
    )
    plot_sex_prs_ancestry_stratified_accuracy(
        "WHR_SEX",
        test_biobank="cartagene",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_whr_cartagene.pdf",
    )
    plot_sex_prs_ancestry_stratified_accuracy(
        "LOG_TST_SEX",
        test_biobank="ukbb",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_testosterone_ukbb.pdf",
    )
    plot_sex_prs_ancestry_stratified_accuracy(
        "LOG_TST_SEX",
        test_biobank="cartagene",
        output_path="figures/section_1/accuracy_stratified_by_ancestry_testosterone_cartagene.pdf",
    )

    plot_phenotypic_variance("LOG_CRTN_SEX", biobank="ukbb")
    plot_phenotypic_variance("URT_SEX", biobank="ukbb")
    plot_phenotypic_variance("WHR_SEX", biobank="ukbb")
    plot_phenotypic_variance("LOG_CRTN_SEX", biobank="cartagene")
    plot_phenotypic_variance("URT_SEX", biobank="cartagene")
    plot_phenotypic_variance("WHR_SEX", biobank="cartagene")
