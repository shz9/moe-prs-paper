import argparse
import glob
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from error_bars import add_error_bars, add_error_bars_to_catplot
from magenpy.utils.system_utils import makedir
from plot_predictive_performance import generate_model_colors, postprocess_metrics_df
from plot_utils import (
    ANALYSIS_TO_TABLE_MAP,
    METRIC_NAME_MAP,
    read_transform_eval_metrics,
    sort_groups,
)
from significance_annotation import add_significance_annotations

# ---------------------------------------------------------------------------


def plot_combined_accuracy_metrics(
    metrics_df,
    output_f=None,
    x="Evaluation Group",
    metric="Incremental_R2",
    column="analysis_id",
    row=None,
    palette="Set2",
    order=None,
    hue_order=None,
    col_order=None,
    col_wrap=None,
    row_order=None,
    test_models=None,  # Test if models are significantly different
    significance_symbols=None,
    sharey=False,
    sharex=False,
    height=5,
    aspect=1,
    x_tick_rotation=None,
    legend_title=None,
    legend_fontsize=None,
    legend_title_fontsize=None,
    ylim=None,
    title=None,
):
    # ---------------------------------------------------------------------
    # Sanity checks / preparation

    if test_models is not None:
        if len(test_models) == 2 and isinstance(test_models[0], str):
            test_models = [test_models]

        for tm in test_models:
            assert len(tm) == 2

        assert f"{metric}_err" in metrics_df

    if hue_order is None:
        _, hue_order = generate_model_colors(metrics_df, metric)

    if order is None and x == "Evaluation Group":
        order = sort_groups(metrics_df["Evaluation Group"].unique())

    # ---------------------------------------------------------------------

    grid = sns.catplot(
        x=x,
        y=metric,
        col=column,
        col_wrap=col_wrap,
        col_order=col_order,
        row=row,
        row_order=row_order,
        order=order,
        hue="Model Name",
        palette=palette,
        hue_order=hue_order,
        kind="bar",
        errorbar=None,
        height=height,
        aspect=aspect,
        sharey=sharey,
        data=metrics_df,
    )

    if f"{metric}_err" in metrics_df.columns:
        is_unfaceted = column is None and row is None
        plot_target = grid.axes.flat[0] if is_unfaceted else grid
        if is_unfaceted:
            add_error_bars(
                plot_target,
                metrics_df,
                x,
                metric,
                hue="Model Name",
                hue_order=hue_order,
                order=order,
            )
        else:
            add_error_bars_to_catplot(
                grid,
                metrics_df,
                x,
                metric,
                hue="Model Name",
                hue_order=hue_order,
                col=column,
                row=row,
                order=order,
            )

        if test_models is not None:
            add_significance_annotations(
                plot_target,
                metrics_df,
                x,
                metric,
                f"{metric}_err",
                hue="Model Name",
                hue_order=hue_order,
                test_pairs=test_models,
                x_labels=order,
                symbols=significance_symbols,
            )

    grid.set_axis_labels(
        x_var=x,
        y_var=METRIC_NAME_MAP[metric],
    )
    if ylim is not None:
        grid.set(ylim=ylim)

    # Update legend title
    if legend_title is not None and grid.legend is not None:
        grid.legend.set_title(legend_title)
    if grid.legend is not None:
        if legend_fontsize is not None:
            for text in grid.legend.texts:
                text.set_fontsize(legend_fontsize)
        if legend_title_fontsize is not None:
            grid.legend.get_title().set_fontsize(legend_title_fontsize)

    subtitle_to_remove = None

    if column is not None:
        subtitle_to_remove = column
    if row is not None:
        subtitle_to_remove = row

    if subtitle_to_remove is not None:
        for ax in grid.axes.flat:
            axis_title = ax.get_title()
            if axis_title.startswith(f"{subtitle_to_remove} = "):
                ax.set_title(axis_title.replace(f"{subtitle_to_remove} = ", ""))

    if title is not None:
        grid.figure.suptitle(title)

    if x_tick_rotation is not None:
        grid.tick_params(axis="x", rotation=x_tick_rotation)
        if x_tick_rotation != 0:
            for ax in grid.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_ha("right")

    if output_f is None:
        plt.show()
    else:
        plt.savefig(output_f, bbox_inches="tight")
        plt.close()

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot predictive performance of PRS models by category."
    )

    parser.add_argument(
        "--biobank",
        dest="biobank",
        type=str,
        required=True,
        choices={"ukbb", "cartagene"},
        help="The name of the biobank to plot the accuracy metrics for.",
    )
    parser.add_argument(
        "--category",
        dest="category",
        type=str,
        default=["Coarse Ancestry"],
        nargs="+",
        help="The category (or list of categories) to plot the predictive performance for.",
    )
    parser.add_argument(
        "--aggregate-single-prs",
        dest="aggregate_single_prs",
        action="store_true",
        default=False,
        help="Aggregate the results for SinglePRS models (select best for each category).",
    )
    parser.add_argument(
        "--restrict-to-same-biobank",
        dest="restrict_to_same_biobank",
        action="store_true",
        default=False,
        help="Restrict the analysis to models trained and tested on the same biobank.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        choices={"train", "test", "full"},
        default=None,
        help=(
            "Dataset to plot. Defaults to held-out folds for UKBB and the full "
            "held-out cohort for CARTaGENE."
        ),
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
        default="Nagelkerke_R2",
        help="The metric to plot for binary phenotypes.",
    )
    parser.add_argument(
        "--metric-kind",
        dest="metric_kind",
        type=str,
        choices={"base", "incremental_vs_ref"},
        default="base",
        help="The type of incremental metric to plot.",
    )
    parser.add_argument(
        "--extension",
        dest="extension",
        type=str,
        default=".png",
        help="The file extension to use for saving the plot.",
    )
    parser.add_argument(
        "--moe-model",
        dest="moe_model",
        type=str,
        default="MoE",
        help="The name of the MoE model to plot as reference.",
    )

    args = parser.parse_args()

    dataset = args.dataset
    if dataset is None:
        dataset = "full" if args.biobank == "cartagene" else "test"

    sns.set_context("paper", font_scale=1.5)

    metrics_dfs = {}

    evaluation_dirs = glob.glob(f"data/evaluation/*/{args.biobank}")
    for evaluation_dir in evaluation_dirs:
        f = osp.join(evaluation_dir, f"{dataset}_data.csv")
        analysis_id = f.split("/")[-3]

        analysis_table_id = ANALYSIS_TO_TABLE_MAP.get(analysis_id)
        if analysis_table_id is None:
            continue

        # Determine stratification variable for evaluation:
        if analysis_table_id == "sex_biased_prs_table":
            strat_var = ["Sex"]
        else:
            strat_var = ["Coarse Ancestry"]

        df = read_transform_eval_metrics(f)
        if "train_biobank" in df.columns:
            df["train_biobank"] = df["train_biobank"].astype(str).str.lower()

        keep_train_biobanks = [args.biobank]
        if not args.restrict_to_same_biobank:
            other_biobank = ["cartagene", "ukbb"][args.biobank == "cartagene"]
            keep_train_biobanks.append(other_biobank)

        df = df.loc[
            (df["model_category"] != "MoE")
            | (
                (df["model_name"] == args.moe_model)
                & (df["train_biobank"].isin(keep_train_biobanks))
            )
        ]

        if args.restrict_to_same_biobank:
            df = df.loc[df["train_biobank"] == df["test_biobank"]]

        eval_metric = (
            args.binary_metric
            if args.binary_metric in set(df["metric"].unique())
            else "Incremental_R2"
        )

        for eval_cat in strat_var:
            eval_df = postprocess_metrics_df(
                df,
                eval_metric,
                metric_kind=args.metric_kind,
                category=eval_cat,
                aggregate_single_prs=args.aggregate_single_prs,
                add_training_biobank_to_model_name=not args.restrict_to_same_biobank,
            )

            if args.binary_metric in eval_df.columns:
                eval_df["Incremental_R2"] = eval_df[args.binary_metric]
                if f"{args.binary_metric}_err" in eval_df.columns:
                    eval_df["Incremental_R2_err"] = eval_df[f"{args.binary_metric}_err"]

            metric_cat = f"{analysis_table_id}_{eval_cat}"

            if metric_cat not in metrics_dfs:
                metrics_dfs[metric_cat] = [eval_df]
            else:
                metrics_dfs[metric_cat].append(eval_df)

    # The output directory is determined by the metric kind, biobank, and dataset.
    output_dir = f"figures/accuracy/{args.biobank}/{dataset}/{args.metric_kind}/"

    output_dir = osp.join(
        output_dir, ["cross_biobank", "same_biobank"][args.restrict_to_same_biobank]
    )

    for pheno_eval_cat, dfs in metrics_dfs.items():
        if len(dfs) < 1:
            raise ValueError(
                f"No data to plot after applying filters for {pheno_eval_cat}."
            )

        pheno_cat = "_".join(pheno_eval_cat.split("_")[:-1])
        eval_cat = pheno_eval_cat.split("_")[-1]

        output_dir_cat = osp.join(output_dir, pheno_cat)
        makedir(output_dir_cat)

        plot_combined_accuracy_metrics(
            pd.concat(dfs, axis=0).reset_index(drop=True),
            osp.join(
                output_dir_cat,
                f"combined_metrics_{eval_cat}_{args.moe_model}{args.extension}",
            ),
            metric="Incremental_R2",
            # col_order=phenotype_cats[pheno_cat],
            col_wrap=min(5, len(dfs)),
            test_models=[
                (f"{args.moe_model} ({args.biobank})", f"MultiPRS ({args.biobank})"),
                (f"{args.moe_model} ({args.biobank})", "Best Single Source PRS"),
            ],
            significance_symbols=("*", "+"),  # "◆"),
        )
