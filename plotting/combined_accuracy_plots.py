import argparse
import glob
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from error_bars import add_error_bars_to_catplot
from magenpy.utils.system_utils import makedir
from plot_predictive_performance import generate_model_colors, postprocess_metrics_df
from plot_utils import (
    ANALYSIS_TO_TABLE_MAP,
    read_eval_metrics,
    sort_groups,
    transform_eval_metrics,
)
from significance_annotation import add_significance_annotations

# ---------------------------------------------------------------------------


def plot_combined_accuracy_metrics(
    metrics_df,
    output_f=None,
    x="Evaluation Group",
    metric="Incremental_R2",
    column="AnalysisID",
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
        height=height,
        aspect=aspect,
        sharey=sharey,
        data=metrics_df,
    )

    if f"{metric}_err" in metrics_df.columns:
        add_error_bars_to_catplot(
            grid,
            metrics_df,
            x,
            metric,
            hue="Model Name",
            hue_order=hue_order,
            col=column,
            row=row,
        )

        if test_models is not None:
            add_significance_annotations(
                grid,
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
        y_var={
            "Incremental_R2": "Incremental $R^2$",
            "Liability_R2": "Liability $R^2$",
            "CORR": "Pearson $R$",
        }[metric],
    )

    subtitle_to_remove = None

    if column is not None:
        subtitle_to_remove = column
    if row is not None:
        subtitle_to_remove = row

    if subtitle_to_remove is not None:
        for ax in grid.axes.flat:
            title = ax.get_title()
            if title.startswith(f"{subtitle_to_remove} = "):
                ax.set_title(title.replace(f"{subtitle_to_remove} = ", ""))

    if x_tick_rotation is not None:
        grid.tick_params(axis="x", rotation=x_tick_rotation)

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
        default=["Ancestry"],
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
        choices={"train", "test"},
        default="test",
        help="The type of dataset to plot predictive performance on.",
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

    sns.set_context("paper", font_scale=1.5)

    metrics_dfs = {}

    for f in glob.glob(f"data/evaluation/*/{args.biobank}/{args.dataset}_data.csv"):
        analysis_id = f.split("/")[-3]

        analysis_table_id = ANALYSIS_TO_TABLE_MAP.get(analysis_id)
        if analysis_table_id is None:
            continue

        # Determine stratification variable for evaluation:
        if analysis_table_id == "sex_biased_prs_table":
            strat_var = ["Sex"]
        else:
            strat_var = ["Ancestry", "Coarse Ancestry"]

        df = transform_eval_metrics(read_eval_metrics(f))

        keep_models_moe = [f"{args.moe_model} ({args.biobank})"]
        if not args.restrict_to_same_biobank:
            other_biobank = ["cartagene", "ukbb"][args.biobank == "cartagene"]
            keep_models_moe.append(f"{args.moe_model} ({other_biobank})")

        df = df.loc[
            (df["Model Category"] != "MoE") | df["Model Name"].isin(keep_models_moe)
        ]

        if args.restrict_to_same_biobank:
            df = df.loc[df["Training biobank"] == df["Test biobank"]]

        if "Liability_R2" in df.columns:
            df["Incremental_R2"] = df["Liability_R2"]
            df["Incremental_R2_err"] = df["Liability_R2_err"]

        for eval_cat in strat_var:
            eval_df = postprocess_metrics_df(
                df,
                "Incremental_R2",
                category=eval_cat,
                aggregate_single_prs=args.aggregate_single_prs,
            )

            metric_cat = f"{analysis_table_id}_{eval_cat}"

            if metric_cat not in metrics_dfs:
                metrics_dfs[metric_cat] = [eval_df]
            else:
                metrics_dfs[metric_cat].append(eval_df)

    output_dir = f"figures/accuracy/{args.biobank}/{args.dataset}/"
    output_dir = osp.join(
        output_dir, ["", "same_biobank"][args.restrict_to_same_biobank]
    )

    makedir(output_dir)

    for pheno_eval_cat, dfs in metrics_dfs.items():
        if len(dfs) < 1:
            raise ValueError(
                f"No data to plot after applying filters for {pheno_eval_cat}."
            )

        pheno_cat = "_".join(pheno_eval_cat.split("_")[:-1])

        plot_combined_accuracy_metrics(
            pd.concat(dfs, axis=0).reset_index(drop=True),
            osp.join(
                output_dir, f"combined_metrics_{args.moe_model}_{pheno_eval_cat}.eps"
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
