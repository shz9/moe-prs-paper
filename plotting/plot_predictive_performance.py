import argparse
import os.path as osp
import sys

import seaborn as sns
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(parent_dir, "evaluation/"))
from accuracy_plots import grouped_plot
from plot_utils import postprocess_metrics_df, read_transform_eval_metrics, sort_groups


def generate_model_colors(metrics_df, metric, palette="Set2", n_model_types=4):
    palette = sns.color_palette(palette, n_model_types)

    if "SinglePRS+Covariates" in metrics_df["model_category"].unique():
        single_model_label = "SinglePRS+Covariates"
    else:
        single_model_label = "SinglePRS"

    # ---------------------------------------------------------------------
    # Determine the hue order:
    single_hue_order = (
        metrics_df.loc[metrics_df["model_category"] == single_model_label]
        .groupby("Model Name")["value"]
        .mean()
        .sort_values(ascending=metric.endswith("MSE"))
        .index
    )

    single_model_colors = dict(
        zip(
            single_hue_order,
            sns.light_palette(palette[0], max(len(single_hue_order), 5), reverse=True),
        )
    )

    multiprs_hue_order = (
        metrics_df.loc[metrics_df["model_category"] == "MultiPRS"]
        .groupby("Model Name")["value"]
        .mean()
        .sort_values(ascending=metric.endswith("MSE"))
        .index
    )

    multiprs_hue_colors = dict(
        zip(
            multiprs_hue_order,
            sns.light_palette(
                palette[1], max(len(multiprs_hue_order), 5), reverse=True
            ),
        )
    )

    awm_hue_order = (
        metrics_df.loc[metrics_df["model_category"] == "AncestryWeightedPRS"]
        .groupby("Model Name")["value"]
        .mean()
        .sort_values(ascending=metric.endswith("MSE"))
        .index
    )

    awm_hue_colors = dict(
        zip(
            awm_hue_order,
            sns.light_palette(palette[2], max(len(awm_hue_order), 5), reverse=True),
        )
    )

    moe_hue_order = (
        metrics_df.loc[metrics_df["model_category"] == "MoE"]
        .groupby("Model Name")["value"]
        .mean()
        .sort_values(ascending=metric.endswith("MSE"))
        .index
    )

    moe_hue_colors = dict(
        zip(
            moe_hue_order,
            sns.light_palette(palette[3], max(len(moe_hue_order), 5), reverse=True),
        )
    )

    hue_order = (
        list(moe_hue_order)
        + list(multiprs_hue_order)
        + list(awm_hue_order)
        + list(single_hue_order)
    )
    colors = {
        **moe_hue_colors,
        **multiprs_hue_colors,
        **awm_hue_colors,
        **single_model_colors,
    }

    return colors, hue_order


def performance_by_category_plots(
    metrics_df,
    output_file=None,
    category="Coarse Ancestry",
    metric="Incremental_R2",
    min_sample_size=100,
    aggregate_single_prs=True,
):
    sub_metrics_df = postprocess_metrics_df(
        metrics_df,
        metric=metric,
        category=category,
        min_sample_size=min_sample_size,
        aggregate_single_prs=aggregate_single_prs,
    )

    colors, hue_order = generate_model_colors(sub_metrics_df, metric=metric)

    phenotype = sub_metrics_df["phenotype"].iloc[0]

    # ---------------------------------------------------------------------

    grouped_plot(
        sub_metrics_df,
        kind="bar",
        order=sort_groups(sub_metrics_df["eval_group"].unique()),
        palette=colors,
        hue_order=hue_order,
        title=f"Prediction accuracy for {phenotype}",
        output_file=output_file,
        metric="value",
    )

    return sub_metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot predictive performance of PRS models by category."
    )

    parser.add_argument(
        "--metrics-file",
        dest="metrics_file",
        type=str,
        required=True,
        help="The path to the metrics file.",
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
        "--metrics",
        dest="metrics",
        type=str,
        default=None,
        nargs="+",
        help="The performance metric to plot.",
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
        "--train-dataset",
        dest="train_dataset",
        type=str,
        default=None,
        help="If specified, then use models trained on this dataset only.",
    )
    parser.add_argument(
        "--extension",
        dest="extension",
        type=str,
        default=".png",
        help="The file extension to use for saving the plot.",
    )

    args = parser.parse_args()

    print(
        "> Plotting predictive performance for the following evaluation metrics:\n",
        args.metrics_file,
    )

    eval_df = read_transform_eval_metrics(args.metrics_file)

    if args.train_dataset is not None:
        eval_df = eval_df.loc[eval_df["train_source"] == args.train_dataset]
        d_suffix = f"_{args.train_dataset.replace('/', '_')}"
    else:
        d_suffix = ""

    if args.restrict_to_same_biobank:
        eval_df = eval_df.loc[eval_df["train_biobank"] == eval_df["test_biobank"]]
        rs_suffix = "_rs"
    else:
        rs_suffix = ""

    if len(eval_df) < 1:
        raise ValueError("No data to plot after applying filters.")

    sns.set_context("paper", font_scale=2.0)

    output_dir = args.metrics_file.replace(
        "data/evaluation", "figures/accuracy"
    ).replace(".csv", "")
    makedir(output_dir)

    if args.metrics is None:
        if "Incremental_R2" in set(eval_df["metric"].unique()):
            metrics = ["Incremental_R2"]
        elif "Liability_R2" in set(eval_df["metric"].unique()):
            metrics = ["Liability_R2", "PR_AUC", "ROC_AUC"]
        else:
            metrics = sorted(eval_df["metric"].dropna().unique().tolist())
    elif isinstance(args.metrics, str):
        metrics = [args.metrics]
    else:
        metrics = args.metrics

    for cat in args.category:
        for met in metrics:
            performance_by_category_plots(
                eval_df,
                category=cat,
                metric=met,
                aggregate_single_prs=args.aggregate_single_prs,
                output_file=osp.join(
                    output_dir, f"{cat}_{met}{rs_suffix}{d_suffix}{args.extension}"
                ),
            )
