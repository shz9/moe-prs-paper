import argparse
import os.path as osp
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))


from gate_interpretation import plot_expert_weights
from eval_utils import DEFAULT_MIN_GROUP_SIZE
from moe import MoEPRS
from moe_pytorch import TorchMoEPRS
from plot_utils import SEX_LABEL_MAP, MODEL_NAME_MAP, sort_groups
from PRSDataset import PRSDataset


def plot_admixture_graphs(
    prs_dataset,
    model,
    title=None,
    output_file=None,
    group_col=None,
    sort_col=None,
    sorted_groups=None,
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    max_group_size=10_000,
    subsample=False,
    agg_mechanism="mean",
    figsize="auto",
    palette="Set3",
    drop_legend=False,
    tick_rotation=90,
):
    assert agg_mechanism in ["mean", "sort"], (
        "Aggregation mechanism must be either 'mean' or 'sort'."
    )

    prs_dataset.set_backend("numpy")

    analysis_id = prs_dataset.analysis_id

    proba = np.asarray(model.predict_proba(prs_dataset))

    # Map the PRS IDs:
    mapped_prs_ids = []
    for prs_id in model.expert_cols:
        mapped_prs_ids.append(MODEL_NAME_MAP[analysis_id].get(prs_id, prs_id))

    proba = pd.DataFrame(proba, columns=mapped_prs_ids)

    # If the user requests, add a sorting column:
    if sort_col is not None:
        proba[sort_col] = prs_dataset.get_data_columns(sort_col).flatten()

    if group_col is not None:
        proba[group_col] = prs_dataset.get_data_columns(group_col).flatten()

        # Filter tiny groups:
        if min_group_size is not None and min_group_size > 0:
            group_counts = proba[group_col].value_counts()
            group_counts = group_counts[group_counts >= min_group_size]
            proba = proba[proba[group_col].isin(group_counts.index)]

        # Map the group names:
        if group_col == "Sex":
            proba[group_col] = (
                proba[group_col]
                .astype(int)
                .astype(str)
                .map(SEX_LABEL_MAP)
                .fillna(proba[group_col])
            )

        if sorted_groups is None and group_col in ("Ancestry", "UMAP_Cluster"):
            sorted_groups = sort_groups(proba[group_col].unique())

        if subsample:
            max_group_size = 2 * min(
                int(np.median(proba.groupby(group_col).size())), max_group_size // 2
            )

            def cond_subsample_func(x):
                if len(x) > max_group_size:
                    return x.sample(max_group_size)
                else:
                    return x

            proba = (
                proba.groupby(group_col)
                .apply(cond_subsample_func)
                .reset_index(drop=True)
            )

        if figsize == "auto":
            if agg_mechanism == "sort" and sorted_groups is not None:
                figsize = (25, 5)
            else:
                figsize = (12, 6)

        return plot_expert_weights(
            proba,
            agg_col=group_col,
            agg_mechanism=agg_mechanism,
            agg_order=sorted_groups,
            sort_col=sort_col,
            figsize=figsize,
            title=title,
            palette=palette,
            output_file=output_file,
            drop_legend=drop_legend,
            tick_rotation=tick_rotation,
        )
    else:
        if subsample and proba.shape[0] > max_group_size:
            proba = proba.sample(max_group_size)

        return plot_expert_weights(
            proba,
            sort_col=sort_col,
            title=title,
            palette=palette,
            agg_order=sorted_groups,
            output_file=output_file,
            drop_legend=drop_legend,
            tick_rotation=tick_rotation,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the admixture graph (gate probabilities) for a given model and dataset."
    )

    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="Path to trained model.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
        help="Path to harmonized PRSDataset.",
    )
    parser.add_argument(
        "--group-col",
        dest="group_col",
        type=str,
        nargs="+",
        default=None,
        help="Column(s) to stratify by (e.g., Ancestry, Sex, UMAP_Cluster).",
    )
    parser.add_argument(
        "--agg-mechanism",
        dest="agg_mechanism",
        type=str,
        default="sort",
        choices={"mean", "sort"},
        help="Aggregation mechanism: mean (group-average) or sort (individual bars).",
    )
    parser.add_argument(
        "--extension",
        dest="extension",
        type=str,
        default=".png",
        help="File extension for plots.",
    )
    parser.add_argument(
        "--subsample",
        dest="subsample",
        action="store_true",
        default=False,
        help="Subsample within large groups for cleaner sort plots.",
    )

    args = parser.parse_args()

    sns.set_context("paper", font_scale=2)

    p_dataset = PRSDataset.from_pickle(args.dataset)

    analysis_id = args.dataset.split("/")[2]

    try:
        moe_model = TorchMoEPRS.from_saved_model(args.model_path)
    except Exception as e:
        moe_model = MoEPRS.from_saved_model(args.model_path)

    # mirror your previous output folder logic
    data_path = args.dataset.replace(
        "data/harmonized_data", "figures/admixture_graphs"
    ).replace(".pkl", "")
    model_path = "_".join(
        args.model_path.replace(".pkl", "").split("/")[-3:]
    )

    makedir(data_path)

    if args.group_col is None:
        plot_output_file = osp.join(data_path, model_path + args.extension)
        plot_admixture_graphs(p_dataset, moe_model, output_file=plot_output_file)
    else:
        for gcol in args.group_col:
            plot_admixture_graphs(
                p_dataset,
                moe_model,
                group_col=gcol,
                output_file=osp.join(
                    data_path, model_path + f"_{gcol}{args.extension}"
                ),
                agg_mechanism=args.agg_mechanism,
                subsample=args.subsample,
            )
