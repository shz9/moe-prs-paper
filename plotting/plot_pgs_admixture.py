import os.path as osp
import sys
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
from magenpy.utils.system_utils import makedir
from gate_interpretation import plot_expert_weights
from PRSDataset import PRSDataset
from moe import MoEPRS
from plot_utils import sort_groups, GROUP_MAP, MODEL_NAME_MAP
import seaborn as sns
import pandas as pd
import numpy as np
import argparse


def plot_admixture_graphs(prs_dataset,
                          model,
                          title=None,
                          output_file=None,
                          group_col=None,
                          min_group_size=50,
                          subsample_within_groups=False,
                          agg_mechanism='mean',
                          palette='gist_ncar'):
    """
    Plot the admixture graph for a given model and dataset.
    This function takes a PRSDataset object and a trained MoE model and figures the admixture graph
    (gating model weights) for individuals or groups in the dataset.

    :param prs_dataset: A PRSDataset object
    :param model: A trained MoE model
    :param title: The title of the plot
    :param output_file: The path to save the plot to (if not given, the plot will be displayed)
    :param group_col: The column to use for stratification / grouping samples.
    :param min_group_size: The minimum size of the groups to include in the plot.
    :param subsample_within_groups: Whether to subsample the larger groups to generate more even looking figures.
    :param agg_mechanism: The mechanism to use for aggregation. Can be 'mean' or 'sort'.
    If 'mean', the mean gating weights for each group will be plotted. If 'sort', the gating weights
    for each individual will be shown within their respective groups, sorted by the mean gating weight.
    """

    assert agg_mechanism in ['mean', 'sort'], "Aggregation mechanism must be either 'mean' or 'sort'."

    prs_dataset.set_backend("numpy")

    try:
        proba = np.array(model.predict_proba(prs_dataset))
    except TypeError:
        prs_dataset.set_backend("torch")
        proba = model.predict_proba(prs_dataset).detach().numpy()
        prs_dataset.set_backend("numpy")

    # Map the PRS IDs:
    mapped_prs_ids = []
    for prs_id in model.expert_cols:
        try:
            mapped_prs_ids.append(MODEL_NAME_MAP[prs_id])
        except KeyError:
            mapped_prs_ids.append(prs_id)

    proba = pd.DataFrame(proba, columns=mapped_prs_ids)

    if group_col is not None:

        proba[group_col] = prs_dataset.get_data_columns(group_col).flatten()

        # Filter tiny groups:
        if min_group_size is not None and min_group_size > 0:
            # Get the counts of each group:
            group_counts = proba[group_col].value_counts()

            # Remove groups with less than min_group_size samples:
            group_counts = group_counts[group_counts >= min_group_size]

            # Filter the dataframe:
            proba = proba[proba[group_col].isin(group_counts.index)]

        # Map the group names:
        if group_col in ('Ancestry', 'Sex'):
            proba[group_col] = proba[group_col].astype(str).map(GROUP_MAP)

        if group_col in ('Ancestry', 'UMAP_Cluster'):
            sorted_groups = sort_groups(proba[group_col].unique())
        else:
            sorted_groups = None

        if subsample_within_groups:
            # Determine the median size of the groups:
            median_group_size = int(np.median(proba.groupby(group_col).size()))

            # Define a function to sub-sample only if the group size is
            # more than twice larger than the median:

            def cond_subsample_func(x):
                if len(x) > 2 * median_group_size:
                    return x.sample(2 * median_group_size)
                else:
                    return x

            # Conditionally subsample within each group:
            proba = proba.groupby(group_col).apply(cond_subsample_func).reset_index(drop=True)

        # Adjust the figure size:
        if agg_mechanism == 'sort':
            figsize = (25, 5)
        else:
            figsize = (12, 6)

        plot_expert_weights(proba,
                            agg_col=group_col,
                            agg_mechanism=agg_mechanism,
                            agg_order=sorted_groups,
                            figsize=figsize,
                            title=title,
                            palette=palette,
                            output_file=output_file)
    else:
        plot_expert_weights(proba,
                            title=title,
                            palette=palette,
                            output_file=output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot the admixture graph for a given model and dataset.'
    )

    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='The path to the trained model.')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='The path to the dataset.')
    parser.add_argument('--group-col', dest='group_col', type=str, nargs='+', default=None,
                        help='The column to use for stratification.')
    parser.add_argument('--agg-mechanism', dest='agg_mechanism', type=str, default='sort',
                        choices={'mean', 'sort'},
                        help='The mechanism to use for aggregation.')
    parser.add_argument('--extension', dest='extension', type=str, default='.png',
                        help='The file extension to use for saving the plot.')
    parser.add_argument('--subsample', dest='subsample', action='store_true', default=False,
                        help='Subsample the larger groups to generate more even looking figures.')

    args = parser.parse_args()

    sns.set_context("paper", font_scale=2)

    p_dataset = PRSDataset.from_pickle(args.dataset)
    moe_model = MoEPRS.from_saved_model(args.model)

    data_path = args.dataset.replace('data/harmonized_data', 'figures/admixture_graphs').replace('.pkl', '')
    model_path = '_'.join(args.model.replace('.pkl', '').split('/')[-3:])

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
                output_file=osp.join(data_path, model_path + f'_{gcol}{args.extension}'),
                agg_mechanism=args.agg_mechanism,
                subsample_within_groups=args.subsample
            )
