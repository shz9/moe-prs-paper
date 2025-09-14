import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(parent_dir, "evaluation/"))
from magenpy.utils.system_utils import makedir
import pandas as pd
from accuracy_plots import grouped_plot
from plot_utils import sort_groups, read_eval_metrics, transform_eval_metrics
import seaborn as sns
import argparse


def postprocess_metrics_df(metrics_df,
                           metric,
                           category='Ancestry',
                           min_sample_size=100,
                           aggregate_single_prs=True,
                           include_cohort_matched=True):

    relevant_cols = ['PGS', 'Model Category', 'Model Name', 'Training dataset', 'Training cohort', 'Test biobank',
                     'Phenotype', 'EvalCategory', 'Evaluation Group', 'N', metric]

    if f'{metric}_err' in metrics_df.columns:
        relevant_cols.append(f'{metric}_err')

    sub_metrics_df = metrics_df[relevant_cols]

    # Filter the metrics dataframe:
    sub_metrics_df = sub_metrics_df.loc[sub_metrics_df.EvalCategory.isin([category, 'All'])]
    sub_metrics_df = sub_metrics_df.loc[sub_metrics_df.N >= min_sample_size]

    if 'SinglePRS+Covariates' in sub_metrics_df['Model Category'].unique():
        single_model_label = 'SinglePRS+Covariates'
    else:
        single_model_label = 'SinglePRS'

    if metric in ('PR_AUC', 'ROC_AUC', 'MSE', 'CORR'):
        model_cats = ['MoE', 'MultiPRS', 'AncestryWeightedPRS', 'Covariates', single_model_label]
    else:
        model_cats = ['MoE', 'MultiPRS', 'AncestryWeightedPRS', single_model_label]

    sub_metrics_df = sub_metrics_df.loc[sub_metrics_df['Model Category'].isin(model_cats)]
    #sub_metrics_df['Model Name'] = sub_metrics_df['Model Name'].str.replace('-covariates', '')

    if include_cohort_matched:

        mask = ((sub_metrics_df['Model Category'] == single_model_label) &
                (sub_metrics_df['Evaluation Group'] == sub_metrics_df['Training cohort']))

        matched_df = sub_metrics_df.loc[mask].copy()
        matched_df['Model Name'] = f'{category}-matched PRS'

        sub_metrics_df = pd.concat([matched_df.reset_index(drop=True),
                                    sub_metrics_df.reset_index(drop=True)],
                                   ignore_index=True)

    if aggregate_single_prs:

        # Get entries for SinglePRS methods:
        mask = ((sub_metrics_df['Model Category'] == single_model_label) &
                (sub_metrics_df['Model Name'] != f'{category}-matched PRS'))

        grouped = sub_metrics_df.loc[mask].groupby('Evaluation Group')
        if metric == 'MSE':
            single_prs_agg = grouped.apply(lambda x: x.loc[x[metric].idxmin()])
        else:
            single_prs_agg = grouped.apply(lambda x: x.loc[x[metric].idxmax()])

        single_prs_agg = single_prs_agg.reset_index(drop=True)
        single_prs_agg['Model Name'] = 'Best Single Source PRS'

        sub_metrics_df = pd.concat([single_prs_agg.reset_index(drop=True),
                                    sub_metrics_df.loc[~mask].reset_index(drop=True)],
                                   ignore_index=True)


    return sub_metrics_df


def generate_model_colors(metrics_df,
                          metric,
                          palette='Set2',
                          n_model_types=4):

    palette = sns.color_palette(palette, n_model_types)

    if 'SinglePRS+Covariates' in metrics_df['Model Category'].unique():
        single_model_label = 'SinglePRS+Covariates'
    else:
        single_model_label = 'SinglePRS'

    # ---------------------------------------------------------------------
    # Determine the hue order:
    single_hue_order = metrics_df.loc[metrics_df['Model Category'] == single_model_label].groupby(
        'Model Name'
    )[metric].mean().sort_values(ascending=metric == 'MSE').index

    single_model_colors = dict(zip(single_hue_order,
                                   sns.light_palette(palette[0], max(len(single_hue_order), 5),
                                                     reverse=True)))

    multiprs_hue_order = metrics_df.loc[metrics_df['Model Category'] == 'MultiPRS'].groupby(
        'Model Name'
    )[metric].mean().sort_values(ascending=metric == 'MSE').index

    multiprs_hue_colors = dict(zip(multiprs_hue_order,
                                   sns.light_palette(palette[1], max(len(multiprs_hue_order), 5),
                                                     reverse=True)))

    awm_hue_order = metrics_df.loc[metrics_df['Model Category'] == 'AncestryWeightedPRS'].groupby(
        'Model Name'
    )[metric].mean().sort_values(ascending=metric == 'MSE').index

    awm_hue_colors = dict(zip(awm_hue_order,
                              sns.light_palette(palette[2], max(len(awm_hue_order), 5),
                                                reverse=True)))

    moe_hue_order = metrics_df.loc[metrics_df['Model Category'] == 'MoE'].groupby(
        'Model Name'
    )[metric].mean().sort_values(ascending=metric == 'MSE').index

    moe_hue_colors = dict(zip(moe_hue_order,
                              sns.light_palette(palette[3], max(len(moe_hue_order), 5),
                                                reverse=True)))

    hue_order = list(moe_hue_order) + list(multiprs_hue_order) + list(awm_hue_order) + list(single_hue_order)
    colors = {**moe_hue_colors, **multiprs_hue_colors, **awm_hue_colors, **single_model_colors}

    return colors, hue_order


def performance_by_category_plots(metrics_df,
                                  output_file=None,
                                  category='Ancestry',
                                  metric='Incremental_R2',
                                  min_sample_size=100,
                                  aggregate_single_prs=True):

    sub_metrics_df = postprocess_metrics_df(metrics_df, metric=metric, category=category,
                                            min_sample_size=min_sample_size,
                                            aggregate_single_prs=aggregate_single_prs)

    colors, hue_order = generate_model_colors(sub_metrics_df, metric=metric)

    phenotype = sub_metrics_df['Phenotype'].iloc[0]

    # ---------------------------------------------------------------------

    grouped_plot(sub_metrics_df,
                 kind="bar",
                 order=sort_groups(sub_metrics_df['Evaluation Group'].unique()),
                 palette=colors,
                 hue_order=hue_order,
                 title=f'Prediction accuracy for {phenotype}',
                 output_file=output_file,
                 metric=metric)

    return sub_metrics_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot predictive performance of PRS models by category."
    )

    parser.add_argument("--metrics-file", dest="metrics_file", type=str, required=True,
                        help="The path to the metrics file.")
    parser.add_argument("--category", dest="category", type=str, default=['Ancestry'], nargs='+',
                        help="The category (or list of categories) to plot the predictive performance for.")
    parser.add_argument("--metrics", dest="metrics", type=str, default=None, nargs='+',
                        help="The performance metric to plot.")
    parser.add_argument('--aggregate-single-prs', dest='aggregate_single_prs', action='store_true',
                        default=False,
                        help="Aggregate the results for SinglePRS models (select best for each category).")
    parser.add_argument('--restrict-to-same-biobank', dest='restrict_to_same_biobank', action='store_true',
                        default=False,
                        help="Restrict the analysis to models trained and tested on the same biobank.")
    parser.add_argument('--train-dataset', dest='train_dataset', type=str, default=None,
                        help='If specified, then use models trained on this dataset only.')
    parser.add_argument('--extension', dest='extension', type=str, default='.png',
                        help='The file extension to use for saving the plot.')

    args = parser.parse_args()

    print("> Plotting predictive performance for the following evaluation metrics:\n", args.metrics_file)

    eval_df = transform_eval_metrics(read_eval_metrics(args.metrics_file))

    if args.train_dataset is not None:
        eval_df = eval_df.loc[eval_df['Training dataset'] == args.train_dataset]
        d_suffix = f'_{args.train_dataset.replace("/", "_")}'
    else:
        d_suffix = ''

    if args.restrict_to_same_biobank:
        eval_df = eval_df.loc[eval_df['Training biobank'] == eval_df['Test biobank']]
        rs_suffix = '_rs'
    else:
        rs_suffix = ''

    if len(eval_df) < 1:
        raise ValueError("No data to plot after applying filters.")

    sns.set_context("paper", font_scale=2.)

    output_dir = args.metrics_file.replace('data/evaluation', 'figures/accuracy').replace('.csv', '')
    makedir(output_dir)

    if args.metrics is None:
        if 'Incremental_R2' in eval_df.columns:
            metrics = ['Incremental_R2']
        elif 'Liability_R2' in eval_df.columns:
            metrics = ['Liability_R2', 'PR_AUC', 'ROC_AUC']
        else:
            raise Exception("Could not automatically determine the performance metric to plot. "
                            "Please specify it explicitly with the --metrics option.")
    elif isinstance(args.metrics, str):
        metrics = [args.metrics]
    else:
        metrics = args.metrics

    for cat in args.category:
        for met in metrics:
            performance_by_category_plots(eval_df,
                                          category=cat,
                                          metric=met,
                                          aggregate_single_prs=args.aggregate_single_prs,
                                          output_file=osp.join(output_dir,
                                                              f'{cat}_{met}{rs_suffix}{d_suffix}{args.extension}'))


