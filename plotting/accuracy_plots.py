import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_performance_metrics(metrics_df,
                             title=None,
                             output_file=None,
                             metric='MSE',
                             colormap='cool',
                             color=None):

    ordered_metrics = metrics_df.groupby('PGS').mean(metric).sort_values(metric, ascending=metric == 'MSE')

    plt.figure(figsize=(15, 8))

    if 'avg_weight' in metrics_df.columns:

        norm = matplotlib.colors.Normalize(vmin=0., vmax=1., clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=colormap)

        ax = sns.barplot(data=metrics_df,
                         x='PGS',
                         y=metric,
                         order=ordered_metrics.index,
                         palette=[mapper.to_rgba(v) for v in ordered_metrics['avg_weight'].values])
    else:

        if color is not None:
            color = [color[m] for m in ordered_metrics.index]

        ax = sns.barplot(data=metrics_df,
                         x='PGS',
                         y=metric,
                         order=ordered_metrics.index,
                         palette=color)

    if f'{metric}_err' in metrics_df.columns:
        ax.errorbar(data=metrics_df, x='PGS', y=metric,
                    yerr=f'{metric}_err', ls='', lw=3, color='black')

    plt.xlabel("PGS")
    plt.ylabel(metric)

    if title is not None:
        plt.title(title)

    plt.xticks(rotation=90)

    if 'avg_weight' in metrics_df.columns:
        cbar = plt.colorbar(mapper)
        cbar.ax.set_ylabel('Average Expert Weight', rotation=270)

    plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.close()


def plot_accuracy_with_fst(metrics_df, metric, reference_cluster='17 ENG-BRI'):
    # Read the cluster interpretation file:
    clust_df = pd.read_csv("metadata/cluster_interpretation.csv")
    ref_id = clust_df.loc[clust_df['Description'] == reference_cluster, "Cluster"]

    # Read the fst file:
    fst_df = pd.read_csv("metadata/weighted_mean_fst_from_plink_LDthinned_20200214_065646.txt",
                         names=['C1', 'C2', 'FST'])
    unique_clusters = set(fst_df[['C1', 'C2']].values.flatten()).intersection(set(clust_df['Cluster'].values))

    fst_df = fst_df.loc[fst_df['C1'].isin(unique_clusters) & fst_df['C2'].isin(unique_clusters),]

    fst_mat = np.zeros((len(unique_clusters), len(unique_clusters)))
    fst_mat[fst_df['C1'].values, fst_df['C2'].values] = fst_df['FST'].values

    fst_mat += fst_mat.T

    fst_df = pd.DataFrame({'Cluster': clust_df['Description'], 'Fst': fst_mat[ref_id, :].flatten()})

    mdf = metrics_df.merge(fst_df, left_on='Evaluation Group', right_on='Cluster')

    sns.lmplot(data=mdf, x="Fst", y=metric, hue="PGS", ci=None)


def grouped_plot(metrics_df,
                 keep_models=None,
                 hue_order=None,
                 order=None,
                 col=None,
                 title=None,
                 output_file=None,
                 metric="Incremental_R2",
                 kind="box",
                 palette=None):

    if keep_models is not None:
        metrics_df = metrics_df.loc[metrics_df.PGS.isin(keep_models)]

    if hue_order is None:
        sorted_pgs = metrics_df.groupby('Model Name')[metric].mean().sort_values()
        hue_order = sorted_pgs.index

    plt.figure(figsize=(15, 7))

    if col is None:

        if kind == "box":
            ax = sns.boxplot(data=metrics_df,
                             x="Evaluation Group",
                             y=metric,
                             hue="Model Name",
                             order=order,
                             hue_order=hue_order,
                             showfliers=False,
                             palette=palette)
        else:
            ax = sns.barplot(data=metrics_df,
                             x="Evaluation Group",
                             y=metric,
                             hue="Model Name",
                             order=order,
                             hue_order=hue_order,
                             palette=palette)

        if len(metrics_df['Evaluation Group'].unique()) > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        if f'{metric}_err' in metrics_df.columns:

            # Modified from:
            # https://stackoverflow.com/questions/66895284/seaborn-barplot-with-specified-confidence-intervals
            num_hues = len(hue_order)
            dodge_dists = np.linspace(-0.4, 0.4, 2 * num_hues + 1)[1::2]
            ordered_groups = [l.get_text() for l in ax.get_xticklabels()]

            # Are there better ways to do the same thing?
            for i, hue in enumerate(hue_order):
                dodge_dist = dodge_dists[i]
                df_hue = metrics_df.loc[metrics_df['Model Name'] == hue].copy()

                df_hue['ordered_group'] = df_hue["Evaluation Group"].map(
                    dict(zip(ordered_groups, range(len(ordered_groups))))
                )
                df_hue = df_hue.sort_values('ordered_group')
                bars = ax.errorbar(data=df_hue, x='Evaluation Group', y=metric,
                                   yerr=f'{metric}_err', ls='', lw=[.75, 2][len(ordered_groups) < 5], color='black')
                xys = bars.lines[0].get_xydata()
                bars.remove()
                ax.errorbar(data=df_hue, x=xys[:, 0] + dodge_dist, y=metric,
                            yerr=f'{metric}_err', ls='', lw=[.75, 2][len(ordered_groups) < 5], color='black')

    else:

        ax = sns.catplot(data=metrics_df,
                         kind=kind,
                         col=col,
                         x="Evaluation Group",
                         y=metric,
                         hue="Model Name",
                         hue_order=hue_order,
                         showfliers=False,
                         palette=palette)

    ax.set_xlabel("Evaluation Group")

    try:
        y_label = {"CORR": "Pearson Correlation",
                   "Incremental_R2": "Incremental R-Squared",
                   "MSE": "MSE",
                   "Partial_CORR": "Partial Correlation",
                   "Liability_R2": "Liability R-Squared",
                   "Nagelkerke_R2": "Nagelkerke R-Squared"}[metric]
    except KeyError:
        y_label = metric

    ax.set_ylabel(y_label)

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.close()

