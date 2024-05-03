import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import assign_models_consistent_colors


def plot_expert_weights_wrt_gate_input(moe_model, dataset, steps=1000):
    """
    Plot the expert weights as a function of the gating inputs (e.g. PCs, age, sex, etc.)
    This is best used for continuous variables (such as age or PCs) in order to get an
    idea about the decision boundaries of the gating model.

    :param dataset: A PhenotypePRSDataset object
    :param moe_model: A trained MoEPRS model
    :param steps: The number of steps to use when interpolating the gating inputs
    """

    covars = dataset.get_covariates()

    mean_vals = np.repeat([covars.mean(axis=0)], steps, axis=0)
    min_vals = covars.min(axis=0)
    max_vals = covars.max(axis=0)

    for i, cov in enumerate(dataset.covariates):

        if len(np.unique(covars[:, i])) > 2:

            mean_vals[:, i] = np.linspace(min_vals[i], max_vals[i], steps)

            probs = moe_model.predict_proba(mean_vals)

            for j in range(probs.shape[1]):
                plt.plot(mean_vals[:, i], probs[:, j], label=dataset.expert_ids[j])

            plt.xlabel(cov)
            plt.title(f"Expert weights as a function of {cov}")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()


def gate_parameters_heatmap(moe_model, dataset, output_file=None):
    """
    Plot the gating parameters as a heatmap. On the X-axis (horizontally), we'll show
    the covariates (input to the gating model) and on the Y-axis (vertically), we'll show
    the experts. The color of each cell will represent the weight or parameter of the linear model.

    :param moe_model: A trained MoEPRS model
    :param dataset: A PhenotypePRSDataset object
    :param output_file: The name of the file to save the plot to. If None, the plot will be shown.
    """

    covars = dataset.covariates

    if moe_model.gate_add_intercept:
        covars = ['Intercept'] + covars

    cg = sns.clustermap(pd.DataFrame(moe_model.gate_params.T,
                                     columns=covars,
                                     index=dataset.expert_ids),
                        figsize=(15, 12),
                        center=0.,
                        dendrogram_ratio=0.2,
                        yticklabels=True, xticklabels=True,
                        cmap='RdBu')
    cg.ax_row_dendrogram.set_visible(False)

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.close()


def plot_expert_distribution(moe_model,
                             dataset,
                             title=None,
                             pgs_order=None,
                             alpha=0.5,
                             use_umap=False,
                             x_covar='PC1',
                             y_covar='PC2'):
    """
    Plot the distribution of the experts in the covariate space. This is useful to get an idea
    about the decision boundaries of the gating model.

    :param moe_model: A trained MoEPRS model
    :param dataset: A PhenotypePRSDataset object
    :param title: The title of the plot
    :param pgs_order: The order of the experts in the plot
    :param alpha: The transparency of the points
    :param use_umap: Whether to use UMAP for visualization (X and Y axes will be UMAP coordinates based
    on the available PCs).
    :param x_covar: The name of the covariate to use for the X-axis
    :param y_covar: The name of the covariate to use for the Y-axis
    """

    dataset.set_backend("numpy")

    covars = dataset.get_covariates()

    if use_umap:
        import umap
        fit = umap.UMAP()
        u = fit.fit_transform(covars[:, [i for i, c in enumerate(dataset.covariates) if 'PC' in c.upper()]])
        x = u[:, 0]
        y = u[:, 1]
    else:
        x = dataset.get_covariate_by_name(x_covar)
        y = dataset.get_covariate_by_name(y_covar)

    unique_labels = dataset.expert_ids

    if pgs_order is None:
        pgs_order = list(unique_labels)

    colors = sns.color_palette('gist_ncar', n_colors=len(unique_labels))

    argmax_proba = moe_model.predict_probe_from_dataset(dataset).argmax(axis=1)

    for i, label in enumerate(unique_labels):
        proba_mask = argmax_proba == i
        if np.sum(proba_mask) > 0:
            plt.scatter(x[proba_mask], y[proba_mask],
                        color=colors[pgs_order.index(label)],
                        label=label,
                        alpha=alpha,
                        marker='.')

    if use_umap:
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
    else:
        plt.xlabel(x_covar)
        plt.ylabel(y_covar)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if title is not None:
        plt.title("PRS Model w/ maximum weight")
    else:
        plt.title(title)

    plt.show()


def plot_expert_weights(pgs_weights_df,
                        agg_col=None,
                        agg_mechanism='mean',
                        agg_order=None,
                        title=None,
                        figsize=(12, 6),
                        color_ascending=False,
                        palette='gist_ncar',
                        output_file=None):

    if agg_col is not None:
        assert agg_mechanism in ['mean', 'sort']

    add_color = []

    def sort_pgs_weights_df(df):
        """
        Sort the individuals in the PGS weights data frame based on the maximum
        weight across the experts.

        TODO: Figure out a better way to sort the samples for smoother figures.

        """

        return df.sort_values(list(df.mean(axis=0).sort_values(
            ascending=color_ascending
        ).index.values))

    if agg_col is not None:

        if agg_mechanism == 'mean':
            pgs_weights_df = pgs_weights_df.groupby(agg_col).mean()
            if agg_order is None:
                pgs_weights_df = sort_pgs_weights_df(pgs_weights_df)
            else:
                pgs_weights_df = pgs_weights_df.loc[agg_order, :]

        else:

            grouped_data = pgs_weights_df.groupby(agg_col)
            sep_size = int(0.002*len(pgs_weights_df))  # Determine the separator size
            final_df = []

            x_ticks = []
            x_tick_labels = []
            cum_group_size = 0
            pgs_cols = [c for c in pgs_weights_df.columns if c != agg_col]

            if agg_order is not None:
                group_order = agg_order
            else:
                group_order = grouped_data.groups

            for g_idx, g in enumerate(group_order):

                sub_df = grouped_data.get_group(g).drop(agg_col, axis=1)
                x_ticks.append(cum_group_size + (len(sub_df) // 2))
                cum_group_size += len(sub_df)
                x_tick_labels.append(g)

                sub_df = sort_pgs_weights_df(sub_df)

                # Add a separator line:
                sub_df['_sep'] = 0.

                if g_idx < len(grouped_data.groups) - 1:
                    sep_row = pd.DataFrame(
                        np.repeat(np.concatenate([np.zeros(len(pgs_cols)), [1.]]).reshape(1, -1),
                                  sep_size,
                                  axis=0),
                        columns=pgs_cols + ['_sep']
                    )
                    final_df.append(pd.concat([sub_df, sep_row]))
                    cum_group_size += sep_size
                else:
                    final_df.append(sub_df)

            pgs_weights_df = pd.concat(final_df)
            add_color = ['#000000']

    else:

        pgs_weights_df = sort_pgs_weights_df(pgs_weights_df)

    model_names = sorted([c for c in pgs_weights_df.columns if c != '_sep'])
    colors = assign_models_consistent_colors(model_names, palette)

    if len(add_color) > 0:
        model_names.append('_sep')
        colors['_sep'] = add_color[0]

    pgs_weights_df.plot(
        kind='bar',
        stacked=True,
        figsize=figsize,
        fontsize='small',
        width=1.0,
        color=colors
    )

    if agg_col is None:
        plt.axis('off')
    elif agg_mechanism == 'sort':
        plt.xticks(x_ticks, x_tick_labels)

    if title is not None:
        plt.title(title)

    plt.ylim((0., 1.))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.close()
