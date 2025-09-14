import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import assign_models_consistent_colors
from plot_utils import sort_groups, GROUP_MAP, MODEL_NAME_MAP


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


def gate_parameters_heatmap(moe_model,
                            figsize=(6, 6),
                            annot=False,
                            title=None,
                            output_file=None):
    """
    Plot the gating parameters as a heatmap. On the X-axis (horizontally), we'll show
    the covariates (input to the gating model) and on the Y-axis (vertically), we'll show
    the experts. The color of each cell will represent the weight or parameter of the linear model.

    :param moe_model: A trained MoEPRS model
    :param output_file: The name of the file to save the plot to. If None, the plot will be shown.
    """

    gate_params = moe_model.get_model_parameters()['gate_params']
    # Add a zero column for the missing expert:
    gate_params[[c for c in moe_model.expert_cols if c not in gate_params.columns][0]] = 0.
    gate_params.columns = [MODEL_NAME_MAP.get(c, c) + ['', '(*)'][i == gate_params.shape[1] - 1]
                           for i, c in enumerate(gate_params.columns)]

    # Sort the columns by the PC order:
    pc_elm = [c for c in gate_params.index if 'PC' in c.upper()]
    pc_elm = sorted(pc_elm, key=lambda x: int(x.replace('PC', '')))
    non_pc = [c for c in gate_params.index if c not in pc_elm]

    gate_params = gate_params.loc[non_pc + pc_elm, :]

    cg = sns.clustermap(gate_params,
                        figsize=figsize,
                        center=0.,
                        row_cluster=False,
                        col_cluster=False,
                        yticklabels=True,
                        xticklabels=True,
                        annot=annot,
                        fmt=".2f",
                        cmap='RdBu')

    # Turn off the dendrograms:
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    cbar_ax = cg.ax_cbar

    # Move the colorbar to the left
    cbar_ax.set_position([0.05, 0.2, 0.02, 0.45])  # [left, bottom, width, height] in figure coordinates

    if title is not None:
        cg.fig.suptitle(title, y=.85)

    cg.ax_heatmap.set_xlabel("Stratified PGS (Experts)")
    cg.ax_heatmap.set_ylabel("Gating Model Input Features")

    #plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()


def plot_expert_distribution(moe_model,
                             dataset,
                             mask=None,
                             title=None,
                             alpha=0.5,
                             use_umap=False,
                             palette='gist_ncar',
                             x_covar='PC1',
                             y_covar='PC2'):
    """
    Plot the distribution of the experts in the covariate space. This is useful to get an idea
    about the decision boundaries of the gating model.

    :param moe_model: A trained MoEPRS model
    :param dataset: A PhenotypePRSDataset object
    :param title: The title of the plot
    :param alpha: The transparency of the points
    :param use_umap: Whether to use UMAP for visualization (X and Y axes will be UMAP coordinates based
    on the available PCs).
    :param x_covar: The name of the covariate to use for the X-axis
    :param y_covar: The name of the covariate to use for the Y-axis
    """

    dataset.set_backend("numpy")

    covars = dataset.get_covariates()

    if mask is None:
        mask = np.arange(covars.shape[0])

    if use_umap:
        import umap
        fit = umap.UMAP()
        u = fit.fit_transform(covars[:, [i for i, c in enumerate(dataset.covariates) if 'PC' in c.upper()]])
        x = u[:, 0]
        y = u[:, 1]
    else:
        x, y = dataset.get_data_columns([x_covar, y_covar]).T

    x = x[mask]
    y = y[mask]

    model_names = [MODEL_NAME_MAP[label] for label in dataset.prs_cols]

    colors = assign_models_consistent_colors(model_names, palette)

    argmax_proba = moe_model.predict_proba(dataset).argmax(axis=1)[mask]

    for i, label in enumerate(model_names):
        proba_mask = argmax_proba == i
        if np.sum(proba_mask) > 0:
            plt.scatter(x[proba_mask], y[proba_mask],
                        color=colors[label],
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


def sort_admixture_by_dominant(q_mat: np.ndarray) -> np.ndarray:
    """
    Sorts an admixture Q-matrix for improved visual appearance.

    The sorting is performed in two steps:
    1. Individuals are primarily grouped by their dominant ancestral component.
    2. Within each dominant ancestral group, individuals are sorted
       by the proportion of that dominant component in descending order
       (i.e., individuals with a higher proportion of their dominant ancestry
       come first within that group).

    Args:
        q_mat (np.ndarray): A 2D NumPy array (matrix) where rows represent
                            individuals and columns represent ancestral
                            proportions (Q-matrix from ADMIXTURE/STRUCTURE).
                            Values should sum to approximately 1 across rows.

    Returns:
        np.ndarray: A new Q-matrix with rows (individuals) sorted
                    according to the described criteria.
    """
    if not isinstance(q_mat, np.ndarray) or q_mat.ndim != 2:
        raise ValueError("Input q_mat must be a 2D NumPy array.")

    # Step 1: Identify the dominant ancestry component for each individual
    dominant_component_indices = np.argmax(q_mat, axis=1)

    # Step 2: Get the proportion of the dominant ancestry for each individual
    # We use advanced indexing here: q_mat[row_indices, col_indices]
    dominant_component_proportions = q_mat[np.arange(q_mat.shape[0]), dominant_component_indices]

    # Step 3: Create a compound sorting key
    # We want to sort first by 'dominant_component_indices' (ascending)
    # and then by 'dominant_component_proportions' (descending)
    # To sort descending using np.argsort, we can negate the values.
    # We'll use np.lexsort, which sorts by multiple keys. The last key in the
    # tuple is the primary sort key, and earlier keys are secondary, tertiary, etc.
    # So, (dominant_component_proportions_negated, dominant_component_indices)
    # means sort primarily by dominant_component_indices, then by
    # dominant_component_proportions_negated (which effectively means
    # dominant_component_proportions in descending order).

    sort_indices = np.lexsort((
        -dominant_component_proportions,  # Secondary sort: dominant proportion (descending)
        dominant_component_indices        # Primary sort: dominant component index (ascending)
    ))

    # Apply the sorting to the original q_mat
    sorted_q_mat = q_mat[sort_indices]

    return sorted_q_mat, sort_indices


def plot_expert_weights(pgs_weights_df,
                        agg_col=None,
                        agg_mechanism='mean',
                        agg_order=None,
                        title=None,
                        figsize=(12, 6),
                        color_ascending=False,
                        drop_legend=False,
                        palette='gist_ncar',
                        tick_rotation=90,
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

        #sorted_mat, sorted_indices = sort_admixture_by_dominant(df.values)

        #return pd.DataFrame(sorted_mat, columns=df.columns, index=sorted_indices)


    sorted_idx = []

    if agg_col is not None:

        if agg_mechanism == 'mean':
            pgs_weights_df = pgs_weights_df.groupby(agg_col).mean()
            print(pgs_weights_df)
            if agg_order is None:
                pgs_weights_df = sort_pgs_weights_df(pgs_weights_df)
            else:
                pgs_weights_df = pgs_weights_df.loc[agg_order, :]

        else:

            grouped_data = pgs_weights_df.groupby(agg_col)
            sep_size = max(1, int(0.002*len(pgs_weights_df)))  # Determine the separator size
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
                sorted_idx.append(sub_df.index.values)

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
            sorted_idx = np.concatenate(sorted_idx)
            add_color = ['#000000']

    else:

        pgs_weights_df = sort_pgs_weights_df(pgs_weights_df)
        sorted_idx = pgs_weights_df.index.values

    model_names = sorted([c for c in pgs_weights_df.columns if c != '_sep'])
    colors = assign_models_consistent_colors(model_names, palette)

    if len(add_color) > 0:
        model_names.append('_sep')
        colors['_sep'] = add_color[0]

    ax = pgs_weights_df.plot(
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
        ax.set_xticks(x_ticks, x_tick_labels, rotation=tick_rotation)

    if title is not None:
        plt.title(title)

    if agg_col is not None:
        plt.xlabel(agg_col)
    else:
        plt.xlabel("Samples")

    plt.ylabel("Mixing Proportion")
    plt.ylim((0., 1.))

    if drop_legend:
        plt.legend().remove()
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Stratified PGS")

    plt.tight_layout()

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=400)
        plt.close()

    if agg_mechanism != 'mean':
        return sorted_idx
