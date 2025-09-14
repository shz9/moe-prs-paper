import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import glob
from magenpy.utils.system_utils import makedir
from plot_utils import sort_groups, read_eval_metrics, transform_eval_metrics
from plot_predictive_performance import postprocess_metrics_df, generate_model_colors


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def add_error_bars(plot_obj, data, x, y, yerr=None, hue=None, hue_order=None,
                   error_kw=None, order=None):
    """
    Add error bars to a seaborn catplot (FacetGrid) or barplot (Axes) with hue grouping.

    Parameters:
    -----------
    plot_obj : sns.FacetGrid or matplotlib.axes.Axes
        The FacetGrid object returned by sns.catplot or the Axes object returned by sns.barplot.
    data : DataFrame
        The dataframe containing the data
    x : str
        Column name for x-axis grouping
    y : str
        Column name for y-axis values
    yerr : str, optional
        Column name for error values (defaults to f'{y}_err')
    hue : str, optional
        Column name for hue grouping
    hue_order : list, optional
        Order of hue categories
    error_kw : dict, optional
        Additional keyword arguments for errorbar formatting
    order : list, optional
        Order of x-axis categories (should match the order used in catplot/barplot)
    """

    # Default error bar styling
    default_error_kw = {'ls': '', 'color': 'black', 'capsize': 0, 'capthick': 0}
    if error_kw:
        default_error_kw.update(error_kw)

    if yerr is None:
        yerr = f'{y}_err'

    is_catplot = isinstance(plot_obj, sns.FacetGrid)

    if is_catplot:
        axes = plot_obj.axes_dict
        # Extract col and row information for FacetGrid
        col_name = plot_obj.col_names[0] if plot_obj.col_names else None
        row_name = plot_obj.row_names[0] if plot_obj.row_names else None
        # Get hue info from FacetGrid if available
        if hue is None and plot_obj.hue_vars:
            hue = plot_obj.hue_vars[0]
        if hue_order is None and plot_obj.hue_names:
            hue_order = plot_obj.hue_names
    else:
        axes = {None: plot_obj}  # Treat single Axes like a dict for iteration
        col_name = None
        row_name = None
        # For a single barplot, we need to infer hue and hue_order
        if hue is None: # Try to infer hue if not explicitly provided
            # This is a bit tricky for bare Axes. A common way is to check the legend or how bars are grouped.
            # However, direct inspection of the seaborn barplot internal structure is more reliable if available.
            # A simpler, more robust approach is to *require* `hue` to be passed for barplots if it's used.
            # But let's try to be clever if `hue` is omitted but present in the data for grouping.
            # If a barplot was created with hue, its legend handles often provide the hue order.
            # This is a heuristic.
            if plot_obj.legend_:
                legend_labels = [text.get_text() for text in plot_obj.legend_.get_texts()]
                # If the legend labels match distinct values in any data column, that could be our hue.
                for col in data.columns:
                    if set(legend_labels) == set(data[col].astype(str).drop_duplicates().tolist()):
                        hue = col
                        hue_order = legend_labels
                        break
        elif hue_order is None and hue in data.columns: # If hue is provided but order isn't
            # Try to infer hue order from the order of bars, or default to data order
            # This is complex as it depends on Seaborn's internal bar ordering.
            # A safer default is to use the unique values from the data, which
            # seaborn often sorts alphabetically by default if no order is specified.
            hue_order = data[hue].drop_duplicates().tolist()
            # If there's a legend, prioritize its order
            if plot_obj.legend_:
                legend_labels = [text.get_text() for text in plot_obj.legend_.get_texts()]
                if set(legend_labels) == set(hue_order):
                    hue_order = legend_labels


    # Determine master x-axis category order
    x_labels_master = None
    if order is not None:
        x_labels_master = order
    elif is_catplot:
        for temp_ax in plot_obj.axes.flat:
            temp_labels = [label.get_text() for label in temp_ax.get_xticklabels()]
            if temp_labels and not all(label == '' for label in temp_labels):
                x_labels_master = temp_labels
                break
    else: # For a single barplot
        x_labels_master = [label.get_text() for label in plot_obj.get_xticklabels()]
        if not x_labels_master: # Fallback if labels are not yet set (e.g., for empty plot)
            x_labels_master = data[x].drop_duplicates().tolist()

    if x_labels_master is None:
        x_labels_master = data[x].drop_duplicates().tolist()


    for val, ax in axes.items():
        facet_data = data.copy()
        current_col_val = None
        current_row_val = None

        if is_catplot:
            if isinstance(val, tuple):
                current_row_val, current_col_val = val
            elif col_name is not None:
                current_col_val = val
            else:
                current_row_val = val

            if col_name is not None and current_col_val is not None:
                facet_data = facet_data[facet_data[col_name] == current_col_val]
            if row_name is not None and current_row_val is not None:
                facet_data = facet_data[facet_data[row_name] == current_row_val]

        if facet_data.empty:
            continue

        x_labels = x_labels_master
        x_positions = ax.get_xticks()

        if hue and hue_order:
            num_hues = len(hue_order)
            # Determine dodge width and bar width dynamically from seaborn's internal properties
            # This is the most robust way to align error bars with dodged bars.
            # Iterate through bars to find their positions and widths
            bar_positions_by_hue_x = {} # Stores {hue_val: {x_cat: [bar_centers]}}
            for container in ax.containers:
                # A container holds the bars for a given hue category within an x-group
                for bar in container.patches:
                    x_center = bar.get_x() + bar.get_width() / 2
                    # Need to map this x_center back to a logical hue and x category.
                    # This is tricky without direct access to seaborn's internal mapping.
                    # A more reliable approach is to calculate expected positions
                    # based on seaborn's default dodging logic, and then match.

            # Re-calculating dodge distances based on standard seaborn barplot dodging
            # This assumes seaborn's default dodging logic, which is generally consistent.
            dodge_width = 0.8 # Standard total width for a group of dodged bars
            bar_width = dodge_width / num_hues
            dodge_distances = np.linspace(-dodge_width/2 + bar_width/2,
                                          dodge_width/2 - bar_width/2, num_hues)

            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw['lw'] = lw

            for i, hue_val in enumerate(hue_order):
                hue_data = facet_data[facet_data[hue] == str(hue_val)] # Ensure string comparison for consistency

                if hue_data.empty:
                    continue

                x_to_pos = {cat: pos for pos, cat in enumerate(x_labels)}

                plot_data_for_hue = []
                for x_cat in x_labels:
                    cat_data = hue_data[hue_data[x] == x_cat]
                    if not cat_data.empty:
                        plot_data_for_hue.append({
                            'x_pos': x_to_pos[x_cat] + dodge_distances[i],
                            'y': cat_data[y].iloc[0],
                            'yerr': cat_data[yerr].iloc[0]
                        })

                if plot_data_for_hue:
                    x_pos = [d['x_pos'] for d in plot_data_for_hue]
                    y_vals = [d['y'] for d in plot_data_for_hue]
                    y_errs = [d['yerr'] for d in plot_data_for_hue]

                    ax.errorbar(x_pos, y_vals, yerr=y_errs, **default_error_kw)
        else:

            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw['lw'] = lw

            plot_data_no_hue = []
            for x_cat in x_labels:
                cat_data = facet_data[facet_data[x] == x_cat]
                if not cat_data.empty:
                    plot_data_no_hue.append({
                        'y': cat_data[y].iloc[0],
                        'yerr': cat_data[yerr].iloc[0]
                    })

            if plot_data_no_hue:
                y_vals = [d['y'] for d in plot_data_no_hue]
                y_errs = [d['yerr'] for d in plot_data_no_hue]
                ax.errorbar(x_positions, y_vals, yerr=y_errs, **default_error_kw)


def add_error_bars_to_catplot(grid, data, x, y, yerr=None, hue=None, hue_order=None,
                              col=None, row=None, error_kw=None, order=None):
    """
    Add error bars to a seaborn catplot with facets and hue grouping.

    Parameters:
    -----------
    grid : sns.FacetGrid
        The FacetGrid object returned by sns.catplot
    data : DataFrame
        The dataframe containing the data
    x : str
        Column name for x-axis grouping
    y : str
        Column name for y-axis values
    yerr : str, optional
        Column name for error values (defaults to f'{y}_err')
    hue : str, optional
        Column name for hue grouping
    hue_order : list, optional
        Order of hue categories
    col : str, optional
        Column name for column faceting
    row : str, optional
        Column name for row faceting
    error_kw : dict, optional
        Additional keyword arguments for errorbar formatting
    order : list, optional
        Order of x-axis categories (should match the order used in catplot)
    """

    # Default error bar styling
    default_error_kw = {'ls': '', 'color': 'black', 'capsize': 0, 'capthick': 0}
    if error_kw:
        default_error_kw.update(error_kw)

    if yerr is None:
        yerr = f'{y}_err'

    # Get x-axis category order once - try multiple approaches
    x_labels_master = None
    if order is not None:
        x_labels_master = order
    else:
        # Try to get labels from any subplot that has them visible
        for temp_ax in grid.axes.flat:
            temp_labels = [label.get_text() for label in temp_ax.get_xticklabels()]
            if temp_labels and not all(label == '' for label in temp_labels):
                x_labels_master = temp_labels
                break

        # Fallback: get unique categories from data
        if x_labels_master is None:
            x_labels_master = data[x].drop_duplicates().tolist()

    row_val = None
    col_val = None

    # Iterate through each subplot in the grid
    for val, ax in grid.axes_dict.items():

        # Filter data for this specific facet
        facet_data = data.copy()

        if isinstance(val, tuple):
            row_val, col_val = val
        elif col is not None:
            col_val = val
        else:
            row_val = val

        if col is not None and col_val is not None:
            facet_data = facet_data[facet_data[col] == col_val]
        if row is not None and row_val is not None:
            facet_data = facet_data[facet_data[row] == row_val]

        if facet_data.empty:
            continue

        # Use the master x_labels order for all subplots
        x_labels = x_labels_master
        x_positions = range(len(x_labels))

        if hue and hue_order:
            # Calculate dodge distances for multiple hues
            num_hues = len(hue_order)
            dodge_width = 0.8  # Total width for all bars
            bar_width = dodge_width / num_hues
            dodge_distances = np.linspace(-dodge_width/2 + bar_width/2,
                                          dodge_width/2 - bar_width/2, num_hues)

            # Set line width based on number of groups
            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw['lw'] = lw

            # Add error bars for each hue
            for i, hue_val in enumerate(hue_order):
                hue_data = facet_data[facet_data[hue] == hue_val]

                if hue_data.empty:
                    continue

                # Create mapping from x categories to positions
                x_to_pos = {cat: pos for pos, cat in enumerate(x_labels)}

                # Get data in the correct order
                plot_data = []
                for x_cat in x_labels:
                    cat_data = hue_data[hue_data[x] == x_cat]
                    if not cat_data.empty:
                        plot_data.append({
                            'x_pos': x_to_pos[x_cat] + dodge_distances[i],
                            'y': cat_data[y].iloc[0],
                            'yerr': cat_data[yerr].iloc[0]
                        })

                if plot_data:
                    x_pos = [d['x_pos'] for d in plot_data]
                    y_vals = [d['y'] for d in plot_data]
                    y_errs = [d['yerr'] for d in plot_data]

                    ax.errorbar(x_pos, y_vals, yerr=y_errs, **default_error_kw)
        else:
            # No hue grouping - simpler case
            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw['lw'] = lw

            # Get data in the correct order
            plot_data = []
            for x_cat in x_labels:
                cat_data = facet_data[facet_data[x] == x_cat]
                if not cat_data.empty:
                    plot_data.append({
                        'y': cat_data[y].iloc[0],
                        'yerr': cat_data[yerr].iloc[0]
                    })

            if plot_data:
                y_vals = [d['y'] for d in plot_data]
                y_errs = [d['yerr'] for d in plot_data]
                ax.errorbar(x_positions, y_vals, yerr=y_errs, **default_error_kw)


def plot_combined_accuracy_metrics(metrics_df,
                                   output_f=None,
                                   metric='Incremental_R2',
                                   palette='Set2',
                                   hue_order=None,
                                   col_order=None,
                                   col_wrap=None,
                                   sharey=False,
                                   height=5,
                                   aspect=1):

    if hue_order is None:
        _, hue_order = generate_model_colors(metrics_df, metric)

    # ---------------------------------------------------------------------

    grid = sns.catplot(x='Evaluation Group',
                       y=metric,
                       col='Phenotype',
                       col_wrap=col_wrap,
                       col_order=col_order,
                       order=sort_groups(metrics_df['Evaluation Group'].unique()),
                       hue='Model Name',
                       palette=palette,
                       hue_order=hue_order,
                       kind='bar',
                       height=height,
                       aspect=aspect,
                       sharey=sharey,
                       data=metrics_df)

    if f'{metric}_err' in metrics_df.columns:
        add_error_bars_to_catplot(grid, metrics_df, 'Evaluation Group',
                                  metric, hue='Model Name',
                                  hue_order=hue_order,
                                  col='Phenotype',)

    grid.set_axis_labels(x_var="Evaluation Group", y_var={
        'Incremental_R2': 'Incremental $R^2$',
        'Liability_R2': 'Liability $R^2$',
        'CORR': 'Pearson $R$',
    }[metric])

    # ---------------------------------------------------------------------

    if False and len(metrics_df['Phenotype'].unique()) > 3:
        # Get handles and labels from one of the subplots for the legend
        handles, labels = grid.axes.flat[0].get_legend_handles_labels()

        fig = grid.figure
        fig.legend(handles, labels,
                   bbox_to_anchor=(0.875, 0.25),  # Adjust these coordinates as needed
                   loc='center',
                   frameon=False)

        #grid._legend.remove()

        # Add an 8th subplot for the legend in the bottom-right position
        #legend_ax = grid.figure.add_subplot(2, 4, 8)  # 2x4 grid, position 8
        #legend_ax.legend(handles, labels, loc='center', frameon=False)
        #legend_ax.set_axis_off()  # Hide the axes for the legend subplot

    for ax in grid.axes.flat:
        title = ax.get_title()
        if title.startswith('Phenotype = '):
            ax.set_title(title.replace('Phenotype = ', ''))

    if output_f is None:
        plt.show()
    else:
        plt.savefig(output_f)
        plt.close()

    return grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot predictive performance of PRS models by category."
    )

    parser.add_argument("--biobank", dest="biobank", type=str, required=True,
                        choices={'ukbb', 'cartagene'},
                        help="The name of the biobank to plot the accuracy metrics for.")
    parser.add_argument("--category", dest="category", type=str, default=['Ancestry'], nargs='+',
                        help="The category (or list of categories) to plot the predictive performance for.")
    parser.add_argument('--aggregate-single-prs', dest='aggregate_single_prs', action='store_true',
                        default=False,
                        help="Aggregate the results for SinglePRS models (select best for each category).")
    parser.add_argument('--restrict-to-same-biobank', dest='restrict_to_same_biobank', action='store_true',
                        default=False,
                        help="Restrict the analysis to models trained and tested on the same biobank.")
    parser.add_argument('--dataset', dest='dataset', type=str,
                        choices={'train', 'test'}, default='test',
                        help='The type of dataset to plot predictive performance on.')
    parser.add_argument('--extension', dest='extension', type=str, default='.png',
                        help='The file extension to use for saving the plot.')
    parser.add_argument('--moe-model', dest='moe_model', type=str, default='MoE',
                        help="The name of the MoE model to plot as reference.")

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)

    phenotype_cats = {
        'binary': ['ASTHMA', 'T2D'],
        'continuous': ['HEIGHT', 'BMI', 'FEV1_FVC', 'HDL', 'LDL', 'LOG_TG', 'TC'],
        'sex_biased': ['TST', 'URT', 'CRTN']
    }

    metric = {
        'binary': 'Liability_R2',
        'continuous': 'Incremental_R2',
        'sex_biased': 'Incremental_R2'
    }

    category = {
        'binary': 'Ancestry',
        'continuous': 'Ancestry',
        'sex_biased': 'Sex'
    }

    metrics_dfs = {
        'binary': [],
        'continuous': [],
        'sex_biased': []
    }

    for f in glob.glob(f"data/evaluation/*/{args.biobank}/{args.dataset}_data.csv"):

        pheno = f.split("/")[-3]
        try:
            pheno_cat = [k for k, v in phenotype_cats.items() if pheno in v][0]
        except IndexError:
            continue

        df = transform_eval_metrics(read_eval_metrics(f))

        df = df.loc[(df['Model Category'] != 'MoE') | df['Model Name'].isin([#f'MoE-CFG ({args.biobank})',
                                                                             f'{args.moe_model} ({args.biobank})'])]

        if args.restrict_to_same_biobank:
            df = df.loc[df['Training biobank'] == df['Test biobank']]


        df = postprocess_metrics_df(df,
                                    metric[pheno_cat],
                                    category=category[pheno_cat],
                                    aggregate_single_prs=args.aggregate_single_prs)

        metrics_dfs[pheno_cat].append(df)

    makedir(f"figures/accuracy/{args.biobank}/{args.dataset}/")

    for pheno_cat, dfs in metrics_dfs.items():
        if len(dfs) < 1:
            raise ValueError(f"No data to plot after applying filters for {pheno_cat}.")

        plot_combined_accuracy_metrics(pd.concat(dfs, axis=0).reset_index(drop=True),
                                       f"figures/accuracy/{args.biobank}/{args.dataset}/combined_metrics_{args.moe_model}_{pheno_cat}.eps",
                                       metric=metric[pheno_cat],
                                       col_order=phenotype_cats[pheno_cat],)
