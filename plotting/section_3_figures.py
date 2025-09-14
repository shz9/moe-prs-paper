import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))
from plot_utils import (
    sort_groups, read_eval_metrics, transform_eval_metrics,
    assign_ancestry_consistent_colors, assign_models_consistent_colors, MODEL_NAME_MAP,
    BIOBANK_NAME_MAP_SHORT, GROUP_MAP
)
from plot_predictive_performance import postprocess_metrics_df, generate_model_colors
from plot_pgs_admixture import plot_admixture_graphs
from combined_accuracy_plots import plot_combined_accuracy_metrics, add_error_bars
from plot_stratified_prediction_accuracy import extract_stratified_evaluation_metrics
from magenpy.utils.system_utils import makedir
from PRSDataset import PRSDataset
from moe import MoEPRS
from baseline_models import MultiPRS
from viprs.eval.eval_utils import fit_linear_model
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse



def extract_complementarity_data(pheno, biobank, residualize_prs=True, residualize_pheno=True):

    dat = PRSDataset.from_pickle(f"data/harmonized_data/{pheno}/{biobank}/full_data.pkl")

    preds = dat.get_prs_predictions()
    covars = pd.DataFrame(dat.get_covariates(), columns=dat.covariates_cols)

    if residualize_pheno:

        pheno = fit_linear_model(dat.get_phenotype().flatten(),
                                 covars,
                                 add_intercept=True).resid
    else:
        pheno = dat.get_phenotype().flatten()

    if residualize_prs:

        for i in range(preds.shape[1]):
            preds[:, i] = fit_linear_model(preds[:, i],
                                           covars,
                                           add_intercept=True).resid

    comp_df = pd.DataFrame(preds, columns=[MODEL_NAME_MAP[c] for c in dat.prs_cols])
    comp_df['phenotype'] = pheno
    comp_df[['Sex', 'Age', 'Ancestry']] = dat.get_data_columns(['Sex', 'Age', 'Ancestry'])
    comp_df['SexG'] = dat.data['Sex'].astype(int).astype(str).map(GROUP_MAP)
    comp_df['AgeGroup2'] = np.array(['Age<=55', 'Age>55']).take(dat.get_data_columns("Age").flatten() > 55)
    comp_df['Sex+Age'] = comp_df['SexG'].values + ' (' + comp_df['AgeGroup2'].values + ')'

    return comp_df


def scatter_with_regression_and_corr(data, x, y, hue=None, palette=None):

    if hue is not None:
        if palette is None:
            groups = data[hue].unique()
            palette_dict = dict(zip(groups, sns.color_palette(n_colors=len(groups))))
        else:
            palette_dict = palette
    else:
        palette_dict = None


    # Base scatterplot
    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=palette_dict, alpha=.25)

    # Add grey dashed lines at zero (make sure they are behind the points)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5, zorder=-1)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5, zorder=-1)

    # Unique hue groups
    if hue is None:
        r, _ = pearsonr(data[x], data[y])
        sns.regplot(data=data, x=x, y=y, label=f'(R = {r:.2f})')
        ax.legend()
    else:
        # Store custom legend handles
        custom_handles = []

        for group, color in palette_dict.items():
            subset = data[data[hue] == group]

            # Plot regression line
            sns.regplot(
                data=subset, x=x, y=y, scatter=False, ax=ax,
                color=color, label=None
            )

            # Compute Pearson correlation
            r, _ = pearsonr(subset[x], subset[y])
            label = f"{group} (R = {r:.2f})"

            # Add a custom handle for the legend
            handle = Line2D([0], [0], color=color, lw=2)
            custom_handles.append((handle, label))

        # Build the legend
        handles, labels = zip(*custom_handles)
        ax.legend(handles=handles, labels=labels, title=hue)


def generate_metrics_figures(biobank='ukbb'):

    # -----------------------------------------------------------------
    # Plot the accuracy metrics for LDL-C in EUR:
    ldl_metrics_eur = extract_stratified_evaluation_metrics(
        pheno='LDL',
        biobank=biobank,
        keep_ancestry=['EUR'],
        category=['SexG', 'AgeGroup3']
    )
    ldl_metrics_eur = ldl_metrics_eur.loc[ldl_metrics_eur['PGS'] == 'EUR']
    ldl_metrics_eur = ldl_metrics_eur.loc[ldl_metrics_eur['EvalGroup'] != 'All']
    ldl_metrics_eur.rename(columns={'PGS': 'Stratified PGS'}, inplace=True)
    ldl_metrics_eur = ldl_metrics_eur.reset_index(drop=True)

    plt.figure(figsize=(5, 5))
    g = sns.barplot(ldl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue='Stratified PGS',
                    palette=assign_models_consistent_colors(ldl_metrics_eur['Stratified PGS'].unique()),
                    order=['Female', 'Male', 'Age<50', 'Age 50–60', 'Age>60']
                    )

    add_error_bars(g, ldl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue_order=['EUR'])

    plt.title(f"Prediction accuracy on LDL Cholesterol\nin samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")

    plt.xlabel("Evaluation Group")
    plt.ylabel('Incremental $R^2$')

    plt.tight_layout()
    plt.savefig(f"figures/section_3/{biobank}_ldl_stratified_accuracy.eps")
    plt.close()

    # -----------------------------------------------------------------
    # Plot the accuracy metrics for LDL-C (Adj) in EUR:

    if biobank == 'ukbb':
        ldl_metrics_eur = extract_stratified_evaluation_metrics(
            pheno='LDL_adj',
            biobank=biobank,
            keep_ancestry=['EUR'],
            category=['SexG', 'AgeGroup3']
        )
        ldl_metrics_eur = ldl_metrics_eur.loc[ldl_metrics_eur['PGS'] == 'EUR']
        ldl_metrics_eur = ldl_metrics_eur.loc[ldl_metrics_eur['EvalGroup'] != 'All']
        ldl_metrics_eur.rename(columns={'PGS': 'Stratified PGS'}, inplace=True)
        ldl_metrics_eur = ldl_metrics_eur.reset_index(drop=True)

        plt.figure(figsize=(5, 5))
        g = sns.barplot(ldl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue='Stratified PGS',
                        palette=assign_models_consistent_colors(ldl_metrics_eur['Stratified PGS'].unique()),
                        order=['Female', 'Male', 'Age<50', 'Age 50–60', 'Age>60']
                        )

        add_error_bars(g, ldl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue_order=['EUR'])

        plt.title(f"Prediction accuracy on LDL Cholesterol (Adj.)\nin samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")

        plt.xlabel("Evaluation Group")
        plt.ylabel('Incremental $R^2$')

        plt.tight_layout()
        plt.savefig(f"figures/section_3/{biobank}_ldl_adj_stratified_accuracy.eps")
        plt.close()

    # -----------------------------------------------------------------
    # Plot the accuracy metrics for HDL-C in EUR:

    hdl_metrics_eur = extract_stratified_evaluation_metrics(
        pheno='HDL',
        biobank=biobank,
        keep_ancestry=['EUR'],
        category=['SexG']
    )
    hdl_metrics_eur.rename(columns={'PGS': 'Stratified PGS'}, inplace=True)
    hdl_metrics_eur = hdl_metrics_eur.reset_index(drop=True)

    plt.figure(figsize=(5, 5))
    g = sns.barplot(hdl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue='Stratified PGS',
                    palette=assign_models_consistent_colors(hdl_metrics_eur['Stratified PGS'].unique()),
                    order=['All', 'Female', 'Male']
                    )

    add_error_bars(g, hdl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue_order=['EUR'])

    plt.title(f"Prediction accuracy on HDL Cholesterol\nin samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")

    plt.xlabel("Evaluation Group")
    plt.ylabel('Incremental $R^2$')

    plt.tight_layout()
    plt.savefig(f"figures/section_3/{biobank}_hdl_stratified_accuracy.eps")
    plt.close()

    # -----------------------------------------------------------------
    # Plot the complementarity between EUR and CSA PGS for HDL-C in EUR samples:

    """
    hdl_comp_eur = extract_complementarity_data(pheno='HDL', biobank=biobank)
    hdl_comp_eur.rename(columns={'Sex': 'SexG', 'SexG': 'Sex'}, inplace=True)

    for x, y in [('EUR', 'CSA'), ('EUR', 'EAS'), ('EUR', 'AFR')]:

        plt.figure(figsize=(5, 5))

        x_label = f'EUR Residuals ($PGS_{{{x}}} - y$)'
        y_label = f'Prediction difference\n($PGS_{{{y}}} - PGS_{{{x}}}$)'

        hdl_comp_eur[x_label] = hdl_comp_eur[x].values - hdl_comp_eur['phenotype'].values
        hdl_comp_eur[y_label] =  hdl_comp_eur[y].values - hdl_comp_eur[x].values

        scatter_with_regression_and_corr(
            data=hdl_comp_eur, x=x_label, y=y_label, hue='Sex',
            palette={
                'Male': '#A1BE95',
                'Female': '#F98866'
            }
        )

        plt.title(f"Complementarity of {x} and {y} PGS for\nHDL in samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")
        plt.tight_layout()
        plt.savefig(f"figures/section_3/{biobank}_hdl_complementarity_{x}_{y}.png", dpi=300)
        plt.close()

    # -----------------------------------------------------------------
    # Plot the complementarity between EUR and CSA PGS for LDL-C in EUR samples:

    ldl_comp_eur = extract_complementarity_data(pheno='LDL', biobank=biobank)
    ldl_comp_eur.rename(columns={'Sex': 'SexG', 'SexG': 'Sex'}, inplace=True)

    for x, y in [('EUR', 'CSA'), ('EUR', 'EAS'), ('EUR', 'AFR')]:

        plt.figure(figsize=(5, 5))

        x_label = f'EUR Residuals ($PGS_{{{x}}} - y$)'
        y_label = f'Prediction difference\n($PGS_{{{y}}} - PGS_{{{x}}}$)'

        ldl_comp_eur[x_label] = ldl_comp_eur[x].values - ldl_comp_eur['phenotype'].values
        ldl_comp_eur[y_label] =  ldl_comp_eur[y].values - ldl_comp_eur[x].values

        scatter_with_regression_and_corr(
            data=ldl_comp_eur, x=x_label, y=y_label, hue='Sex',
            palette={
                'Male': '#A1BE95',
                'Female': '#F98866'
            }
        )

        plt.title(f"Complementarity of {x} and {y} PGS for\nLDL in samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")
        plt.tight_layout()
        plt.savefig(f"figures/section_3/{biobank}_ldl_complementarity_{x}_{y}.png", dpi=300)
        plt.close()
    """
    # -----------------------------------------------------------------
    # Plot stratified performance metrics for log(TG) in CSA and MID samples:

    logtg_metrics = extract_stratified_evaluation_metrics(
        pheno='LOG_TG',
        biobank=biobank,
        keep_ancestry=['CSA', 'MID'],
        category=['Genetic_Distance_Q']
    )
    logtg_metrics = logtg_metrics.loc[logtg_metrics['EvalGroup'] != 'All']
    logtg_metrics.rename(columns={'PGS': 'Stratified PGS'}, inplace=True)
    logtg_metrics = logtg_metrics.reset_index(drop=True)

    plt.figure(figsize=(6, 5))
    g = sns.barplot(logtg_metrics, x='EvalGroup', y='Incremental_R2', hue='Stratified PGS',
                    palette=assign_models_consistent_colors(logtg_metrics['Stratified PGS'].unique()),
                    )

    add_error_bars(g, logtg_metrics, x='EvalGroup', y='Incremental_R2')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Stratified PGS")
    plt.title(f"Prediction accuracy on log(Triglycerides) in samples\nof Middle Eastern and South Asian ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")

    plt.xlabel("Quartile of genetic distance from Europeans")
    plt.ylabel('Incremental $R^2$')

    plt.tight_layout()
    plt.savefig(f"figures/section_3/{biobank}_logtg_stratified_accuracy.eps")
    plt.close()




def generate_weight_figures(weights_df, biobank='ukbb'):

    # Plot data for HDL:
    plt.figure(figsize=(5, 5))
    hdl_weights_df = weights_df.loc[weights_df['Phenotype'] == 'HDL']

    x_order = list(product(sort_groups(hdl_weights_df['Ancestry'].unique(),), ['Female', 'Male']))

    hdl_weights_df.groupby(['Ancestry', 'Sex'])[['EUR', 'AFR', 'EAS', 'CSA']].mean().reindex(x_order).plot(
        kind='bar',
        stacked=True,
        color=assign_models_consistent_colors(['EUR', 'EAS', 'CSA', 'AFR'], 'Set3')
    )

    plt.ylabel("Mean mixing proportions")
    plt.title(f"Mean Mixing Weights for HDL Cholesterol ({BIOBANK_NAME_MAP_SHORT[biobank]})")
    plt.ylim((0., 1.))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Stratified PGS")
    plt.tight_layout()
    plt.savefig(f"figures/section_3/weights_HDL_{biobank}.png", dpi=300)
    plt.close()

    # =================================================================
    # Plot data for LDL:

    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=weights_df.loc[(weights_df['Phenotype'] == 'LDL') & (weights_df['Ancestry'] == 'EUR')],
                    x='Age', y='EUR',
                    hue='Sex',
                    palette={
                        'Male': '#A1BE95',
                        'Female': '#F98866'
                    },
                    alpha=.7)
    plt.xlabel("Age at recruitment")
    plt.ylabel("Mixing weight for EUR PGS")
    plt.title(f"Mixing weights for EUR PGS for LDL Cholesterol\nin samples of "
              f"European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")
    plt.tight_layout()
    plt.savefig(f"figures/section_3/weights_LDL_{biobank}.png", dpi=300)
    plt.close()

    # =================================================================
    # Plot data for LDL adj:

    if biobank == 'ukbb':
        plt.figure(figsize=(5, 5))
        sns.scatterplot(data=weights_df.loc[(weights_df['Phenotype'] == 'LDL_adj') &
                                            (weights_df['Ancestry'] == 'EUR')],
                        x='Age',
                        y='EUR',
                        hue='Sex',
                        palette={
                            'Male': '#A1BE95',
                            'Female': '#F98866'
                        },
                        alpha=.7)
        plt.ylim([0., 1.])
        plt.xlabel("Age at recruitment")
        plt.ylabel("Mixing weight for EUR PGS")
        plt.title(f"Mixing weights for EUR PGS for LDL Cholesterol (adj)\nin samples of "
                  f"European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})")
        plt.tight_layout()
        plt.savefig(f"figures/section_3/weights_LDL_adj_{biobank}.png", dpi=300)
        plt.close()


    # =================================================================
    # Plot data for log(TG):

    plt.figure(figsize=(5, 5))
    subdf = weights_df.loc[(weights_df['Phenotype'] == 'LOG_TG') & (weights_df['Ancestry'].isin(['MID', 'CSA']))]

    anc_symbols = {
        'MID': 'x',
        'CSA': 'o',
    }

    for anc, symbol in anc_symbols.items():
        subsubdf = subdf.loc[subdf['Ancestry'] == anc]

        plt.scatter(subsubdf['PC1'].values,
                    subsubdf['PC2'].values,
                    c=subsubdf['EUR'],
                    marker=symbol,
                    alpha=.8,
                    label=anc)

    # Plot the medoid of European data:
    eur_df = weights_df.loc[(weights_df['Phenotype'] == 'LOG_TG') & (weights_df['Ancestry'] == 'EUR')]
    eur_centroid = np.median(eur_df[[f'PC{i+1}' for i in range(10)]].values, axis=0)
    plt.scatter(eur_centroid[0], eur_centroid[1], marker='*',
                color='red', label='EUR centroid')

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(title="Ancestry group")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Mixing weight for EUR PGS", rotation=90)
    plt.title(f"Mixing weight for EUR PGS for log(Triglycerides)\nin minority "
              f"ancestries in ({BIOBANK_NAME_MAP_SHORT[biobank]})")

    plt.tight_layout()
    plt.savefig(f"figures/section_3/weights_LOG_TG_{biobank}.png", dpi=300)
    plt.close()


def extract_weights_data(biobank='ukbb'):

    w_dfs = []

    for pheno in phenotypes:

        # Extract expert weights from model for same dataset:
        try:
            dataset = PRSDataset.from_pickle(f"data/harmonized_data/{pheno}/{biobank}/test_data.pkl")
            moe_model = MoEPRS.from_saved_model(f"data/trained_models/{pheno}/{biobank}/train_data/{args.moe_model}.pkl")
        except Exception as e:
            print(e)
            continue

        w_df = pd.DataFrame(np.array(['Female', 'Male'])[dataset.get_data_columns("Sex")], columns=['Sex'])
        extract_cols = ["Age", "Ancestry"] + ['PC' + str(i) for i in range(1, 11)]
        w_df[extract_cols] = dataset.get_data_columns(extract_cols)

        prs_col_names = [MODEL_NAME_MAP[prs_col] for prs_col in dataset.prs_cols]

        w_df[prs_col_names] = moe_model.predict_proba(dataset)
        w_df['Phenotype'] = pheno

        w_dfs.append(w_df)

    return pd.concat(w_dfs, axis=0).reset_index(drop=True)


def extract_accuracy_data(test_biobank='ukbb',
                          metric='Incremental_R2',
                          dataset='test_data'):

    dfs = []

    for pheno in phenotypes:
        # Extract accuracy metrics:
        f = f"data/evaluation/{pheno}/{test_biobank}/{dataset}.csv"

        try:
            df = transform_eval_metrics(read_eval_metrics(f))
        except Exception as e:
            print(e)
            continue

        df = df.loc[(df['Model Category'] != 'MoE') | df['Model Name'].isin([
            f'{args.moe_model} (ukbb)',
            f'{args.moe_model} (cartagene)',
        ])]

        df = df.loc[(df['Model Category'] == 'MoE') |
                    (df['Model Category'] == 'MultiPRS') |
                    (df['Training biobank'] == test_biobank.upper())]

        # Rename the models for clarity:
        df['Model Name'] = df['Model Name'].str.replace(f'{args.moe_model} (ukbb)', 'MoEPRS (UKB)', regex=False)
        df['Model Name'] = df['Model Name'].str.replace(f'{args.moe_model} (cartagene)', 'MoEPRS (CaG)', regex=False)
        df['Model Name'] = df['Model Name'].str.replace(f'MultiPRS (ukbb)', 'MultiPRS (UKB)', regex=False)
        df['Model Name'] = df['Model Name'].str.replace(f'MultiPRS (cartagene)', 'MultiPRS (CaG)', regex=False)

        df = postprocess_metrics_df(df,
                                    metric=metric,
                                    category="Ancestry",
                                    min_sample_size=50,
                                    aggregate_single_prs=True,
                                    include_cohort_matched=True)

        dfs.append(df)

    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)

    dfs['Phenotype'] = dfs['Phenotype'].map(phenotypes)
    dfs['Phenotype'] += {
        'ukbb': ' (UKB)',
        'cartagene': ' (CaG)'
    }[test_biobank]

    return dfs


def extract_moe_model_hdl_data(biobank,
                               dataset='test_data',
                               keep_ancestry=None,
                               exclude_ancestry=None):

    if isinstance(keep_ancestry, str):
        keep_ancestry = [keep_ancestry]

    if isinstance(exclude_ancestry, str):
        exclude_ancestry = [exclude_ancestry]

    dat = PRSDataset.from_pickle(f"data/harmonized_data/HDL/{biobank}/{dataset}.pkl")
    model = MoEPRS.from_saved_model(f"data/trained_models/HDL/{biobank}/train_data/MoE-global-int.pkl")

    if keep_ancestry is not None:
        dat.filter_samples(dat.data['Ancestry'].isin(keep_ancestry))
    elif exclude_ancestry is not None:
        dat.filter_samples(~dat.data['Ancestry'].isin(exclude_ancestry))

    # Extract data:
    preds = model.get_predictions(dat)
    pheno = dat.get_phenotype(scaler=model.data_scaler).flatten()
    sex = dat.get_data_columns(['Sex']).flatten()

    plot_data = []

    for i, prs in enumerate(dat.prs_cols):
        plot_data.append(pd.DataFrame({
            'sq_error': (pheno - preds[:, i])**2,
            'HDL': pheno,
            'Sex': sex,
            'Expert PGS': MODEL_NAME_MAP[prs]
        }))

    plot_data = pd.concat(plot_data)
    plot_data['Sex'] = plot_data['Sex'].astype(int).astype(str).map(GROUP_MAP)

    plot_data_cp = plot_data.copy()
    plot_data_cp['Sex'] = 'All'

    return pd.concat([plot_data, plot_data_cp])


def plot_hdl_variance_and_performance_characteristics(biobank='ukbb'):

    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(5, 5))

    dataset = PRSDataset.from_pickle(f"data/harmonized_data/HDL/{biobank}/train_data.pkl")
    summary = dataset.data.groupby('Sex')['HDL'].var()
    summary.index = summary.index.map({0: 'Female', 1: 'Male'})
    summary = pd.concat([summary, pd.Series([np.var(dataset.data['HDL'])], index=['All'])])

    sns.barplot(summary, order=['All', 'Female', 'Male'],
                palette={'All': '#C0C0C0', 'Female': '#F98866', 'Male': '#A1BE95'},
                width=.5,
                ax=ax1)
    ax1.set_yticks(np.round(np.linspace(0., np.ceil(summary.max() / 0.05) * 0.05, 3), 2))
    ax1.set_axisbelow(True)
    ax1.grid(True, axis='y')
    ax1.set_ylabel("HDL Variance")

    sq_error_plot_data = extract_moe_model_hdl_data(biobank, keep_ancestry=['EUR'])

    sns.barplot(data=sq_error_plot_data, x='Sex', y='sq_error', hue='Expert PGS',
                palette=assign_models_consistent_colors(sq_error_plot_data['Expert PGS'].unique()),
                order=['All', 'Female', 'Male'],
                ax=ax2,
                #showmeans=True,
                #showfliers=False,
                #meanprops={"markerfacecolor": "black",
                #           "markeredgecolor": "black"}
                )
    ax2.set_ylabel("MSE")
    ax2.set_yticks([0., .5, 1.])
    ax2.set_axisbelow(True)
    ax2.grid(True, axis='y')

    hdl_metrics_eur = extract_stratified_evaluation_metrics(
        pheno='HDL',
        biobank=biobank,
        keep_ancestry=['EUR'],
        category=['SexG']
    )
    hdl_metrics_eur.rename(columns={'PGS': 'Stratified PGS'}, inplace=True)
    hdl_metrics_eur = hdl_metrics_eur.reset_index(drop=True)

    g = sns.barplot(hdl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue='Stratified PGS',
                    palette=assign_models_consistent_colors(hdl_metrics_eur['Stratified PGS'].unique()),
                    ax=ax3,
                    order=['All', 'Female', 'Male'])

    add_error_bars(g, hdl_metrics_eur, x='EvalGroup', y='Incremental_R2', hue_order=['EUR'])

    ax3.set_xlabel("Evaluation Group")
    ax3.set_ylabel('Incremental $R^2$')
    ax3.set_yticks([0., .05, 0.1, 0.15])
    ax3.set_axisbelow(True)
    ax3.grid(True, axis='y')

    ax2.get_legend().remove()
    ax3.get_legend().remove()

    fig.suptitle("Sex-specific phenotypic variance and prediction accuracy\n"
                 "of ancestry-stratified PGS on HDL Cholesterol\n"
                 f"in samples of European ancestry ({BIOBANK_NAME_MAP_SHORT[biobank]})",
                 fontsize='medium')

    plt.tight_layout()
    plt.savefig(f"figures/section_3/hdl_variance_performance_{biobank}.eps")


def plot_ldl_medication_use_subpanel():

    ldl_metrics_eur = extract_stratified_evaluation_metrics(
        pheno='LDL',
        biobank='ukbb',
        keep_ancestry=['EUR'],
        category=['SexG', 'AgeGroup3']
    )
    ldl_metrics_eur = ldl_metrics_eur.loc[ldl_metrics_eur['PGS'] == 'EUR']
    ldl_metrics_eur = ldl_metrics_eur.loc[ldl_metrics_eur['EvalGroup'] != 'All']
    ldl_metrics_eur.rename(columns={'PGS': 'Stratified PGS'}, inplace=True)
    ldl_metrics_eur = ldl_metrics_eur.reset_index(drop=True)

    # Convert column to ordered categorical
    ordered_cats = ['Female', 'Male', 'Age<50', 'Age 50–60', 'Age>60']
    ldl_metrics_eur['EvalGroup'] = pd.Categorical(ldl_metrics_eur['EvalGroup'],
                                                  categories=ordered_cats,
                                                  ordered=True)
    ldl_metrics_eur = ldl_metrics_eur.sort_values('EvalGroup')

    chol_med_prev = pd.read_csv("data/misc/cholesterol_medication_prevalence.csv")
    chol_med_prev['Group'] = pd.Categorical(chol_med_prev['Group'],
                                            categories=ordered_cats,
                                            ordered=True)
    chol_med_prev = chol_med_prev.sort_values('Group')

    fig, ax1 = plt.subplots(figsize=(5, 5))
    width = 0.4

    x = np.arange(len(ordered_cats))
    y1 = ldl_metrics_eur['Incremental_R2'].values
    y2 = chol_med_prev['Proportion_Using_Medication'].values

    eur_color = assign_models_consistent_colors(['EUR'])['EUR']
    bars1 = ax1.bar(x - width/2, y1, width=width, color=eur_color, label='Quantity 1')
    ax1.set_ylabel('Incremental $R^2$', color='#3C6B64')
    ax1.tick_params(axis='y', labelcolor='#3C6B64')

    # Add the appropriate tick positions to shift the error bars:
    ax1.set_xticks(x - width/2)
    ax1.set_xticklabels(ordered_cats)

    add_error_bars(ax1, ldl_metrics_eur, x='EvalGroup', y='Incremental_R2')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, y2, width=width, color='#B56D7F',
                    hatch='//', edgecolor='#DAB6BF', label='Quantity 2')
    ax2.set_ylabel('Proportion of samples', color='#A03C56')
    ax2.tick_params(axis='y', labelcolor='#A03C56')

    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered_cats, rotation=30)
    ax1.set_xlabel('Evaluation Group')

    # Only include one bar per group in the legend
    custom_legend = [bars1[0], bars2[0]]
    labels = ['Prediction accuracy (EUR)', 'Cholesterol-lowering\nmedication use']
    ax1.legend(custom_legend, labels, loc='upper left')

    ax1.set_ylim([0., 0.22])
    ax2.set_ylim([0., 0.32])
    plt.title("Prediction accuracy on LDL Cholesterol and\n"
              "prevalence of cholesterol-lowering medication\n"
              "in samples of European ancestry (UKB)")

    plt.tight_layout()

    plt.savefig(f"figures/section_3/ldl_accuracy_medication_use_ukbb.eps")
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot Figure 3 of manuscript"
    )

    parser.add_argument('--moe-model', dest='moe_model',
                        type=str, default='MoE-global-int',
                        help="The name of the MoE model to plot as reference.")

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)
    makedir("figures/section_3/")

    phenotypes = {
        'LDL': 'LDL Cholesterol',
        'LDL_adj': 'LDL Cholesterol (adj.)',
        'HDL': 'HDL Cholesterol',
        'LOG_TG': 'log(Triglycerides)',
    }

    # ---------------- Plot accuracy subpanels ----------------
    ukb_data = extract_accuracy_data('ukbb')

    g = plot_combined_accuracy_metrics(ukb_data,
                                       output_f='figures/section_3/accuracy_subpanels_ukbb.eps',
                                       col_order=['LDL Cholesterol (UKB)', 'HDL Cholesterol (UKB)', 'log(Triglycerides) (UKB)'],
                                       hue_order=['MoEPRS (UKB)', 'MoEPRS (CaG)', 'MultiPRS (UKB)', 'MultiPRS (CaG)',
                                                  'Best Single Source PRS', 'Ancestry-matched PRS'],
                                       col_wrap=3,
                                       height=4,
                                       aspect=1.15,
                                       palette={
                                           'MoEPRS (UKB)': '#375E97',
                                           'MoEPRS (CaG)': '#8CA8D8',
                                           'MultiPRS (UKB)': '#FFBB00',
                                           'MultiPRS (CaG)': '#FFE066',
                                           'Best Single Source PRS': '#BC80BD',
                                           'Ancestry-matched PRS': '#66C2A5'
                                       })

    # Plot performance for LDL/LDL_adj only:
    plot_combined_accuracy_metrics(ukb_data,
                                   output_f='figures/section_3/accuracy_subpanels_ukbb_ldl_adj.eps',
                                   col_order=['LDL Cholesterol (UKB)', 'LDL Cholesterol (adj.) (UKB)'],
                                   hue_order=['MoEPRS (UKB)', 'MultiPRS (UKB)',
                                              'Best Single Source PRS', 'Ancestry-matched PRS'],
                                   col_wrap=2,
                                   height=4,
                                   aspect=1.15,
                                   palette={
                                       'MoEPRS (UKB)': '#375E97',
                                       'MoEPRS (CaG)': '#8CA8D8',
                                       'MultiPRS (UKB)': '#FFBB00',
                                       'MultiPRS (CaG)': '#FFE066',
                                       'Best Single Source PRS': '#BC80BD',
                                       'Ancestry-matched PRS': '#66C2A5'
                                   })

    # Plot cartagene data:

    cag_data = extract_accuracy_data('cartagene')

    g = plot_combined_accuracy_metrics(cag_data,
                                       output_f='figures/section_3/accuracy_subpanels_cag.eps',
                                       col_order=['LDL Cholesterol (CaG)', 'HDL Cholesterol (CaG)', 'log(Triglycerides) (CaG)'],
                                       hue_order=['MoEPRS (UKB)', 'MoEPRS (CaG)', 'MultiPRS (UKB)', 'MultiPRS (CaG)',
                                                  'Best Single Source PRS', 'Ancestry-matched PRS'],
                                       col_wrap=3,
                                       height=4,
                                       aspect=1.15,
                                       palette={
                                           'MoEPRS (UKB)': '#375E97',
                                           'MoEPRS (CaG)': '#8CA8D8',
                                           'MultiPRS (UKB)': '#FFBB00',
                                           'MultiPRS (CaG)': '#FFE066',
                                           'Best Single Source PRS': '#BC80BD',
                                           'Ancestry-matched PRS': '#66C2A5'
                                       })


    # ---------------- Plot PGS admixture graphs for the phenotypes ----------------

    for pheno in phenotypes:
        for biobank in ('ukbb', 'cartagene'):

            data_path = f"data/harmonized_data/{pheno}/{biobank}/test_data.pkl"
            model_path = f"data/trained_models/{pheno}/{biobank}/train_data/{args.moe_model}.pkl"

            try:
                p_dataset = PRSDataset.from_pickle(data_path)
                moe_model = MoEPRS.from_saved_model(model_path)
            except Exception as e:
                print(e)
                continue

            biobank_name = BIOBANK_NAME_MAP_SHORT[biobank]

            # Generate the admixture graphs:
            plot_admixture_graphs(p_dataset,
                                  moe_model,
                                  group_col='Ancestry',
                                  title=f"PGS Admixture Graph for {phenotypes[pheno]} ({biobank_name})",
                                  output_file=f"figures/section_3/admixture_graphs_{pheno}_{biobank}.png",
                                  subsample_within_groups=True,
                                  agg_mechanism='sort',
                                  figsize=(g.fig.get_size_inches()[0], 3.1))

    # =================================================================
    # Plot accuracy / admixture graph for HDL for the model with fixed variance:

    args.moe_model = 'MoE-fixed-resid-global-int'

    ukb_data = extract_accuracy_data('ukbb')

    g = plot_combined_accuracy_metrics(ukb_data,
                                       output_f='figures/section_3/accuracy_subpanels_ukbb_fixed_var.eps',
                                       col_order=['LDL Cholesterol (UKB)', 'HDL Cholesterol (UKB)', 'log(Triglycerides) (UKB)'],
                                       hue_order=['MoEPRS (UKB)', 'MoEPRS (CaG)', 'MultiPRS (UKB)', 'MultiPRS (CaG)',
                                                  'Best Single Source PRS', 'Ancestry-matched PRS'],
                                       col_wrap=3,
                                       height=4,
                                       aspect=1.15,
                                       palette={
                                           'MoEPRS (UKB)': '#375E97',
                                           'MoEPRS (CaG)': '#8CA8D8',
                                           'MultiPRS (UKB)': '#FFBB00',
                                           'MultiPRS (CaG)': '#FFE066',
                                           'Best Single Source PRS': '#BC80BD',
                                           'Ancestry-matched PRS': '#66C2A5'
                                       })

    cag_data = extract_accuracy_data('cartagene')

    g = plot_combined_accuracy_metrics(cag_data,
                                       output_f='figures/section_3/accuracy_subpanels_cag_fixed_var.eps',
                                       col_order=['LDL Cholesterol (CaG)', 'HDL Cholesterol (CaG)', 'log(Triglycerides) (CaG)'],
                                       hue_order=['MoEPRS (UKB)', 'MoEPRS (CaG)', 'MultiPRS (UKB)', 'MultiPRS (CaG)',
                                                  'Best Single Source PRS', 'Ancestry-matched PRS'],
                                       col_wrap=3,
                                       height=4,
                                       aspect=1.15,
                                       palette={
                                           'MoEPRS (UKB)': '#375E97',
                                           'MoEPRS (CaG)': '#8CA8D8',
                                           'MultiPRS (UKB)': '#FFBB00',
                                           'MultiPRS (CaG)': '#FFE066',
                                           'Best Single Source PRS': '#BC80BD',
                                           'Ancestry-matched PRS': '#66C2A5'
                                       })


    for pheno in phenotypes:
        # skip LDL_adj:
        if pheno == 'LDL_adj':
            continue

        for biobank in ('ukbb', 'cartagene'):

            p_dataset = PRSDataset.from_pickle(f"data/harmonized_data/{pheno}/{biobank}/test_data.pkl")
            moe_fixed_var = MoEPRS.from_saved_model(f"data/trained_models/{pheno}/{biobank}/train_data/MoE-fixed-resid-global-int.pkl")

            biobank_name = BIOBANK_NAME_MAP_SHORT[biobank]

            # Generate the admixture graphs:
            plot_admixture_graphs(p_dataset,
                                  moe_fixed_var,
                                  group_col='Ancestry',
                                  title=f"PGS Admixture Graph for {phenotypes[pheno]} ({biobank_name})",
                                  output_file=f"figures/section_3/admixture_graphs_{pheno}_{biobank}_fixed_var.png",
                                  subsample_within_groups=True,
                                  agg_mechanism='sort',
                                  figsize=(g.fig.get_size_inches()[0], 3.1))

    args.moe_model = 'MoE-global-int'

    # ---------------- Plot expert weights for the phenotypes ----------------
    sns.set_context("paper", font_scale=1.25)

    ukbb_weights = extract_weights_data(biobank='ukbb')
    generate_weight_figures(ukbb_weights, biobank='ukbb')

    cag_weights = extract_weights_data(biobank='cartagene')
    generate_weight_figures(cag_weights, biobank='cartagene')

    sns.set_context("paper", font_scale=1.25)

    generate_metrics_figures(biobank='ukbb')
    generate_metrics_figures(biobank='cartagene')

    plot_ldl_medication_use_subpanel()

    sns.set_context("paper", font_scale=1.25)
    plot_hdl_variance_and_performance_characteristics(biobank='ukbb')
    plot_hdl_variance_and_performance_characteristics(biobank='cartagene')
