import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))
from plot_utils import (sort_groups, read_eval_metrics, transform_eval_metrics,
                        assign_ancestry_consistent_colors, GROUP_MAP, BIOBANK_NAME_MAP_SHORT)
from plot_predictive_performance import postprocess_metrics_df, generate_model_colors
from combined_accuracy_plots import plot_combined_accuracy_metrics
from evaluate_predictive_performance import stratified_evaluation
from magenpy.utils.system_utils import makedir
from PRSDataset import PRSDataset
from moe import MoEPRS
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def assign_ancestry_consistent_markers(groups, markers=None):
    """
    Assign consistent markers to the ancestry groups for plotting.
    :param groups: A list of ancestry group names
    :param markers: A dictionary of group names and markers
    :return: A dictionary of group names and markers
    """
    if markers is None:
        markers = {
            'AFR': 'o',
            'AMR': 's',
            'EAS': '^',
            'EUR': 'D',
            'CSA': 'v',
            'MID': '*',
            'OTH': 'X',
        }

    return {k: markers[k] for k in groups if k in markers}


def plot_gate_mixing_weights_colored_by_ancestry(weights_df,
                                                 output_f,
                                                 order=None):

    g = sns.relplot(
        data=weights_df,
        x='Age',
        y='P(Male_PGS)',
        col='Phenotype',        # Creates one subplot per phenotype
        col_order=order,
        hue='Ancestry',         # Color by Sex
        hue_order=sort_groups(weights_df['Ancestry'].unique()),
        style='Sex',            # Marker style by Ancestry
        kind='scatter',
        alpha=0.3,
        height=5,
        aspect=1,
        palette=assign_ancestry_consistent_colors(weights_df['Ancestry'].unique()),
        markers={
            'Male': 'o',
            'Female': 'X',
        },
        facet_kws={'sharex': True, 'sharey': True}
    )

    # Set the alpha of legend handles to 1 (full opacity)
    for lh in g.legend.legend_handles:
        lh.set_alpha(1)

    # Remove the "Phenotype = " prefix from the title:
    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith('Phenotype = '):
            ax.set_title(title.replace('Phenotype = ', ''))

    g.set_axis_labels(x_var="Age at recruitment", y_var="Mixing weight for male PGS\nP(Male_PGS)")

    plt.savefig(output_f, bbox_inches='tight', dpi=400)
    plt.close()


def plot_gate_mixing_weights_colored_by_sex(weights_df,
                                            output_f,
                                            order=None):

    g = sns.relplot(
        data=weights_df,
        x='Age',
        y='P(Male_PGS)',
        col='Phenotype',        # Creates one subplot per phenotype
        col_order=order,
        hue='Sex',         # Color by Sex
        hue_order=['Female', 'Male'],
        style='Ancestry',            # Marker style by Ancestry
        kind='scatter',
        alpha=0.3,
        height=5,
        aspect=1,
        palette={
            'Male': '#A1BE95',
            'Female': '#F98866',
        },
        markers=assign_ancestry_consistent_markers(weights_df['Ancestry'].unique()),
        style_order=sort_groups(weights_df['Ancestry'].unique()),
        facet_kws={'sharex': True, 'sharey': True}
    )

    # Set the alpha of legend handles to 1 (full opacity)
    for lh in g.legend.legend_handles:
        lh.set_alpha(1)

    # Remove the "Phenotype = " prefix from the title:
    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith('Phenotype = '):
            ax.set_title(title.replace('Phenotype = ', ''))

    g.set_axis_labels(x_var="Age at recruitment", y_var="Mixing weight for male PGS\nP(Male_PGS)")

    plt.savefig(output_f, bbox_inches='tight', dpi=400)
    plt.close()


def extract_weights_data(biobank='ukbb'):

    dfs = []

    for pheno in phenotypes:

        # Extract expert weights from model for same dataset:
        try:
            dataset = PRSDataset.from_pickle(f"data/harmonized_data/{pheno}/{biobank}/test_data.pkl")
            moe_model = MoEPRS.from_saved_model(f"data/trained_models/{pheno}/{biobank}/train_data/{args.moe_model}.pkl")
        except Exception as e:
            print(e)
            continue

        w_df = pd.DataFrame(np.array(['Female', 'Male'])[dataset.get_data_columns("Sex")], columns=['Sex'])
        w_df[['Age', 'Ancestry']] = dataset.get_data_columns(["Age", "Ancestry"])

        prs_col_names = []
        for prs_col in dataset.prs_cols:
            if '_F' in prs_col:
                prs_col_names.append('P(Female_PGS)')
            else:
                prs_col_names.append('P(Male_PGS)')

        w_df[prs_col_names] = moe_model.predict_proba(dataset)
        w_df['Phenotype'] = phenotypes[pheno] + {
            'ukbb': ' (UKB)',
            'cartagene': ' (CaG)'
        }[biobank]

        dfs.append(w_df)

    return pd.concat(dfs, axis=0).reset_index(drop=True)




def extract_stratified_evaluation_metrics(pheno,
                                          test_biobank,
                                          keep_ancestry=None,
                                          exclude_ancestry=None,
                                          category='Sex+Age'
                                          ):

    if isinstance(keep_ancestry, str):
        keep_ancestry = [keep_ancestry]

    if isinstance(exclude_ancestry, str):
        exclude_ancestry = [exclude_ancestry]

    dat = PRSDataset.from_pickle(f"data/harmonized_data/{pheno}/{test_biobank}/full_data.pkl")

    # Apply filters:
    if keep_ancestry is not None:
        dat.filter_samples(dat.data['Ancestry'].isin(keep_ancestry))
    elif exclude_ancestry is not None:
        dat.filter_samples(~dat.data['Ancestry'].isin(exclude_ancestry))

    dat.data['Ancestry+Sex'] = dat.data['Ancestry'] + '-' + dat.data['Sex'].astype(int).astype(str).map(GROUP_MAP)
    dat.data['Sex+Age'] = (dat.data['Sex'].astype(int).astype(str).map(GROUP_MAP) +
                           '\n(' + np.array(['Age<=55', 'Age>55']).take(dat.get_data_columns("Age").flatten() > 55) + ')'
                           )

    eval_df = stratified_evaluation(dat,
                                    trained_models=None,
                                    cat_group_cols=category,
                                    min_group_size=20
                                    )

    # Remove the "All" category:
    eval_df = eval_df.loc[eval_df['EvalGroup'] != 'All']

    eval_df = eval_df.loc[eval_df['PGS'].isin([f'PGS_{pheno}_F', f'PGS_{pheno}_M'])]

    tr_df = eval_df.pivot(index='EvalGroup', columns='PGS', values='Incremental_R2').reset_index()
    tr_df['Ratio'] = tr_df[f'PGS_{pheno}_M'] / tr_df[f'PGS_{pheno}_F']

    return tr_df


def plot_stratified_evaluation_creatinine():

    ukb_mid_data = extract_stratified_evaluation_metrics('CRTN','ukbb', keep_ancestry=['MID'])
    ukb_mid_data['Cohort'] = 'MID Samples in UKB'
    ukb_afr_data = extract_stratified_evaluation_metrics('CRTN', 'ukbb', keep_ancestry=['AFR'])
    ukb_afr_data['Cohort'] = 'AFR Samples in UKB'
    ukb_csa_data = extract_stratified_evaluation_metrics('CRTN','ukbb', keep_ancestry=['CSA'])
    ukb_csa_data['Cohort'] = 'CSA Samples in UKB'
    cag_noneur_data = extract_stratified_evaluation_metrics('CRTN','cartagene', exclude_ancestry=['EUR'])
    cag_noneur_data['Cohort'] = 'Non-European Samples in CaG'
    cag_eur_data = extract_stratified_evaluation_metrics('CRTN','cartagene', keep_ancestry=['EUR'])
    cag_eur_data['Cohort'] = 'European Samples in CaG'

    combined_df = pd.concat([ukb_mid_data, ukb_afr_data, ukb_csa_data, cag_noneur_data, cag_eur_data],
                            axis=0, ignore_index=True)

    g = sns.catplot(combined_df, kind='bar', x='EvalGroup', y='Ratio', row='Cohort',
                    height=2., aspect=2.5, sharey=False,
                    hue='EvalGroup',
                    palette=['#F98866',  '#F98866', '#A1BE95', '#A1BE95'])

    for ax in g.axes.flatten():
        ax.axhline(y=1., color='#878787', linestyle=':')

    g.set_axis_labels(x_var="", y_var="")
    g.fig.supylabel("Relative Incremental $R^2$\n(Male PGS/Female PGS Ratio)", multialignment='center')
    g.fig.supxlabel("Evaluation Group", multialignment='center')
    g.fig.suptitle("Stratified Relative Prediction Accuracy (Creatinine)", multialignment='center')

    g.fig.tight_layout()

    plt.savefig("figures/section_1/stratified_creatinine_accuracy.pdf", bbox_inches='tight', dpi=400)
    plt.close()


def plot_stratified_evaluation_urate():

    ukb_eas_data = extract_stratified_evaluation_metrics('URT', 'ukbb', keep_ancestry=['EAS'])
    ukb_eas_data['Cohort'] = 'EAS Samples in UKB'
    ukb_afr_data = extract_stratified_evaluation_metrics('URT', 'ukbb', keep_ancestry=['AFR'])
    ukb_afr_data['Cohort'] = 'AFR Samples in UKB'
    ukb_csa_data = extract_stratified_evaluation_metrics('URT', 'ukbb', keep_ancestry=['CSA'])
    ukb_csa_data['Cohort'] = 'CSA Samples in UKB'
    cag_noneur_data = extract_stratified_evaluation_metrics('URT', 'cartagene', exclude_ancestry=['EUR'])
    cag_noneur_data['Cohort'] = 'Non-European Samples in CaG'
    cag_eur_data = extract_stratified_evaluation_metrics('URT', 'cartagene', keep_ancestry=['EUR'])
    cag_eur_data['Cohort'] = 'European Samples in CaG'

    combined_df = pd.concat([ukb_eas_data, ukb_afr_data, ukb_csa_data, cag_noneur_data, cag_eur_data],
                            axis=0, ignore_index=True)

    g = sns.catplot(combined_df, kind='bar', x='EvalGroup', y='Ratio', row='Cohort',
                    height=2., aspect=2.5, sharey=False,
                    hue='EvalGroup',
                    palette=['#F98866',  '#F98866', '#A1BE95', '#A1BE95'])

    for ax in g.axes.flatten():
        ax.axhline(y=1., color='#878787', linestyle=':')

    g.set_axis_labels(x_var="", y_var="")
    g.fig.supylabel("Relative Incremental $R^2$\n(Male PGS/Female PGS Ratio)", multialignment='center')
    g.fig.supxlabel("Evaluation Group", multialignment='center')
    g.fig.suptitle("Stratified Relative Prediction Accuracy (Urate)", multialignment='center')

    g.fig.tight_layout()

    plt.savefig("figures/section_1/stratified_urate_accuracy.pdf", bbox_inches='tight', dpi=400)
    plt.close()


def extract_accuracy_data(test_biobank='ukbb',
                          train_biobank='ukbb',
                          restrict_to_same_biobank=True):

    dfs = []

    for pheno in phenotypes:

        # Extract accuracy metrics:
        f = f"data/evaluation/{pheno}/{test_biobank}/test_data.csv"
        try:
            df = transform_eval_metrics(read_eval_metrics(f))
        except Exception as e:
            print(e)
            continue

        df = df.loc[(df['Model Category'] != 'MoE') | df['Model Name'].isin([#f'MoE-CFG ({args.biobank})',
            f'{args.moe_model} ({train_biobank})'])]

        df['Model Name'] = df['Model Name'].str.replace(f' ({train_biobank})', '', regex=False)
        df['Model Name'] = df['Model Name'].str.replace(f'{args.moe_model}', 'MoEPRS', regex=False)

        if restrict_to_same_biobank:
            df = df.loc[df['Training biobank'] == df['Test biobank']]

        df = postprocess_metrics_df(df,
                                    "Incremental_R2",
                                    category="Sex",
                                    aggregate_single_prs=False,
                                    include_cohort_matched=False)

        dfs.append(df)

    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    dfs['Phenotype'] = dfs['Phenotype'].map(phenotypes)
    dfs['Phenotype'] += {
        'ukbb': ' (UKB)',
        'cartagene': ' (CaG)'
    }[test_biobank]

    return dfs


def plot_phenotypic_variance(pheno, biobank='ukbb'):

    dataset = PRSDataset.from_pickle(f"data/harmonized_data/{pheno}/{biobank}/full_data.pkl")

    dataset.data['SexG'] = dataset.data['Sex'].astype(int).astype(str).map(GROUP_MAP)
    dataset.data['AgeGroup2'] = np.array(['Age<=55', 'Age>55']).take(dataset.get_data_columns("Age").flatten() > 55)
    dataset.data['Sex+Age'] = dataset.data['SexG'].values + ' (' + dataset.data['AgeGroup2'].values + ')'

    summary = dataset.data.groupby(['Ancestry', 'Sex+Age'])[pheno].agg(['var', 'count']).reset_index()

    unique_ancestries = sort_groups(summary['Ancestry'].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    g = sns.swarmplot(data=summary, x='Ancestry', y='var', hue='Sex+Age', dodge=True,
                      order=unique_ancestries,
                  palette={
                    'Male (Age<=55)': '#A1BE95',
                    'Male (Age>55)': '#5B7F61',
                    'Female (Age<=55)': '#F98866',
                    'Female (Age>55)': '#B23C17',
                    })

    for i, x in enumerate(unique_ancestries):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.2, zorder=0)

    ax.axhline(np.var(dataset.data[pheno]), ls='--', color='silver', label='Overall variance')
    ax.axhline(np.var(dataset.data.loc[dataset.data['SexG'] == 'Female', pheno]),
               ls='--', color='#F98866', label='Female variance')
    ax.axhline(np.var(dataset.data.loc[dataset.data['SexG'] == 'Male', pheno]),
               ls='--', color='#A1BE95', label='Male variance')

    sns.scatterplot(ax=ax, data=dataset.data.groupby('Ancestry')[pheno].agg(['var']).reset_index(),
                    x='Ancestry', y='var',
                    marker='D', color='#b19cd9', s=36,
                    label='Per-ancestry variance')

    ax.set_ylabel(f"Variance of {phenotypes[pheno]}")
    ax.set_title(f"Variance of {phenotypes[pheno]} across ancestry, age, and sex ({BIOBANK_NAME_MAP_SHORT[biobank]})")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f"figures/section_1/phenotypic_variance_{pheno}_{biobank}.pdf",
                bbox_inches='tight', dpi=400)
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot figures related to section 1 of manuscript"
    )

    parser.add_argument('--moe-model', dest='moe_model',
                        type=str, default='MoE-global-int',
                        help="The name of the MoE model to plot as reference.")

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)
    makedir("figures/section_1/")

    phenotypes = {
        'TST': 'Testosterone',
        'URT': 'Urate',
        'CRTN': 'Creatinine'
    }

    palette = {
        'Male PGS': '#A1BE95',
        'Female PGS': '#F98866',
        'MoEPRS': '#375E97',
        'MultiPRS': '#FFBB00',
    }

    hue_order = ['MoEPRS', 'MultiPRS', 'Female PGS', 'Male PGS']

    ukbb_metrics_dfs = extract_accuracy_data()

    ukbb_metrics_dfs['Model Name'] = ukbb_metrics_dfs['Model Name'] + np.where(ukbb_metrics_dfs['Model Category'].isin(['MoE', 'MultiPRS']), '', ' PGS')

    ukbb_w_dfs = extract_weights_data()

    plot_combined_accuracy_metrics(
        ukbb_metrics_dfs,
        "figures/section_1/ukb_accuracy_subpanels.pdf",
        col_order=['Testosterone (UKB)', 'Creatinine (UKB)', 'Urate (UKB)'],
        palette=palette,
        hue_order=hue_order
    )

    plot_gate_mixing_weights_colored_by_sex(
        ukbb_w_dfs,
        "figures/section_1/ukb_weights.png",
        order=['Testosterone (UKB)', 'Creatinine (UKB)', 'Urate (UKB)'],
    )

    plot_gate_mixing_weights_colored_by_ancestry(
        ukbb_w_dfs,
        "figures/section_1/ukb_weights_ancestry_colored.png",
        order=['Testosterone (UKB)', 'Creatinine (UKB)', 'Urate (UKB)'],
    )

    cartagene_metrics_dfs = extract_accuracy_data(test_biobank='cartagene',
                                                  train_biobank='cartagene')

    cartagene_metrics_dfs['Model Name'] = cartagene_metrics_dfs['Model Name'] + np.where(cartagene_metrics_dfs['Model Category'].isin(['MoE', 'MultiPRS']), '', ' PGS')

    cartagene_w_dfs = extract_weights_data(biobank='cartagene')

    plot_combined_accuracy_metrics(
        cartagene_metrics_dfs,
        "figures/section_1/cartagene_accuracy_subpanels.pdf",
        col_order=['Creatinine (CaG)', 'Urate (CaG)'],
        palette=palette,
        hue_order=hue_order
    )

    plot_gate_mixing_weights_colored_by_sex(
        cartagene_w_dfs,
        "figures/section_1/cartagene_weights.png",
        order=['Creatinine (CaG)', 'Urate (CaG)'],
    )

    plot_gate_mixing_weights_colored_by_ancestry(
        cartagene_w_dfs,
        "figures/section_1/cartagene_weights_ancestry_colored.png",
        order=['Creatinine (CaG)', 'Urate (CaG)'],
    )

    sns.set_context("paper", font_scale=1.25)

    plot_stratified_evaluation_creatinine()
    plot_stratified_evaluation_urate()

    plot_phenotypic_variance('CRTN', biobank='ukbb')
    plot_phenotypic_variance('URT', biobank='ukbb')
    plot_phenotypic_variance('CRTN', biobank='cartagene')
    plot_phenotypic_variance('URT', biobank='cartagene')
