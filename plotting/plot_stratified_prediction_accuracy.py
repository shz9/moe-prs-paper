import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))
from evaluate_predictive_performance import stratified_evaluation
from combined_accuracy_plots import add_error_bars
from plot_utils import BIOBANK_NAME_MAP_SHORT, PHENOTYPE_NAME_MAP, GROUP_MAP, MODEL_NAME_MAP, assign_models_consistent_colors
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from magenpy.utils.system_utils import makedir
import pandas as pd
import numpy as np
from moe import MoEPRS
from PRSDataset import PRSDataset


def extract_stratified_evaluation_metrics(pheno,
                                          biobank,
                                          dataset='full_data',
                                          trained_models=None,
                                          keep_ancestry=None,
                                          exclude_ancestry=None,
                                          category='Sex+Age'
                                          ):

    if isinstance(keep_ancestry, str):
        keep_ancestry = [keep_ancestry]

    if isinstance(exclude_ancestry, str):
        exclude_ancestry = [exclude_ancestry]

    if isinstance(category, str):
        category = [category]

    dat = PRSDataset.from_pickle(f"data/harmonized_data/{pheno}/{biobank}/{dataset}.pkl")

    # ------------------ Add information about genetic distance to Europeans -----------------------

    pc_mat = dat.get_data_columns([f'PC{i+1}' for i in range(10)])
    eur_centroid = np.median(pc_mat[(dat.data['Ancestry'] == 'EUR').values, :], axis=0)

    dat.data['Genetic_Distance'] = np.linalg.norm(pc_mat - eur_centroid, axis=1)
    dat.data['PC1_DIST'] = pc_mat[:, 0] - eur_centroid[0]
    dat.data['PC2_DIST'] = pc_mat[:, 1] - eur_centroid[1]

    # ----------------------------------------------------------------------------------------------

    # Apply filters:
    if keep_ancestry is not None:
        dat.filter_samples(dat.data['Ancestry'].isin(keep_ancestry))
    elif exclude_ancestry is not None:
        dat.filter_samples(~dat.data['Ancestry'].isin(exclude_ancestry))

    # ----------------------------------------------------------------------------------------------

    # Find genetic distance quartiles:
    dat.data['Genetic_Distance_Q'] = pd.qcut(dat.data['Genetic_Distance'], 4,
                                             labels=[f'GD_Q{i+1}' for i in range(4)]).astype(str)
    dat.data['PC1_Q'] = pd.qcut(dat.data['PC1_DIST'], 4, labels=[f'PC1_Q{i+1}' for i in range(4)]).astype(str)
    dat.data['PC2_Q'] = pd.qcut(dat.data['PC2_DIST'], 4, labels=[f'PC2_Q{i+1}' for i in range(4)]).astype(str)

    # ----------------------------------------
    # Define age groups:
    bins = [0, 50, 60, float('inf')]
    labels = ['Age<50', 'Age 50–60', 'Age>60']

    dat.data['AgeGroup3'] = pd.cut(dat.data['Age'], bins=bins, labels=labels, right=False).astype(str)
    dat.data['AgeGroup4'] = pd.qcut(dat.data['Age'], 4, labels=[f'Age Q{i+1}' for i in range(4)]).astype(str)
    dat.data['AgeGroup2'] = np.array(['Age<=55', 'Age>55']).take(dat.get_data_columns("Age").flatten() > 55)

    # ----------------------------------------
    # Map sex labels from integers to strings:
    dat.data['SexG'] = dat.data['Sex'].astype(int).astype(str).map(GROUP_MAP)

    # ----------------------------------------
    # Add composite columns for stratification:
    dat.data['Ancestry+Sex'] = dat.data['Ancestry'] + '-' + dat.data['SexG'].values
    dat.data['Sex+Age'] = dat.data['SexG'].values + '\n(' + dat.data['AgeGroup2'].values + ')'

    eval_df = stratified_evaluation(dat,
                                    trained_models=trained_models, #{'MoEPRS': moe_model},
                                    cat_group_cols=category,
                                    min_group_size=20
                                    )

    eval_df['PGS'] = eval_df['PGS'].map(lambda x: MODEL_NAME_MAP.get(x, x))

    return eval_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot gate parameters for a trained MoE model.')
    parser.add_argument('--moe-model', dest='moe_model',
                        type=str, default='MoE-global-int',
                        help="The name of the MoE model to plot as reference.")
    parser.add_argument('--phenotype', dest='phenotype', default='LDL',
                        type=str,
                        help='The phenotype to plot the stratified evaluation metrics for.')
    parser.add_argument('--metric', dest='metric', default='Incremental_R2',
                        type=str,
                        help='The metric to plot (e.g., Incremental_R2, AUC, etc.).')

    args = parser.parse_args()

    biobank_group_dict = {
        'cartagene': [{
            'Name': 'European',
            'Codes': ['EUR'],
        }, {
            'Name': 'Non-European',
            'Codes': ['CSA', 'EAS', 'AMR', 'MID', 'OTH'],
        }],
        'ukbb': [{'Name': 'EUR', 'Codes': ['EUR']},
                  {'Name': 'MID', 'Codes': ['MID']},
                 {'Name': 'CSA', 'Codes': ['CSA']},
                 {'Name': 'AFR', 'Codes': ['AFR']},
                 {'Name': 'EAS', 'Codes': ['EAS']}],

    }

    # Create output directory:
    makedir(f"figures/stratified_prediction_accuracy/")

    for biobank, ancestries in biobank_group_dict.items():

        for anc in ancestries:

            df = extract_stratified_evaluation_metrics(args.phenotype, biobank, keep_ancestry=anc['Codes'],
                                                       category=['AgeGroup2', 'SexG', 'Sex+Age'])

            df.rename(columns={'PGS': 'Stratified PGS'}, inplace=True)

            plt.figure(figsize=(12, 3))
            g = sns.barplot(df, x='EvalGroup', y=f'{args.metric}', hue='Stratified PGS',
                        palette=assign_models_consistent_colors(df['Stratified PGS'].unique()),
                        order=['All', 'Female', 'Male', 'Age<=55', 'Age>55', 'Female\n(Age<=55)',
                               'Female\n(Age>55)', 'Male\n(Age<=55)', 'Male\n(Age>55)']
                        )

            if f'{args.metric}_err' in df.columns:
                add_error_bars(g, df, x='EvalGroup', y=f'{args.metric}')

            g.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

            plt.title(f"Cohort = {anc['Name']} samples in {BIOBANK_NAME_MAP_SHORT[biobank]}")

            plt.xlabel("Evaluation Group")
            plt.ylabel({
                    'Incremental_R2': 'Incremental $R^2$',
                    'Liability_R2': 'Liability $R^2$',
                    'CORR': 'Pearson $R$',
                       }[args.metric])

            plt.tight_layout()
            plt.savefig(f"figures/stratified_prediction_accuracy/{args.phenotype}_{biobank}_{anc['Name']}_{args.metric}.eps",)
            plt.close()
