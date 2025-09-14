import pandas as pd
import numpy as np
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from magenpy.utils.system_utils import makedir
import argparse
from model.PRSDataset import PRSDataset


def create_prs_dataset(biobank,
                       phenotype,
                       pcs_source,
                       ancestry_source,
                       ancestry_subset=None,
                       ):

    # Read the phenotype file for individuals in this biobank:
    pheno_df = pd.read_csv(f"data/phenotypes/{biobank}/{phenotype}.txt",
                           sep=r'\s+', names=['FID', 'IID', phenotype])
    # Drop individuals with missing phenotype information:
    pheno_df.dropna(subset=[phenotype], inplace=True)

    print("> Number of samples with valid measurements:", len(pheno_df))

    # Read the csv file containing the PRS scores for this phenotype and biobank:
    score_df = pd.read_csv(f"data/scores/{phenotype}/{biobank}.csv.gz",
                           sep=r'\s+')

    prs_cols = [col for col in score_df.columns if col not in ('FID', 'IID')]

    # Merge the scores dataframe with the phenotype dataframe:
    score_df = score_df.merge(pheno_df)

    # ----------------------------------------------------------
    # Process cluster/ancestry information for individuals in this biobank:

    # Read the cluster assignment for individuals in this Biobank (from Alex Diaz-Papkovich):
    cluster_assignment = pd.read_csv(f"data/covariates/{biobank}/cluster_assignment.txt",
                                     sep=r'\s+')
    # Read the cluster interpretation file (i.e. map the cluster ID to names / descriptions):
    cluster_interp = pd.read_csv(f"tables/metadata/{biobank}/cluster_interpretation.csv",
                                 index_col=0, header=0)
    # Read the ancestry assignments for individuals in this Biobank:
    ancestry_df = pd.read_csv(f"data/covariates/{biobank}/{ancestry_source}_ancestry_assignments.txt",
                              sep="\t", header=0)

    if ancestry_source == 'gnomad':
        ancestry_df.rename(columns={'ancestry': 'Ancestry'}, inplace=True)

    # Merge the cluster assignment with the cluster interpretation files:
    cluster_merged = cluster_assignment.merge(cluster_interp, on='Cluster')
    cluster_merged = cluster_merged.merge(ancestry_df, on=['FID', 'IID'], how='right')

    ancestry_cols = [col for col in ancestry_df.columns if col not in ('FID', 'IID')]

    cluster_merged = cluster_merged[['FID', 'IID', 'Description'] + ancestry_cols]
    cluster_merged.rename(columns={'Description': 'UMAP_Cluster'}, inplace=True)

    # Merge the cluster information with the scores dataframe:
    score_df = score_df.merge(cluster_merged, on=['FID', 'IID'], how='left')

    score_df['Ancestry'] = score_df['Ancestry'].fillna('OTH')
    score_df['UMAP_Cluster'] = score_df['UMAP_Cluster'].fillna('N/A')
    score_df.fillna(0., inplace=True)

    # If the user requested subsetting to particular ancestry groups,
    # then we do that here:
    if ancestry_subset is not None:

        if isinstance(ancestry_subset, str):
            ancestry_subset = [ancestry_subset]

        score_df = score_df.loc[score_df['Ancestry'].isin(ancestry_subset)]

    # ----------------------------------------------------------
    # Process covariates for individuals in this biobank:

    covariates_cols = ['Sex'] + ['PC' + str(i + 1) for i in range(10)] + ['Age']
    covar_df = pd.read_csv(f"data/covariates/{biobank}/covars_{pcs_source}_pcs.txt",
                           names=['FID', 'IID'] + covariates_cols,
                           sep=r'\s+')

    score_df = score_df.merge(covar_df, on=['FID', 'IID'])

    # Drop samples with missing values:
    n_samples_before = len(score_df)

    score_df = score_df.dropna().reset_index(drop=True)

    if len(score_df) < n_samples_before:
        print(f"Dropped {n_samples_before - len(score_df)} samples with missing values.")

    assert len(score_df) > 0, "No samples left after merging dataframes and purging missing values!"
    print("Final number of samples in the PRS dataset:", len(score_df))

    print("> Phenotype descriptive statistics:")
    print(score_df[phenotype].describe())

    return PRSDataset(
        score_df,
        phenotype,
        meta_cols=['FID', 'IID', 'UMAP_Cluster'] + ancestry_cols,
        covariates_cols=covariates_cols,
        prs_cols=prs_cols
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
        Commandline arguments for creating PRS datasets
    """)

    parser.add_argument('--biobank', dest='biobank', type=str, required=True,
                        choices={'ukbb', 'cartagene'},
                        help='The name of the biobank to create the PRS dataset for.')
    parser.add_argument('--phenotype', dest='phenotype', type=str, required=True,
                        help='The name of the phenotype to create the PRS dataset for.')
    parser.add_argument('--pcs-source', dest='pcs_source', type=str, default='1kghdp',
                        choices={'gnomad', 'cartagene', 'ukbb', '1kghdp'},
                        help='The source of the principal components.')
    parser.add_argument('--ancestry-source', dest='ancestry_source', type=str, default='1kghdp',
                        choices={'gnomad', '1kghdp'},
                        help='The source of the ancestry assignments.')
    parser.add_argument('--data-suffix', dest='data_suffix', type=str, default='',
                        help='The suffix to append to the processed data files (default: "").')
    parser.add_argument('--prop-test', dest='prop_test', type=float, default=0.3,
                        help='The proportion of samples to use for testing (default: 0.3).')
    parser.add_argument('--seed', dest='seed', type=int, default=7209,
                        help='The seed for the random number generator (default: 7209).')

    args = parser.parse_args()

    # Set the random seed:
    np.random.seed(args.seed)

    print(f'> Creating PRS dataset for {args.phenotype} among {args.biobank} participants...')

    prs_dataset = create_prs_dataset(args.biobank, args.phenotype, args.pcs_source, args.ancestry_source)

    print(f"> Saving processed data to: data/harmonized_data/{args.phenotype}/{args.biobank}/")
    makedir(f"data/harmonized_data/{args.phenotype}/{args.biobank}/")

    # Save the entire dataset:
    prs_dataset.save(f"data/harmonized_data/{args.phenotype}/{args.biobank}/full_data{args.data_suffix}.pkl")

    # Split the dataset into training and testing sets:
    train_data, test_data = prs_dataset.train_test_split(test_size=args.prop_test)

    train_data.save(f"data/harmonized_data/{args.phenotype}/{args.biobank}/train_data{args.data_suffix}.pkl")
    test_data.save(f"data/harmonized_data/{args.phenotype}/{args.biobank}/test_data{args.data_suffix}.pkl")
