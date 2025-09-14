import numpy as np
import pandas as pd
import os.path as osp
import argparse


def extract_pgsc_pcs(biobank):
    """
    Extract the principal components and update the covariates file for the biobank.
    """

    # Read the PCs from the PGSC pipeline:
    pc_df = pd.read_csv(f"data/pgsc_calc_scores/{biobank}/{biobank}/score/{biobank}_popsimilarity.txt.gz",
                        sep="\t", usecols=['sampleset', 'FID', 'IID',
                                           'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'],
                        dtype={'FID': str, 'IID': str})
    pc_df = pc_df[pc_df['sampleset'] == biobank]

    # Read non-PC covariates from the covariates file:
    covar_df = pd.read_csv(f"data/covariates/{biobank}/covars_{biobank}_pcs.txt",
                           sep="\t", names=['FID', 'IID', 'Sex', 'Age'],
                           usecols=[0, 1, 2, 13],
                           dtype={'FID': str, 'IID': str})

    # Merge the two dataframes (on FID and IID) and update the covariates file:
    covar_df = covar_df.merge(pc_df, on=['FID', 'IID'])
    # Sort the columns:
    covar_df = covar_df[['FID', 'IID', 'Sex'] + [f'PC{i}' for i in range(1, 11)] + ['Age']]

    # Output the updated covariates file:
    covar_df.to_csv(f"data/covariates/{biobank}/covars_1kghdp_pcs.txt", sep="\t", index=False, header=False)


def extract_pgsc_ancestry(biobank):
    """
    Postprocess the ancestry assignments obtained from the PGSC pipeline.
    """
    anc_df = pd.read_csv(f"data/pgsc_calc_scores/{biobank}/{biobank}/score/{biobank}_popsimilarity.txt.gz",
                         sep="\t", usecols=['sampleset', 'FID', 'IID',
                                            'RF_P_AFR', 'RF_P_AMR', 'RF_P_CSA', 'RF_P_EAS',
                                            'RF_P_EUR', 'RF_P_MID', 'MostSimilarPop', 'MostSimilarPop_LowConfidence'],
                         dtype={'FID': str, 'IID': str})
    anc_df = anc_df[anc_df['sampleset'] == biobank]

    anc_df['Ancestry'] = np.where(anc_df['MostSimilarPop_LowConfidence'], 'OTH', anc_df['MostSimilarPop'])

    anc_df.rename(columns={
        'RF_P_AFR': 'AFR',
        'RF_P_AMR': 'AMR',
        'RF_P_CSA': 'CSA',
        'RF_P_EAS': 'EAS',
        'RF_P_EUR': 'EUR',
        'RF_P_MID': 'MID'
    }, inplace=True)

    anc_df[['FID', 'IID', 'AFR', 'AMR', 'CSA', 'EAS', 'EUR', 'MID', 'Ancestry']].to_csv(
        f"data/covariates/{biobank}/1kghdp_ancestry_assignments.txt", sep="\t", index=False
    )


def extract_pgsc_scores(biobank):
    """
    Postprocess the PRS scores obtained from the PGSC pipeline.
    """

    pgs_df = pd.read_csv(f"data/pgsc_calc_scores/{biobank}/{biobank}/score/{biobank}_pgs.txt.gz",
                         sep="\t", usecols=['sampleset', 'FID', 'IID', 'PGS', 'SUM'],
                         dtype={'sampleset': str, 'FID': str, 'IID': str, 'PGS': str, 'SUM': float})
    pgs_df = pgs_df.loc[pgs_df.sampleset == biobank].drop(columns=['sampleset'])

    # Pivot the PGS dataframe:
    pgs_df = pgs_df.pivot(index=['FID', 'IID'], columns='PGS', values='SUM').reset_index()

    # Rename the PGSs to remove information related to the genome build:

    pgs_df.columns = [col.replace('_hmPOS_GRCh37', '').replace('_hmPOS_GRCh38', '') for col in pgs_df.columns]

    # Read the table mapping phenotypes to PGS IDs:
    pheno_df = pd.read_csv("tables/phenotype_prs_table.csv")
    # Create a dictionary mapping phenotype name (`Phenotype_short`) to list of PGS IDs:
    pheno_map = pheno_df.groupby('Phenotype_short')['PGS'].apply(list).to_dict()

    for pheno, pgss in pheno_map.items():
        pgs_df_subset = pgs_df[['FID', 'IID'] + [col for col in pgss if col in pgs_df.columns]]

        assert len(pgs_df_subset.columns) > 2, f"No PGS scores found for {pheno} in {biobank}."

        pgs_df_subset.to_csv(
            f"data/scores/{pheno}/{biobank}.csv.gz", sep="\t", index=False
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Postprocess pgsc_calc data to prepare for meta PRS pipeline.')
    parser.add_argument('--biobank', dest='biobank', type=str, required=True,
                        choices={'ukbb', 'cartagene'},
                        help='The name of the biobank to postprocess the data for.')

    args = parser.parse_args()

    print("> Extracting principal components from the PGSC pipeline...")
    extract_pgsc_pcs(args.biobank)

    print("> Extracting ancestry assignments from the PGSC pipeline...")
    extract_pgsc_ancestry(args.biobank)

    print("> Extracting PRS scores from the PGSC pipeline...")
    extract_pgsc_scores(args.biobank)

    print("Done!")

