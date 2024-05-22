"""

DEPRECATED (NOT USED ANYMORE)

PC loadings & Random forest model were downloaded from gnomad v3.1:
https://console.cloud.google.com/storage/browser/_details/gcp-public-data--gnomad/release/3.1/pca/
"""

import os.path as osp
import pandas as pd
import numpy as np
import ast
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from data_preparation.utils import liftover_coordinates
import argparse


def process_gnomad_pca_loadings(file_path):
    """
    Process the PC loadings from gnomad v3.1. This function takes a path to a file
    containing the PC loadings from gnomad v3.1 and returns a pandas DataFrame
    containing the parsed/processed loadings.
    """

    df = pd.read_csv(file_path, sep="\t")
    df[['CHR', 'POS']] = df.locus.str.split(':', expand=True)
    df['CHR'] = df.CHR.str.replace('chr', '').astype(np.int64)
    df['POS'] = df['POS'].astype(np.int64)
    df[['A1', 'A2']] = df.alleles.str.extract(r'\"(.*)\",\"(.*)\"')

    loadings = np.array(df['loadings'].apply(ast.literal_eval).tolist())
    df[[f'PC_{i}' for i in range(1, loadings.shape[1] + 1)]] = loadings

    df.drop(columns=['locus', 'alleles', 'loadings'], inplace=True)
    df.rename(columns={'pca_af': 'MAF'}, inplace=True)

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Process the gnomad v3.1 PCA loadings.')
    parser.add_argument('--pca-loadings', dest='pca_loadings',
                        type=str, help='Path to the PCA loadings file.',
                        default='data/gnomad_data/gnomad.v3.1.pca_loadings.tsv.gz')
    parser.add_argument('--output-dir', dest='output_dir',
                        type=str, help='Path to the output directory.',
                        default='data/gnomad_data/')
    parser.add_argument('--liftover-chain', dest='liftover_chain',
                        type=str, help='Path to the liftover chain file.')

    args = parser.parse_args()

    # Read + process the gnomad PC loadings:
    processed_pcl = process_gnomad_pca_loadings(args.pca_loadings)
    print(f"> Extract PCA loadings for {len(processed_pcl)} variants...")
    # Save the processed dataframe to file:
    processed_pcl.to_csv(osp.join(args.output_dir, 'processed_pca_loadings_GRCh38.csv'), index=False)
    # Save bed files for genotype extraction:
    processed_pcl[['CHR', 'POS', 'POS']].to_csv(osp.join(args.output_dir, 'gnomad_pca_snps_GRCh38.bed'),
                                                sep="\t", index=False, header=False)

    # Liftover the coordinates from hg38 to hg19:
    processed_pcl['POS'] = liftover_coordinates(processed_pcl,
                                                source='hg38', target='hg19',
                                                chain_file=args.liftover_chain)
    # Discard SNPs with no matching positions in hg19
    processed_pcl = processed_pcl.loc[processed_pcl.POS > 0]
    print(f"> {len(processed_pcl)} variants remain after liftover to GRCh37...")
    # Save the processed dataframe to file:
    processed_pcl.to_csv(osp.join(args.output_dir, 'processed_pca_loadings_GRCh37.csv'), index=False)
    # Save bed files for genotype extraction:
    processed_pcl[['CHR', 'POS', 'POS']].to_csv(osp.join(args.output_dir, 'gnomad_pca_snps_GRCh37.bed'),
                                                sep="\t", index=False, header=False)
