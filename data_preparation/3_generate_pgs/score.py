import pandas as pd
import glob
import argparse
import os.path as osp
from magenpy.parsers.plink_parsers import parse_bim_file
from magenpy.utils.system_utils import makedir
from magenpy.utils.model_utils import merge_snp_tables
import magenpy as mgp
from viprs.model.BayesPRSModel import BayesPRSModel
import numpy as np


def read_combine_pgs_files(pgs_dir, ref_genotype_dir, add_random_beta=False):

    ukb_snp_table = pd.concat([parse_bim_file(f) 
                               for f in glob.glob(osp.join(ref_genotype_dir, "*.bim"))])

    new_table = ukb_snp_table.copy()
    pgs_ids = []

    for i, pgs_file in enumerate(glob.glob(osp.join(pgs_dir, "*.txt.gz"))):
    
        col_map = {
            'chr_name': 'CHR',
            'effect_allele': 'A1',
            'other_allele': 'A2'
        }

        print(pgs_file)

        pgs_ids.append(osp.basename(pgs_file).replace(".txt.gz", ""))
        df = pd.read_csv(pgs_file, comment='#', sep="\t")

        if 'hm_pos' in df.columns:
            col_map.update({'hm_pos': 'POS'})
        else:
            col_map.update({'chr_position': 'POS'})

        col_map.update({'effect_weight': f'BETA_{i}'})

        df.rename(columns=col_map, inplace=True)

        if df['CHR'].dtype == object:
            df = df.loc[~df['CHR'].isin(['X', 'Y', 'XY', 'MT']),]
            df['CHR'] = df['CHR'].astype(np.int32)
    
        df = merge_snp_tables(new_table,
                              df[['CHR', 'POS', 'A1', 'A2', f'BETA_{i}']],
                              how='left',
                              signed_statistics=[f'BETA_{i}'])

        new_table = new_table.merge(df[['CHR', 'POS', 'A1', 'A2', f'BETA_{i}']], 
                                    how='left',
                                    on=['CHR', 'POS', 'A1', 'A2'])

    new_table = new_table.dropna(subset=[f'BETA_{i}' for i in range(len(pgs_ids))])

    if add_random_beta:
        new_table[f'BETA_{len(pgs_ids)+1}'] = np.random.normal(size=new_table.shape[0],
                                                               scale=np.sqrt(1./new_table.shape[0]))
        pgs_ids.append("Random")

    return new_table, pgs_ids


def score(ref_genotype_dir, pgs_table):

    gdl = mgp.GWADataLoader(bed_files=osp.join(ref_genotype_dir, "*.bed"), backend='plink')

    prsm = BayesPRSModel(gdl)
    prsm.set_model_parameters(pgs_table)

    pgs = prsm.predict()
    pgs_df = pd.DataFrame(pgs)
    pgs_df['FID'] = gdl.sample_table.fid.astype(np.int64)
    pgs_df['IID'] = gdl.sample_table.iid.astype(np.int64)

    return pgs_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform linear scoring using PGS Catalog-formatted files.')

    parser.add_argument('--genotype_dir', dest='genotype_dir', type=str, required=True,
                        help='The directory where the genotype files are stored.')
    parser.add_argument('--pgs-dir', dest='pgs_dir', type=str, required=True,
                        help='The directory where the PGS files are stored.')
    parser.add_argument('--output-file', dest='output_file', type=str, required=True,
                        help='The path to the output file')
    parser.add_argument('--add-random-beta', dest='add_random_beta', default=False, action='store_true',
                        help='If True, add a control PRS with random effect sizes to the output file.')

    args = parser.parse_args()

    pgs_table, pgs_ids = read_combine_pgs_files(args.pgs_dir, args.genotype_dir, args.add_random_beta)
    
    pgs_df = score(args.genotype_dir, pgs_table)

    pgs_df.columns = pgs_ids + ['FID', 'IID']
    pgs_df = pgs_df[['FID', 'IID'] + pgs_ids]

    makedir(osp.dirname(args.output_file))
    pgs_df.to_csv(args.output_file, sep="\t", index=False)

