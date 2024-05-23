import pandas as pd
import os.path as osp
from magenpy.utils.system_utils import makedir
import argparse
import sys


parser = argparse.ArgumentParser(description="""
    Convert inferred BETA files from VIPRS to PGS Catalog format.
""")

parser.add_argument('--input-file', dest='input_file', type=str, required=True,
                    help='The path to the inferred BETA file.')
parser.add_argument('--lift-over', dest='lift_over', action='store_true',
                    default=False,
                    help='If True, creates an additional version of the file with the '
                         'variant coordinates lifted over to GRCh38.')

args = parser.parse_args()

# Read the inferred BETA file
inferred_beta = pd.read_csv(args.input_file, sep="\t")
inferred_beta = inferred_beta.drop(columns=['PIP', 'VAR_BETA'])
inferred_beta.rename(columns={'CHR': 'chr_name', 'SNP': 'rsID', 'POS': 'chr_position',
                              'A1': 'effect_allele', 'A2': 'other_allele',
                              'BETA': 'effect_weight'}, inplace=True)

output_dir = osp.dirname(args.input_file)
phenotype = osp.basename(output_dir)

if f'{phenotype}/F_' in args.input_file:
    output_file = osp.join(output_dir, 'GRCh37', f'{phenotype}_F.txt.gz')
else:
    output_file = osp.join(output_dir, 'GRCh37', f'{phenotype}_M.txt.gz')

makedir(osp.dirname(output_file))
inferred_beta.to_csv(output_file, sep="\t", index=False)

if args.lift_over:

    sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
    from utils import liftover_coordinates

    inferred_beta['chr_name'], inferred_beta['chr_position'] = liftover_coordinates(inferred_beta,
                                                                                    chr_col='chr_name',
                                                                                    pos_col='chr_position')

    variants_to_drop = inferred_beta['chr_position'] == -1

    # print number of variants that could not be lifted over:
    print(f"Number of variants that could not be lifted over: {variants_to_drop.sum()}")
    inferred_beta = inferred_beta.loc[~variants_to_drop]
    makedir(osp.dirname(output_file.replace('GRCh37', 'GRCh38')))
    inferred_beta.to_csv(output_file.replace('GRCh37', 'GRCh38'), sep="\t", index=False)
