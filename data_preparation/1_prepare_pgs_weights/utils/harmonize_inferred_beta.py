import pandas as pd
import os.path as osp
import os
from magenpy.utils.system_utils import makedir
import argparse
import sys


def write_pgs_catalog_file(df, output_f, genome_build='GRCh37'):
    """

    To enable interfacing with the pgsc_calc pipeline, we add some relevant
    metadata in the header.

    Write the inferred BETA data to a PGS Catalog file.
    :param df: The inferred BETA data.
    :param output_f: The path to the output file.
    :param genome_build: The genome build associated with the variant coordinates.
    """

    metadata = [
        "###PGS CATALOG SCORING FILE - see https://www.pgscatalog.org/downloads/#dl_ftp_scoring for additional information",
        "#format_version=2.0",
        "##POLYGENIC SCORE (PGS) INFORMATION",
        f"#pgs_id=PGS_{osp.basename(output_f).replace('.txt', '')}",
        f"#pgs_name={osp.basename(output_f).replace('.txt', '')}",
        f"#trait_reported={phenotype}",
        f"#trait_mapped={phenotype}",
        f"#trait_efo=EFO_0000000",
        f"#genome_build={genome_build}",
        f"#variants_number={len(df)}",
        f"#weight_type=beta",
        "##SOURCE INFORMATION",
        "#pgp_id=PGP000000",
        "#citation=Empty"
    ]

    # Write the metadata to the file:
    with open(output_f, 'w') as f:
        f.writelines(line + '\n' for line in metadata)

    # Write the data to the file:
    df.to_csv(output_f, sep="\t", mode='a', index=False)

    # Compress the file:
    os.system(f"gzip {output_f}")


parser = argparse.ArgumentParser(description="""
    Convert inferred BETA files from VIPRS to PGS Catalog format.
""")

parser.add_argument('--input-file', dest='input_file', type=str, required=True,
                    help='The path to the inferred BETA file.')
parser.add_argument('--pgs-name', dest='pgs_name', type=str, required=True,
                    help='The name for the transformed PGS')
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

output_file = osp.join(output_dir, 'GRCh37', f'{args.pgs_name}.txt')

makedir(osp.dirname(output_file))
write_pgs_catalog_file(inferred_beta, output_file)

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

    new_output_file = output_file.replace('GRCh37', 'GRCh38')

    makedir(osp.dirname(new_output_file))
    write_pgs_catalog_file(inferred_beta, new_output_file, genome_build='GRCh38')
