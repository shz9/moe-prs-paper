import pandas as pd
import os.path as osp
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare samplesheet for pgsc_calc pipeline.')
    parser.add_argument('--genotype-path', dest='genotype_path',
                        default='data/ukbb_qc_genotypes/',
                        type=str, help='Path to the directory containing genotype data.')
    parser.add_argument('--dataset-name', dest='dataset_name', default='ukbb',
                        type=str, help='Name of the dataset.')
    parser.add_argument('--genotype-format', dest='genotype_format', default='bfile',
                        choices={'pfile', 'vcf', 'bfile'},
                        help='Format of the genotype data.')

    args = parser.parse_args()

    data = []

    for chrom in range(1, 23):
        data.append({
            'sampleset': args.dataset_name,
            'path_prefix': osp.join(osp.abspath(args.genotype_path), f'chr_{chrom}'),
            'chrom': chrom,
            'format': args.genotype_format
        })

    print(f"> Sample sheet for {args.dataset_name} can be found at:")
    print("\t pgsc_calc_requirements/samplesheets/ukbb.csv")
    pd.DataFrame(data).to_csv('pgsc_calc_requirements/samplesheets/ukbb.csv', index=False)
