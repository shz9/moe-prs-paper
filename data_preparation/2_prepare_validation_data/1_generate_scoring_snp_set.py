import pandas as pd
import glob
from tqdm import tqdm
from magenpy.utils.system_utils import makedir


genome_builds = ['GRCh37', 'GRCh38']

for gb in genome_builds:

    df_gb = None

    # Loop over all the scoring files and extract the SNP positions from them:
    # We want a union of all the SNP positions:
    pgs_files = glob.glob(f"data/pgs_weights/*/{gb}/*.txt.gz")
    for f in tqdm(pgs_files, total=len(pgs_files), desc=f"Reading {gb} PGS files"):

        df = pd.read_csv(f, sep="\t", comment='#', compression='gzip')

        if 'hm_pos' in df.columns:
            df = df[['hm_chr', 'hm_pos']].copy().rename(columns={'hm_chr': 'chr_name', 'hm_pos': 'chr_position'})
            df.dropna(inplace=True)

            # Filter to only autosomes:
            transform_chr = df['chr_name'].apply(lambda x: str(int(float(x)))
                if str(x).replace('.', '', 1).isdigit() else x
            )
            df = df.loc[transform_chr.isin(list(map(str, range(1, 23))))]

            # Cast to integers:
            df['chr_position'] = df['chr_position'].astype(int)
            df['chr_name'] = df['chr_name'].astype(int)
        else:
            df = df[['chr_name', 'chr_position']].copy()

        if df_gb is None:
            df_gb = df
        else:
            df_gb = df_gb.merge(df, how='outer')

    # Read the gnomad PC coordinates for the current genome build:
    gnomad_pc_coord = pd.read_csv(f"data/gnomad_data/gnomad_pca_snps_{gb}.bed",
                                  sep="\t", header=None, usecols=[0, 1])
    gnomad_pc_coord.columns = ['chr_name', 'chr_position']

    # Merge the gnomad PC coordinates with the SNP positions from the scoring files:
    df_gb = df_gb.merge(gnomad_pc_coord, how='outer')

    # Turn into a BED file:
    df_gb = df_gb[['chr_name', 'chr_position', 'chr_position']]

    print(f"> Total number of unique SNPs / SNP positions: {len(df_gb)}")

    makedir(f"data/snp_sets/")
    # Save the merged SNP positions to file:
    df_gb.to_csv(f"data/snp_sets/{gb}.bed", sep="\t", index=False, header=False)
