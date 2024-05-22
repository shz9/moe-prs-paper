import pandas as pd
import glob
from magenpy.utils.system_utils import makedir
from tqdm import tqdm


genome_builds = ['GRCh37'] #, 'GRCh38']

for gb in genome_builds:

    df_gb = None

    # Loop over all the scoring files and extract the SNP positions from them:
    # We want a union of all the SNP positions:
    pgs_files = glob.glob(f"data/pgs_weights/*/{gb}/*.txt.gz")
    for f in tqdm(pgs_files, total=len(pgs_files), desc=f"Reading {gb} PGS files"):

        df = pd.read_csv(f, sep="\t", comment='#', compression='gzip')

        if df_gb is None:
            df_gb = df[['rsID']].copy()
        else:
            df_gb = df_gb.merge(df[['rsID']], how='outer')

    # Read the gnomad PC coordinates for the current genome build:
    gnomad_pc_coord = pd.read_csv(f"data/gnomad_data/gnomad.v3.1.pca_loadings_annotated.tsv.gz",
                                  sep="\t")

    # Merge the gnomad PC coordinates with the SNP positions from the scoring files:
    df_gb = df_gb.merge(gnomad_pc_coord[['rsid']], how='outer')

    print(f"> Total number of unique SNPs: {len(df_gb)}")

    makedir("data/keep_files/")
    # Save the merged SNP positions to file:
    df_gb.to_csv(f"data/keep_files/scoring_snps.txt",
                 sep="\t", index=False, header=False)
