import pandas as pd
import glob
from magenpy.utils.system_utils import makedir


genome_builds = ['GRCh37', 'GRCh38']

for gb in genome_builds:

    df_gb = None

    # Loop over all the scoring files and extract the SNP positions from them:
    # We want a union of all the SNP positions:
    for f in glob.glob(f"data/pgs_weights/*/{gb}/*.txt.gz"):

        df = pd.read_csv(f, sep="\t", compression='gzip')

        if df_gb is None:
            df_gb = df[['chr_name', 'chr_position', 'chr_position']].copy()
        else:
            df_gb = df_gb.merge(df[['chr_name', 'chr_position', 'chr_position']], how='outer')

    # Read the gnomad PC coordinates for the current genome build:
    gnomad_pc_coord = pd.read_csv(f"data/gnomad_data/gnomad_pca_snps_{gb}.bed",
                                  sep="\t", header=None, names=['chr_name', 'chr_position', 'chr_position'])

    # Merge the gnomad PC coordinates with the SNP positions from the scoring files:
    df_gb = df_gb.merge(gnomad_pc_coord, how='outer')

    print(f"> Total number of unique SNPs / SNP positions: {len(df_gb)}")

    makedir(f"data/snp_sets/")
    # Save the merged SNP positions to file:
    df_gb.to_csv(f"data/snp_sets/{gb}.bed", sep="\t", index=False, header=False)
