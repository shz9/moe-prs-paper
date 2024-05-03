
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# Read the file containing the polygenic scores to download:
pgs_metadata = pd.read_csv("tables/phenotype_prs_table.csv")

# Loop over the table and download the PGSs
for _, row in tqdm(pgs_metadata.iterrows(), total=len(pgs_metadata),
                   desc='Downloading PGSs'):

    phenotype = row['Phenotype_short']
    output_dir = f"data/pgs_weights/{phenotype}"

    # If the PGS Catalog ID is missing or NaN, skip the download
    if row['PGSCatalog_ID'] is None or np.isnan(row['PGSCatalog_ID']):
        continue

    os.system(f"python3 data_preparation/1_prepare_pgs_weights/download_pgs.py "
              f"--pgs-id {row['PGSCatalog_ID']} --output-dir {output_dir}")
    os.system(f"python3 data_preparation/1_prepare_pgs_weights/download_pgs.py "
              f"--pgs-id {row['PGSCatalog_ID']} --output-dir {output_dir} --genome-build GRCh38")

