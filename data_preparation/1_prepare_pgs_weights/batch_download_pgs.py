import argparse
import os
import os.path as osp

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download polygenic score weights from the PGS Catalog"
    )

    parser.add_argument(
        "--pgs-table",
        dest="pgs_table",
        type=str,
        required=True,
        help="Path to a CSV file containing the column PGSCatalog_ID.",
    )

    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="data/pgs_weights/",
        help="Path to store the PGS weights.",
    )

    parser.add_argument(
        "--download-GRCh38",
        dest="grch38",
        action="store_true",
        default=False,
        help="If True, also download weight files mapped to genome build GRCh38",
    )

    args = parser.parse_args()

    # Read the file containing the polygenic scores to download:
    pgs_metadata = pd.read_csv(args.pgs_table)

    # Loop over the table and download the PGSs
    for _, row in tqdm(
        pgs_metadata.iterrows(), total=len(pgs_metadata), desc="Downloading PGSs"
    ):
        # If the PGS Catalog ID is missing or NaN, skip the download
        if row["PGSCatalog_ID"] is None or pd.isna(row["PGSCatalog_ID"]):
            continue

        output_f_name = f"{row['PGSCatalog_ID']}.txt.gz"

        if not osp.isfile(osp.join(args.output_dir, "GRCh37", output_f_name)):
            os.system(
                f"python3 data_preparation/1_prepare_pgs_weights/download_pgs.py "
                f"--pgs-id {row['PGSCatalog_ID']} --output-dir {args.output_dir}"
            )

        if args.grch38 and not osp.isfile(
            osp.join(args.output_dir, "GRCh38", output_f_name)
        ):
            os.system(
                f"python3 data_preparation/1_prepare_pgs_weights/download_pgs.py "
                f"--pgs-id {row['PGSCatalog_ID']} --output-dir {args.output_dir} --genome-build GRCh38"
            )
