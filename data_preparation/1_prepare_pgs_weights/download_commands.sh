#!/bin/bash

# Download PGSs for the phenotypes analyzed in the paper

source env/moe/bin/activate

# Download PGSs from the PGS Catalog:
python3 data_preparation/1_prepare_pgs_weights/batch_download_pgs.py \
    --pgs-table "tables/phenotype_prs_table.csv" \
    --download-GRCh38
python3 data_preparation/1_prepare_pgs_weights/batch_download_pgs.py \
    --pgs-table "tables/multitrait_prs_table.csv" \
    --download-GRCh38

# Download external sumstats:
source data_preparation/1_prepare_pgs_weights/prepare_sex_biased_pgs/download_sumstats.sh

# Download VIPRS reference LD:
source data_preparation/1_prepare_pgs_weights/setup_viprs/download_reference_ld.sh

# Run VIPRS on sex-stratified phenotypes:
source data_preparation/1_prepare_pgs_weights/prepare_sex_biased_pgs/train_viprs.sh
