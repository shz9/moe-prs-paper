#!/bin/bash

source env/moe/bin/activate

echo "Postprocessing UKBB data"

python data_preparation/3_pgsc_calc_pipeline/postprocess_pgsc_data.py \
    --biobank ukbb \
    --pgs-phenotype-table "tables/phenotype_prs_table.csv"

python data_preparation/3_pgsc_calc_pipeline/postprocess_pgsc_data.py \
    --biobank ukbb \
    --pgs-phenotype-table "tables/multitrait_prs_table.csv"

echo "= = = = = = = = = = = = = = = = = = ="

echo "Postprocessing Cartagene data"

python data_preparation/3_pgsc_calc_pipeline/postprocess_pgsc_data.py \
    --biobank cartagene \
    --pgs-phenotype-table "tables/phenotype_prs_table.csv"

python data_preparation/3_pgsc_calc_pipeline/postprocess_pgsc_data.py \
    --biobank cartagene \
    --pgs-phenotype-table "tables/multitrait_prs_table.csv"
