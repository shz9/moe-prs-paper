#!/bin/bash

pgs_pheno_files=(
    "tables/multi_ancestry_prs_table.csv"
    "tables/sex_biased_prs_table.csv"
    "tables/multitrait_prs_table.csv"
)
biobanks=("ukbb" "cartagene")
n_folds=10

source env/moe/bin/activate

for file in "${pgs_pheno_files[@]}"
do
    # Loop over biobanks:
    for biobank in "${biobanks[@]}"
    do

        awk -F',' 'NR>1 {print $1 "," $3}' "$file" | sort -u | \
        while IFS=',' read -r analysis_id phenotype; do
            echo "Processing dataset for: $analysis_id | $biobank"

            python3 data_preparation/4_generate_datasets/create_datasets.py \
                --id "$analysis_id" \
                --biobank "$biobank" \
                --phenotype "$phenotype" \
                --pcs-source "1kghdp" \
                --n-folds "$n_folds"

        done
    done
done

# ----------------------------------------------------------------------
# Create control datasets with non-informative PRSs:
python data_preparation/4_generate_datasets/create_control_datasets.py \
    --analysis-file tables/multitrait_prs_table.csv \
    --analyses T1D_MT,T2D_MT,ASTHMA_MT,GOUT_MT,CAD_MT,AF_MT,HTN_MT,STR_MT,HF_MT \
    --biobank ukbb \
    --output-file tables/control_multitrait_prs_table.csv \
    --pcs-source "1kghdp" \
    --n-folds "$n_folds" \
    --create-harmonized-datasets \
    --match-reference-split \
    --target-biobanks ukbb cartagene
