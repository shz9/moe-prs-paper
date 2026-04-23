#!/bin/bash

pgs_pheno_files=(
    "tables/multi_ancestry_prs_table.csv"
    "tables/sex_biased_prs_table.csv"
    "tables/multitrait_prs_table.csv"
)
biobanks=("ukbb" "cartagene")
prop_test=0.3  # Proportion of samples to use for testing

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
                --prop-test "$prop_test"

        done
    done
done
