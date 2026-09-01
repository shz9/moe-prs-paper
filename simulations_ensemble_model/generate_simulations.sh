#!/bin/bash

source "env/moe/bin/activate"

heritability=(0.1 0.3 0.6)
analysis_ids=("HEIGHT_MA" "LDL_MA")
biobank="ukbb"
n_simulations=10

for h in "${heritability[@]}"; do
    for ((i=1; i<=n_simulations; i++)); do
        for analysis_id in "${analysis_ids[@]}"; do
            python3 simulations_ensemble_model/generate_simulated_datasets.py \
                --dataset "data/harmonized_data/${analysis_id}/${biobank}/full_data.pkl" \
                --h2 $h \
                --output_dir "data/harmonized_data_simulations/sim_${i}/"
        done
    done
done
