#!/bin/bash

n_sims=10
analysis_ids=("HEIGHT_MA" "LDL_MA")
biobank="ukbb"
sim_scenarios=("context_Ancestry" "context_Sex" "context_Age" "moe" "multiprs" "single_model")
heritability=(0.1 0.3 0.6)

for sim in $(seq 1 $n_sims)
do
    for ssc in "${sim_scenarios[@]}"
    do
        for h2 in "${heritability[@]}"
        do
            for analysis_id in "${analysis_ids[@]}"
            do
                scenario="sim_${sim}/${analysis_id}/${biobank}/${ssc}_h${h2}"
                mkdir -p "./log/model_fit/simulations/$scenario"
                sbatch -J "$scenario" simulations_ensemble_model/train_job.sh "$scenario"
            done
        done
    done
done
