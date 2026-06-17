#!/bin/bash

source env/moe/bin/activate

models=("MoE-GS" "MoE" "MoE-fixed-resid" "TorchMoEPRS" "TorchMoEPRS-ensemble")
biobanks=("ukbb" "cartagene")

for model in "${models[@]}"; do
    for bb in "${biobanks[@]}"; do
        python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --moe-model "$model" --metric-kind "incremental_vs_ref"
        python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --restrict-to-same-biobank --moe-model "$model" --metric-kind "incremental_vs_ref"
    done
done
