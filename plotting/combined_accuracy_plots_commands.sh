#!/bin/bash

source env/moe/bin/activate

models=("MoE" "MoE-GS") # "MoE-fixed-resid-global-int")
biobanks=("ukbb" "cartagene")

for model in "${models[@]}"; do
    for bb in "${biobanks[@]}"; do
        python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --dataset train --moe-model "$model"
        python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --moe-model "$model"
        python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --restrict-to-same-biobank --dataset train --moe-model "$model"
        python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --restrict-to-same-biobank --moe-model "$model"
    done
done
