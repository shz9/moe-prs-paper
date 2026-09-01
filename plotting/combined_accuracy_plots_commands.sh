#!/bin/bash

source env/moe/bin/activate

models=("MoE-GS" "MoE" "TorchMoEPRS" "TorchMoEPRS-ensemble" "MoE-GS-prs-gating" "TorchMoEPRS-ensemble-prs-gating")
metric_kinds=("incremental_vs_ref" "base")
biobanks=("ukbb" "cartagene")

for model in "${models[@]}"; do
    for bb in "${biobanks[@]}"; do
        for metric_kind in "${metric_kinds[@]}"; do
            echo "Plotting $model on $bb with metric kind $metric_kind"
            python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --moe-model "$model" --metric-kind "$metric_kind"
            python plotting/combined_accuracy_plots.py --biobank "$bb" --aggregate-single-prs --restrict-to-same-biobank --moe-model "$model" --metric-kind "$metric_kind"
        done
    done
done
