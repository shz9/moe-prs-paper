#!/bin/bash

# Loop over the datasets in data/harmonized_data directory,
# find the relevant MoE models for each one, and then
# invoke the plot_pgs_admixture.py script to generate
# the admixture figures for each one:

for dataset in data/harmonized_data/EFO_0004339/ukbb/*.pkl
do
  # Extract the phenotype name from the 3rd field of the path:
  pheno=$(echo "$dataset" | cut -d/ -f3)
  echo "Processing $pheno ..."
  for model in "data/trained_models/$pheno"/ukbb/*/MoE-no*.pkl
  do
    # Check that the model exists before invoking the plotting script:
    if [ ! -f "$model" ]; then
      echo "Model not found: $model"
      continue
    fi
    python3 plotting/plot_pgs_admixture.py --model "$model" \
                                           --dataset "$dataset" \
                                           --group-col Ancestry \
                                           --subsample
  done
done
