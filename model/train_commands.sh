#!/bin/bash

# Loop over training datasets in "harmonized_data" directory
# and invoke the training script for each one:

for dataset in data/harmonized_data/*/*/train_data.pkl
do
  python3 model/train_models.py --dataset-path "$dataset" --residualize-phenotype
done
