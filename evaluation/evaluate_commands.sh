#!/bin/bash

# Loop over test sets in "harmonized_data" directory
# and invoke the evaluation script for each one:

for dataset in data/harmonized_data/*/*/test_data.pkl
do
  python3 evaluation/evaluate_predictive_performance.py --test-data "$dataset" \
                                                        --cat-group-cols UMAP_Cluster Ancestry Sex \
                                                        --cont-group-cols Age \
                                                        --cont-group-bins 4 \
                                                        --pc-clusters 5
done

