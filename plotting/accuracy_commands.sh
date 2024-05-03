#!/bin/bash

# Loop over the accuracy metrics in data/evaluation directory
# and invoke the plot_predictive_performance.py script to generate
# the accuracy figures for each one:

for dataset in data/evaluation/*/*/*.csv
do
  python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
                                                  --aggregate-single-prs \
                                                  --category Ancestry UMAP_Cluster \
                                                  --restrict-to-same-biobank \
                                                  --train-dataset "ukbb/train_data_rph_rprs"

  python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
                                                  --aggregate-single-prs \
                                                  --category Ancestry UMAP_Cluster \
                                                  --restrict-to-same-biobank \
                                                  --train-dataset "ukbb/train_data_rph"

  python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
                                                  --aggregate-single-prs \
                                                  --category Ancestry UMAP_Cluster \
                                                  --restrict-to-same-biobank \
                                                  --train-dataset "ukbb/train_data_rprs"

  #python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
  #                                                --aggregate-single-prs \
  #                                                --category Ancestry UMAP_Cluster Sex
  #python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
  #                                                --aggregate-single-prs \
  #                                                --category Ancestry UMAP_Cluster Sex \
  #                                                --restrict-to-same-biobank
done
