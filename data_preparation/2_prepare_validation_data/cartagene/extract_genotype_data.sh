#!/bin/bash

mkdir -p "./log/data_preparation/2_prepare_validation_data/cartagene/"

for c in $(seq 1 22)
do
  sbatch -J "chr_$c" data_preparation/2_prepare_validation_data/cartagene/cartagene_qc_job.sh "$c"
done
