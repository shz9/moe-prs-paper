#!/bin/bash

mkdir -p "./log/data_preparation/2_prepare_validation_data/ukbb/"

for c in $(seq 1 22)
do
  sbatch -J "chr_$c" data_preparation/2_prepare_validation_data/ukbb/ukbb_qc_job.sh "$c"
done
