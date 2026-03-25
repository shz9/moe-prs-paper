#!/bin/bash

mkdir -p ./log/model_fit/

mapfile -t analyses_ids < <(
  find data/harmonized_data/ -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
)

for an_id in "${analyses_ids[@]}"
do
  sbatch -J "$an_id" model/train_job.sh "$an_id"
done
