#!/bin/bash


executor=${1:-"sbatch"}

mkdir -p ./log/evaluation/

mapfile -t analysis_ids < <(
  find data/harmonized_data/ -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
)

for analysis_id in "${analysis_ids[@]}"
do
  # Execute the evaluation script for each analysis:
  # Use the executor variable to determine how to run the job
  # e.g., sbatch or bash
  # The script will be run with the analysis ID as an argument
  # and the output will be saved in the log directory

  # Check if the executor is sbatch or bash

  if [[ "$executor" == "sbatch" ]]; then
    # Submit the job using sbatch
    sbatch -J "$analysis_id" evaluation/evaluate_job.sh "$analysis_id"
  else
    # Run the job directly using bash
    bash evaluation/evaluate_job.sh "$analysis_id"
  fi

done
