#!/bin/bash


executor=${1:-"sbatch"}

mkdir -p ./log/evaluation/

phenotypes=("CRTN" "HDL" "LDL" "LDL_adj" "TST" "BMI" "FEV1_FVC" "HEIGHT" "LOG_TG" "TC" "URT" "T2D" "ASTHMA")

for phenotype in "${phenotypes[@]}"
do
  # Execute the evaluation script for each phenotype:
  # Use the executor variable to determine how to run the job
  # e.g., sbatch or bash
  # The script will be run with the phenotype as an argument
  # and the output will be saved in the log directory

  # Check if the executor is sbatch or bash

  if [[ "$executor" == "sbatch" ]]; then
    # Submit the job using sbatch
    sbatch -J "$phenotype" evaluation/evaluate_job.sh "$phenotype"
  else
    # Run the job directly using bash
    bash evaluation/evaluate_job.sh "$phenotype"
  fi

done
