#!/bin/bash

# Generate scores for the UK Biobank + CARTAGENE samples using downloaded PGSs:

mkdir -p "./log/data_preparation/3_generate_pgs/"

phenotypes=("ASTHMA" "CRTN" "HDL" "LDL" "T2D" "TST" "BMI" "FEV1_FVC" "HEIGHT" "LOG_TG" "TC" "URT")

for phenotype in "${phenotypes[@]}"
do
  sbatch -J "$phenotype" data_preparation/3_generate_pgs/score_job.sh "$phenotype"
done
