#!/bin/bash

mkdir -p ./log/model_fit/
phenotypes=("CRTN" "HDL" "LDL" "LDL_adj" "TST" "BMI" "FEV1_FVC" "HEIGHT" "LOG_TG" "TC" "URT" "T2D" "ASTHMA")

for phenotype in "${phenotypes[@]}"
do
  sbatch -J "$phenotype" model/train_job.sh "$phenotype"
done
