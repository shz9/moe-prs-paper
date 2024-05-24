#!/bin/bash

mkdir -p ./log/model_fit/
phenotypes=("ASTHMA" "CRTN" "HDL" "LDL" "T2D" "TST" "BMI" "FEV1_FVC" "HEIGHT" "LOG_TG" "TC" "URT")

for phenotype in "${phenotypes[@]}"
do
  sbatch -J "$phenotype" model/train_job.sh "$phenotype"
done
