#!/bin/bash

biobanks=("ukbb" "cartagene")
phenotypes=("ASTHMA" "CRTN" "HDL" "LDL" "T2D" "TST" "BMI" "FEV1_FVC" "HEIGHT" "LOG_TG" "TC" "URT")
prop_test=0.3  # Proportion of samples to use for testing

# Loop over phenotypes:
for phenotype in "${phenotypes[@]}"
do
  # Loop over biobanks:
  for biobank in "${biobanks[@]}"
  do
    python3 data_preparation/create_datasets.py --biobank "$biobank" --phenotype "$phenotype" --pcs-source "gnomad" --prop-test "$prop_test"
  done
done
