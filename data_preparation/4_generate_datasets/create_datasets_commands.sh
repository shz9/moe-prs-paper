#!/bin/bash

biobanks=("ukbb" "cartagene")
phenotypes=("EFO_0004713" "EFO_0004339" "EFO_0004340" "EFO_0004611" "EFO_0004612" "MONDO_0005148" "MONDO_0004979" "EFO_0004908" "EFO_0004518" "EFO_0004531")
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
