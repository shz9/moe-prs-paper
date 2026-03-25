#!/bin/bash

set -e

mkdir -p data/covariates/ukbb/
mkdir -p data/misc/

source env/moe/bin/activate

# Extract covariates / phenotype data / QC filters:
python data_preparation/2_prepare_validation_data/ukbb/generate_qc_filters.py
python data_preparation/2_prepare_validation_data/ukbb/extract_medication_data.py

# Extract phenotype data:
sbatch data_preparation/2_prepare_validation_data/ukbb/extract_phenotypes.sh

# Extract genotype data:
bash data_preparation/2_prepare_validation_data/ukbb/extract_genotype_data.sh
