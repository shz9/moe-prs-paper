#!/bin/bash

source env/moe/bin/activate

# Create the samplesheet for the pgsc_calc pipeline:
mkdir -p pgsc_calc_requirements/samplesheets/
python data_preparation/3_pgsc_calc_pipeline/cartagene/prepare_samplesheet.py

# Submit a job to perform scoring on the CARTaGENE data:
mkdir -p ./log/pgsc_pipeline/

sbatch -J "cartagene" --time "02:00:00" data_preparation/3_pgsc_calc_pipeline/run_pgsc_calc.sh
