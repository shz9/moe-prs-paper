#!/bin/bash

source env/moe/bin/activate

echo "Postprocessing UKBB data"
python data_preparation/3_pgsc_calc_pipeline/postprocess_pgsc_data.py --biobank ukbb
echo "Postprocessing Cartagene data"
python data_preparation/3_pgsc_calc_pipeline/postprocess_pgsc_data.py --biobank cartagene
