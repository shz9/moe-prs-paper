#!/bin/bash

# Download PGSs for the phenotypes analyzed in the paper

source env/moe/bin/activate

python3 data_preparation/1_prepare_pgs_weights/batch_download_pgs.py
