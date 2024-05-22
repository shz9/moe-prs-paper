#!/bin/bash

mkdir -p data/gnomad_data/
mkdir -p data/hail_data/

# You may need to install gsutil for the following step:

gsutil -m cp gs://gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.pca_loadings.tsv.gz ./data/gnomad_data/
gsutil -m cp gs://gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.RF_fit.pkl ./data/gnomad_data/
gsutil -m cp gs://gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.RF_fit.onnx ./data/gnomad_data/

gsutil -o GSUtil:parallel_process_count=1 -m cp -r \
  "gs://gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.pca_loadings.ht" \
  ./data/gnomad_data/

# Process the coordinates of the SNPs in the loadings files and make them available for as BED files:
module load java/11
source env/moe/bin/activate

python3 data_preparation/2_prepare_validation_data/postprocess_gnomad_pc_loadings.py
python3 data_preparation/2_prepare_validation_data/annotate_gnomad_pc_loadings.py

# Download hail liftover chain files:
gsutil -m cp gs://hail-common/references/grch37_to_grch38.over.chain.gz ./data/hail_data/
gsutil -m cp gs://hail-common/references/grch38_to_grch37.over.chain.gz ./data/hail_data/
