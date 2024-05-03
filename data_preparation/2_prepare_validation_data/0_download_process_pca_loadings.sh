#!/bin/bash

mkdir -p data/gnomad_data/
mkdir -p data/hail_data/

# Download the PCA loadings tsv file from gnomad:
wget -O data/gnomad_data/release_3.1_pca_gnomad.v3.1.pca_loadings.tsv.gz \
    https://storage.cloud.google.com/gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.pca_loadings.tsv.gz

# Download the RF classifier from gnomad (.pkl format):
wget -O data/gnomad_data/release_3.1_pca_gnomad.v3.1.RF_fit.pkl \
    https://storage.cloud.google.com/gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.RF_fit.pkl

# Download the RF classifier from gnomad (.onnx format):
wget -O data/gnomad_data/release_3.1_pca_gnomad.v3.1.RF_fit.onnx \
    https://storage.cloud.google.com/gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.RF_fit.onnx

# Process the coordinates of the SNPs in the loadings files and make them available for as BED files:
source env/moe/bin/activate

python3 data_prepatation/2_prepare_validation_data/process_gnomad_pc_loadings.py

# Download hail liftover chain files:
wget -O data/hail_data/grch37_to_grch38.over.chain.gz \
    https://storage.cloud.google.com/hail-common/references/grch37_to_grch38.over.chain.gz

wget -O data/hail_data/grch38_to_grch37.over.chain.gz \
    https://storage.cloud.google.com/hail-common/references/grch38_to_grch37.over.chain.gz

# You may need to install gsutil for the following step:

gsutil -m cp -r \
  "gs://gcp-public-data--gnomad/release/3.1/pca/gnomad.v3.1.pca_loadings.ht" \
  ./data/gnomad_data/
