#!/bin/bash

# Activate the Hail environment:
module load java/11
source env/hail/bin/activate

mkdir -p "data/covariates/"
python3 data_preparation/2_prepare_validation_data/pca_projection.py \
        --bed-files "data/ukbb_qc_genotypes/*.bed" \
        --pca-loadings "data/gnomad_data/gnomad.v3.1.pca_loadings.ht" \
        --covar-file "data/covariates/ukbb/covars_ukbb_pcs.txt" \
        --onnx-model "data/gnomad_data/release_3.1_pca_gnomad.v3.1.RF_fit.onnx" \
        --output-dir "data/covariates/ukbb/" \
        --genotype-ref-genome "GRCh37"

python3 data_preparation/2_prepare_validation_data/pca_projection.py \
        --bed-files "data/cartagene_qc_genotypes/*.bed" \
        --pca-loadings "data/gnomad_data/gnomad.v3.1.pca_loadings.ht" \
        --covar-file "data/covariates/cartagene/covars_cartagene_pcs.txt" \
        --onnx-model "data/gnomad_data/release_3.1_pca_gnomad.v3.1.RF_fit.onnx" \
        --output-dir "data/covariates/cartagene/" \
        --genotype-ref-genome "GRCh38"

