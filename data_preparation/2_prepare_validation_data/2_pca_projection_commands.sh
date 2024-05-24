#!/bin/bash

# Activate the Hail environment:
module load java/11
source env/hail/bin/activate

mkdir -p "data/covariates/"

python3 data_preparation/2_prepare_validation_data/pca_projection.py \
        --bed-files "data/cartagene_qc_genotypes/*.bed" \
        --pca-loadings "data/gnomad_data/gnomad.v3.1.pca_loadings.ht" \
        --covar-file "data/covariates/cartagene/covars_cartagene_pcs.txt" \
        --rf-model "data/gnomad_data/gnomad.v3.1.RF_fit.onnx" \
        --output-dir "data/covariates/cartagene/" \
        --genotype-ref-genome "GRCh38"

python3 data_preparation/2_prepare_validation_data/pca_projection.py \
        --bed-files "../gnomad_ukbb_genotypes/*.bed" \
        --pca-loadings "data/gnomad_data/gnomad.v3.1.pca_loadings.ht" \
        --covar-file "data/covariates/ukbb/covars_ukbb_pcs.txt" \
        --rf-model "data/gnomad_data/gnomad.v3.1.RF_fit.onnx" \
        --output-dir "data/covariates/ukbb/" \
        --liftover-chain "/home/szabad/projects/ctb-sgravel/data/liftover_chains/hail/grch38_to_grch37.over.chain.gz" \
        --genotype-ref-genome "GRCh37"

