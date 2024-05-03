#!/bin/bash

mkdir -p "metadata/ukbb/"
python3 data_preparation/pca_projection.py --bed-files "../cluster_analysis_alex/data/ukbb_qc_genotypes/*.bed" --pca-loadings "data/gnomad_data/release_3.1_pca_gnomad.v3.1.pca_loadings.tsv.gz" --onnx-model "data/gnomad_data/release_3.1_pca_gnomad.v3.1.RF_fit.onnx" --output-dir "metadata/ukbb/"

mkdir -p "metadata/cartagene/"
python3 data_preparation/pca_projection.py --bed-files "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/*.bed" --pca-loadings "data/gnomad_data/release_3.1_pca_gnomad.v3.1.pca_loadings.tsv.gz" --onnx-model "data/gnomad_data/release_3.1_pca_gnomad.v3.1.RF_fit.onnx" --output-dir "metadata/cartagene/"

