#!/bin/bash

source env/moe/bin/activate

phenotypes=("URT" "CRTN" "TST")
strata=("M" "F")


for phenotype in "${phenotypes[@]}"
do
  # Clear the directory before fitting:
  rm -rf "data/pgs_weights/${phenotype}/"*
  for stratum in "${strata[@]}"
  do
    viprs_fit -l "data/ld/ukbb_50k_windowed/int8/chr_*/" \
              -s "data/external_sumstats/sex_stratified/${phenotype}/${stratum}.glm.linear" \
              --sumstats-format "plink2" \
              --max-iter 10000 \
              --output-dir "data/pgs_weights/${phenotype}/" \
              --output-file-prefix "${stratum}_"
    python3 data_preparation/1_prepare_pgs_weights/utils/harmonize_inferred_beta.py \
            --input-file "data/pgs_weights/${phenotype}/${stratum}_VIPRS_EM.fit.gz" \
            --pgs-name "${phenotype}_${stratum}" \
            --lift-over
    rm -rf "data/pgs_weights/${phenotype}/${stratum}_VIPRS_EM.fit.gz"
  done
  # Remove hyperparameter files:
  rm -rf "data/pgs_weights/${phenotype}/"*.hyp
done
