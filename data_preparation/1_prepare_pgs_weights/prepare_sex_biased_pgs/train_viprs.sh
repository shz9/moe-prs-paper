#!/bin/bash

source env/moe/bin/activate

phenotypes=("urate" "creatinine" "testosterone")
strata=("male_all" "female_all")


for phenotype in "${phenotypes[@]}"
do
  for stratum in "${strata[@]}"
  do
    viprs_fit -l "data/ld/eur/old_format/ukbb_50k_windowed/chr_*/" \
              -s "data/external_sumstats/sex_stratified/${phenotype}/${stratum}.${phenotype}.glm.linear" \
              --sumstats-format "plink2" \
              --output-dir "data/pgs_weights/${phenotype^}/" \
              --output-file-prefix "${stratum}_"
    python3 data_preparation/1_prepare_pgs_weights/harmonize_inferred_beta.py \
            --input-file "data/pgs_weights/${phenotype^}/${stratum}_VIPRS_EM.fit.gz" \
            --lift-over
  done
done
