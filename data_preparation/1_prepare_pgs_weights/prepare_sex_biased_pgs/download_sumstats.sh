#!/bin/bash

# Download summary statistics for the sex-biased phenotypes
# from Zenodo:

mkdir -p data/external_sumstats/sex_stratified/

# ==============================================================================
# Download/extract the summary statistics for Urate:
wget -O data/external_sumstats/sex_stratified/Urate.tar.gz \
    https://zenodo.org/records/7222725/files/urate.tar.gz?download=1

tar -xzf data/external_sumstats/sex_stratified/Urate.tar.gz \
    -C data/external_sumstats/sex_stratified/

# Rename the files:
mv data/external_sumstats/sex_stratified/urate/female_all.urate.glm.linear \
  data/external_sumstats/sex_stratified/urate/F.glm.linear

mv data/external_sumstats/sex_stratified/urate/male_all.urate.glm.linear \
  data/external_sumstats/sex_stratified/urate/M.glm.linear

# Rename the directory:
mv data/external_sumstats/sex_stratified/urate \
  data/external_sumstats/sex_stratified/URT

rm data/external_sumstats/sex_stratified/Urate.tar.gz


# ==============================================================================
# Download/extract the summary statistics for Creatinine:

wget -O data/external_sumstats/sex_stratified/Creatinine.tar.gz \
    https://zenodo.org/records/7222725/files/creatinine.tar.gz?download=1

tar -xzf data/external_sumstats/sex_stratified/Creatinine.tar.gz \
    -C data/external_sumstats/sex_stratified/

# Rename the files:
mv data/external_sumstats/sex_stratified/creatinine/female_all.creatinine.glm.linear \
  data/external_sumstats/sex_stratified/creatinine/F.glm.linear

mv data/external_sumstats/sex_stratified/creatinine/male_all.creatinine.glm.linear \
  data/external_sumstats/sex_stratified/creatinine/M.glm.linear

# Rename the directory:
mv data/external_sumstats/sex_stratified/creatinine \
  data/external_sumstats/sex_stratified/CRTN

rm data/external_sumstats/sex_stratified/Creatinine.tar.gz

# ==============================================================================
# Download/extract the summary statistics for Testosterone:

wget -O data/external_sumstats/sex_stratified/Testosterone.tar.gz \
    https://zenodo.org/records/7222725/files/testosterone.tar.gz?download=1

tar -xzf data/external_sumstats/sex_stratified/Testosterone.tar.gz \
    -C data/external_sumstats/sex_stratified/

# Rename the files:
mv data/external_sumstats/sex_stratified/testosterone/female_all.testosterone.glm.linear \
  data/external_sumstats/sex_stratified/testosterone/F.glm.linear

mv data/external_sumstats/sex_stratified/testosterone/male_all.testosterone.glm.linear \
  data/external_sumstats/sex_stratified/testosterone/M.glm.linear

# Rename the directory:
mv data/external_sumstats/sex_stratified/testosterone \
  data/external_sumstats/sex_stratified/TST

rm data/external_sumstats/sex_stratified/Testosterone.tar.gz
