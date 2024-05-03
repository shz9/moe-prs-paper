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

rm data/external_sumstats/sex_stratified/Urate.tar.gz


# ==============================================================================
# Download/extract the summary statistics for Creatinine:

wget -O data/external_sumstats/sex_stratified/Creatinine.tar.gz \
    https://zenodo.org/records/7222725/files/creatinine.tar.gz?download=1

tar -xzf data/external_sumstats/sex_stratified/Creatinine.tar.gz \
    -C data/external_sumstats/sex_stratified/

rm data/external_sumstats/sex_stratified/Creatinine.tar.gz

# ==============================================================================
# Download/extract the summary statistics for Testosterone:

wget -O data/external_sumstats/sex_stratified/Testosterone.tar.gz \
    https://zenodo.org/records/7222725/files/testosterone.tar.gz?download=1

tar -xzf data/external_sumstats/sex_stratified/Testosterone.tar.gz \
    -C data/external_sumstats/sex_stratified/

rm data/external_sumstats/sex_stratified/Testosterone.tar.gz
