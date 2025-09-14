#!/bin/bash

# Download summary statistics for lung function phenotypes from Shrine et al. 2023

mkdir -p data/external_sumstats/lungfunc_sumstats/FEV1_FVC/

# Download sumstats for FEV1/FVC:

wget -O data/external_sumstats/lungfunc_sumstats/FEV1_FVC/EAS.txt.gz \
    https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90292001-GCST90293000/GCST90292631/GCST90292631.tsv.gz

wget -O data/external_sumstats/lungfunc_sumstats/FEV1_FVC/AMR.txt.gz \
    https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90292001-GCST90293000/GCST90292635/GCST90292635.tsv.gz

wget -O data/external_sumstats/lungfunc_sumstats/FEV1_FVC/EUR.txt.gz \
    https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90292001-GCST90293000/GCST90292611/GCST90292611.tsv.gz

wget -O data/external_sumstats/lungfunc_sumstats/FEV1_FVC/AFR.txt.gz \
    https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90292001-GCST90293000/GCST90292623/GCST90292623.tsv.gz

wget -O data/external_sumstats/lungfunc_sumstats/FEV1_FVC/CSA.txt.gz \
    https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90292001-GCST90293000/GCST90292627/GCST90292627.tsv.gz

