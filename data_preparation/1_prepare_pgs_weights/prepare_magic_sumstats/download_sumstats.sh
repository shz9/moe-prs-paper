#!/bin/bash

mkdir -p data/external_sumstats/magic_sumstats/HbA1c/

# Download sumstats for HbA1c:

wget -O data/external_sumstats/magic_sumstats/HbA1c/AA.txt.gz \
    http://magicinvestigators.org/downloads/MAGIC1000G_HbA1c_AA.tsv.gz

wget -O data/external_sumstats/magic_sumstats/HbA1c/EAS.txt.gz \
    http://magicinvestigators.org/downloads/MAGIC1000G_HbA1c_EAS.tsv.gz

wget -O data/external_sumstats/magic_sumstats/HbA1c/EUR.txt.gz \
    http://magicinvestigators.org/downloads/MAGIC1000G_HbA1c_EUR.tsv.gz

wget -O data/external_sumstats/magic_sumstats/HbA1c/AMR.txt.gz \
    http://magicinvestigators.org/downloads/MAGIC1000G_HbA1c_HISP.tsv.gz

wget -O data/external_sumstats/magic_sumstats/HbA1c/CSA.txt.gz \
    http://magicinvestigators.org/downloads/MAGIC1000G_HbA1c_SAS.tsv.gz

