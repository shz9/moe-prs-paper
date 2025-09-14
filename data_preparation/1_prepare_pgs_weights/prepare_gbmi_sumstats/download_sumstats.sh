#!/bin/bash

# Download GWAS summary statistics from the Global Biobank Meta-Analysis (GBMI) project

mkdir -p data/external_sumstats/gbmi/ASTHMA/

# -------------------------------------------------------------------------------

# Download sumstats for Asthma:

wget -O data/external_sumstats/gbmi/ASTHMA/AFR.txt.gz \
    https://gbmi-sumstats.s3.amazonaws.com/Asthma_Bothsex_afr_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz

wget -O data/external_sumstats/gbmi/ASTHMA/AMR.txt.gz \
    https://gbmi-sumstats.s3.amazonaws.com/Asthma_Bothsex_amr_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz

wget -O data/external_sumstats/gbmi/ASTHMA/EAS.txt.gz \
    https://gbmi-sumstats.s3.amazonaws.com/Asthma_Bothsex_eas_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz

wget -O data/external_sumstats/gbmi/ASTHMA/EUR.txt.gz \
    https://gbmi-sumstats.s3.amazonaws.com/Asthma_Bothsex_eur_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz

wget -O data/external_sumstats/gbmi/ASTHMA/CSA.txt.gz \
    https://gbmi-sumstats.s3.amazonaws.com/Asthma_Bothsex_sas_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz