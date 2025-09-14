#!/bin/bash


mkdir -p data/external_sumstats/glgc/HDL/
mkdir -p data/external_sumstats/glgc/LDL/
mkdir -p data/external_sumstats/glgc/TC/
mkdir -p data/external_sumstats/glgc/logTG/


# ---------------------------------------------------------------------------
# Download sumstats for HDL:

wget -O data/external_sumstats/glgc/HDL/AFR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/HDL_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/HDL/EAS.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/HDL_INV_EAS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/HDL/EUR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/HDL_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/HDL/AMR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/HDL_INV_HIS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/HDL/CSA.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/HDL_INV_SAS_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

# ---------------------------------------------------------------------------

# Download sumstats for LDL:

wget -O data/external_sumstats/glgc/LDL/AFR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/LDL_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LDL/EAS.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/LDL_INV_EAS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LDL/EUR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/LDL_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LDL/AMR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/LDL_INV_HIS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LDL/CSA.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/LDL_INV_SAS_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

# ---------------------------------------------------------------------------

# Download sumstats for TC:

wget -O data/external_sumstats/glgc/TC/AFR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/TC_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/TC/EAS.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/TC_INV_EAS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/TC/EUR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/TC_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/TC/AMR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/TC_INV_HIS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/TC/CSA.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/TC_INV_SAS_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

# ---------------------------------------------------------------------------

# Download sumstats for logTG:

wget -O data/external_sumstats/glgc/LOG_TG/AFR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/logTG_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LOG_TG/EAS.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/logTG_INV_EAS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LOG_TG/EUR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/logTG_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LOG_TG/AMR.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/logTG_INV_HIS_1KGP3_ALL.meta.singlevar.results.gz

wget -O data/external_sumstats/glgc/LOG_TG/CSA.txt.gz \
    https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/logTG_INV_SAS_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

