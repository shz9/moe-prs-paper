#!/bin/bash

source env/moe/bin/activate

for sumstats_f in data/external_sumstats/glgc/*/*.txt.gz; do

    # Extract the directory name (e.g., glgc/HDL/)
    dir_name=$(dirname "$sumstats_f")

    # Extract the phenotype name (e.g., HDL)
    phenotype=$(basename "$dir_name")

    # Extract the ancestry from the filename (e.g., AFR, EAS, EUR, AMR, CSA)
    ancestry=$(basename "$sumstats_f" .txt.gz)

    viprs_fit -s "$sumstats_f" \
        -l "../viprs-benchmarks-paper/data/ld_xarray/hq_imputed_variants_hm3/${ancestry}/block/int16/chr_*/" \
        --output-dir "data/pgs_weights/${phenotype}/" \
        --sumstats-format "custom" \
        --custom-sumstats-mapper "rsID=SNP,CHROM=CHR,POS_b37=POS,REF=A2,ALT=A1,POOLED_ALT_AF=MAF,EFFECT_SIZE=BETA,pvalue=PVAL" \
        --output-file-prefix "GLGC_${ancestry}_"

    python3 data_preparation/1_prepare_pgs_weights/utils/harmonize_inferred_beta.py \
            --input-file "data/pgs_weights/${phenotype}/GLGC_${ancestry}_VIPRS_EM.fit.gz" \
            --pgs-name "PGS_VIPRS_${phenotype}_${ancestry}" \
            --lift-over

done