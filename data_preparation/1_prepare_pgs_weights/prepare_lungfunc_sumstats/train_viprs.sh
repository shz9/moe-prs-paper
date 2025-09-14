#!/bin/bash

source env/moe/bin/activate

for sumstats_f in data/external_sumstats/lungfunc_sumstats/*/*.txt.gz; do

    # Extract the directory name (e.g., lungfunc_sumstats/FEV1_FVC/)
    dir_name=$(dirname "$sumstats_f")

    # Extract the phenotype name (e.g., FEV1_FVC)
    phenotype=$(basename "$dir_name")

    # Extract the ancestry from the filename (e.g., AFR, EAS, EUR, AMR, CSA)
    ancestry=$(basename "$sumstats_f" .txt.gz)

    viprs_fit -s "$sumstats_f" \
        -l "../viprs-benchmarks-paper/data/ld_xarray/hq_imputed_variants_hm3/${ancestry}/block/int16/chr_*/" \
        --output-dir "data/pgs_weights/${phenotype}/" \
        --sumstats-format "custom" \
        --custom-sumstats-mapper "variant_id=SNP,chromosome=CHR,base_pair_location=POS,other_allele=A2,effect_allele=A1,effect_allele_frequency=MAF,beta=BETA,se=SE,p_value=PVAL" \
        --output-file-prefix "LUNG_${ancestry}_"

    python3 data_preparation/1_prepare_pgs_weights/utils/harmonize_inferred_beta.py \
            --input-file "data/pgs_weights/${phenotype}/LUNG_${ancestry}_VIPRS_EM.fit.gz" \
            --pgs-name "PGS_VIPRS_${phenotype}_${ancestry}" \
            --lift-over

done