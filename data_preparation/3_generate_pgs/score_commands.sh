#!/bin/bash

# Generate scores for the UK Biobank + CARTAGENE samples using downloaded PGSs:

# FEV1/FVC
mkdir -p data/pgs_weights/EFO_0004713/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004713/GRCh37/" --output-file "data/scores/EFO_0004713/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004713/GRCh38/" --output-file "data/scores/EFO_0004713/cartagene.csv.gz"

# Height
mkdir -p data/pgs_weights/EFO_0004339/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004339/GRCh37/" --output-file "data/scores/EFO_0004339/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004339/GRCh38/" --output-file "data/scores/EFO_0004339/cartagene.csv.gz"

# BMI
mkdir -p data/pgs_weights/EFO_0004340/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004340/GRCh37/" --output-file "data/scores/EFO_0004340/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004340/GRCh38/" --output-file "data/scores/EFO_0004340/cartagene.csv.gz"

# LDL
mkdir -p data/pgs_weights/EFO_0004611/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004611/GRCh37/" --output-file "data/scores/EFO_0004611/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004611/GRCh38/" --output-file "data/scores/EFO_0004611/cartagene.csv.gz"

# HDL
mkdir -p data/pgs_weights/EFO_0004612/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004612/GRCh37/" --output-file "data/scores/EFO_0004612/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004612/GRCh38/" --output-file "data/scores/EFO_0004612/cartagene.csv.gz"

# T2D
mkdir -p data/pgs_weights/MONDO_0005148/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/MONDO_0005148/GRCh37/" --output-file "data/scores/MONDO_0005148/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/MONDO_0005148/GRCh38/" --output-file "data/scores/MONDO_0005148/cartagene.csv.gz"

# Asthma
mkdir -p data/pgs_weights/MONDO_0004979/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/MONDO_0004979/GRCh37/" --output-file "data/scores/MONDO_0004979/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/MONDO_0004979/GRCh38/" --output-file "data/scores/MONDO_0004979/cartagene.csv.gz"

# Testosterone
mkdir -p data/pgs_weights/EFO_0004908/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004908/GRCh37/" --output-file "data/scores/EFO_0004908/ukbb.csv.gz"
python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004908/GRCh38/" --output-file "data/scores/EFO_0004908/cartagene.csv.gz"

# Creatinine
mkdir -p data/pgs_weights/EFO_0004518/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004518/GRCh37/" --output-file "data/scores/EFO_0004518/ukbb.csv.gz"
#python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004518/GRCh38/" --output-file "data/scores/EFO_0004518/cartagene.csv.gz"

# Urate
mkdir -p data/pgs_weights/EFO_0004531/
python3 data_preparation/score.py --genotype_dir "../cluster_analysis_alex/data/ukbb_qc_genotypes/" --pgs-dir "data/pgs_weights/EFO_0004531/GRCh37/" --output-file "data/scores/EFO_0004531/ukbb.csv.gz"
#python3 data_preparation/score.py --genotype_dir "../../cartagene/research/prs_analysis/data/imputed_genotypes_BED/" --pgs-dir "data/pgs_weights/EFO_0004531/GRCh38/" --output-file "data/scores/EFO_0004531/cartagene.csv.gz"

