#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=02:00:00
#SBATCH --output=./log/data_preparation/2_prepare_validation_data/cartagene/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

module load plink

CHR=${1:-22}  # Chromosome number (default 22)
OUTPUT_DIR=${2:-"data/cartagene_qc_genotypes"}
snp_keep="data/keep_files/scoring_snps.txt"

mkdir -p $"$OUTPUT_DIR"

plink2 --vcf "$HOME/projects/ctb-sgravel/cartagene/research/flagship_project/processed_data/imputed_genotypes/chr${CHR}.vcf.gz" \
      --make-bed \
      --allow-no-sex \
      --extract "$snp_keep" \
      --hard-call-threshold 0.1 \
      --out "$OUTPUT_DIR/chr_${CHR}"


echo "Job finished with exit code $? at: `date`"
