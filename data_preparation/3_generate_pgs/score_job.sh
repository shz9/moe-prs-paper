#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=02:00:00
#SBATCH --output=./log/data_preparation/3_generate_pgs/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"

module load plink
source env/moe/bin/activate

phenotype=${1:-"HEIGHT"}

mkdir -p "data/scores/$phenotype/"
python3 data_preparation/3_generate_pgs/score.py \
          --genotype_dir "data/ukbb_qc_genotypes/" \
          --pgs-dir "data/pgs_weights/$phenotype/GRCh37/" \
          --output-file "data/scores/$phenotype/ukbb.csv.gz"

python3 data_preparation/3_generate_pgs/score.py \
          --genotype_dir "data/ukbb_qc_cartagene/" \
          --pgs-dir "data/pgs_weights/$phenotype/GRCh38/" \
          --output-file "data/scores/$phenotype/cartagene.csv.gz"

echo "Job finished with exit code $? at: `date`"
