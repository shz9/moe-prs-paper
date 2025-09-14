#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=02:00:00
#SBATCH --output=./log/evaluation/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

phenotype=${1:-"HEIGHT"}

echo "Job started at: `date`"

source "env/moe/bin/activate"

for dataset in data/harmonized_data/"$phenotype"/*/*_data.pkl
do
  python3 evaluation/evaluate_predictive_performance.py --test-data "$dataset" \
                                                        --cat-group-cols Ancestry Sex
done

echo "Job finished with exit code $? at: `date`"
