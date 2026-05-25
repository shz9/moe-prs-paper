#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=03:00:00
#SBATCH --output=./log/evaluation/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

analysis_id=${1:-"HEIGHT_MA"}

echo "Job started at: `date`"

source "env/moe/bin/activate"

for dataset in data/harmonized_data/"$analysis_id"/*/*_data.pkl
do
    echo "Evaluating on: $dataset"
    python3 evaluation/evaluate_predictive_performance.py --test-data "$dataset" \
                                                        --cat-group-cols Ancestry Sex \
                                                        --include-coarse-ancestry
done

echo "Job finished with exit code $? at: `date`"
