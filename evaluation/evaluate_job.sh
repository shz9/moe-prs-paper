#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=02:00:00
#SBATCH --output=./log/evaluation/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

phenotype=${1:-"HEIGHT"}

echo "Job started at: `date`"

for dataset in data/harmonized_data/"$phenotype"/*/test_data.pkl
do
  python3 evaluation/evaluate_predictive_performance.py --test-data "$dataset" \
                                                        --cat-group-cols UMAP_Cluster Ancestry Sex \
                                                        --cont-group-cols Age \
                                                        --cont-group-bins 4 \
                                                        --pc-clusters 5
done

echo "Job finished with exit code $? at: `date`"
