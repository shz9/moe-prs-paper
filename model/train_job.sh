#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=08:00:00
#SBATCH --output=./log/model_fit/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Loop over training datasets in "harmonized_data" directory
# and invoke the training script for each one:

source "env/moe/bin/activate"

phenotype=${1:-"HEIGHT"}

for dataset in data/harmonized_data/"$phenotype"/*/train_data.pkl
do
  python3 model/train_models.py --dataset-path "$dataset"
  #python3 model/train_models.py --dataset-path "$dataset" --residualize-prs
  #python3 model/train_models.py --dataset-path "$dataset" --residualize-phenotype --residualize-prs
done
