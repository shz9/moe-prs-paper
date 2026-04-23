#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=08:00:00
#SBATCH --output=./log/model_fit/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Loop over training datasets in "harmonized_data" directory
# and invoke the training script for each one:

source "env/moe/bin/activate"

analysis_id=${1:-"HEIGHT_MA"}

for dataset in data/harmonized_data/"$analysis_id"/*/train_data.pkl
do
  python3 model/train_models.py \
    --dataset-path "$dataset" \
    --skip-moe-pytorch
done
