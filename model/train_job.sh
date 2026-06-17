#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=05:00:00
#SBATCH --output=./log/model_fit/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Loop over training datasets in "harmonized_data" directory
# and invoke the training script for each one:

source "env/moe/bin/activate"

analysis_id=${1:-"HEIGHT_MA"}

for dataset in data/harmonized_data/"$analysis_id"/*/train_data.pkl
do
  # If the analysis ID contains *_MT*, add PRS to gate input
  # Otherwise, skip adding PRS to gate input
  if [[ "$analysis_id" == *_MT* ]]; then
    python3 model/train_models.py \
      --dataset-path "$dataset" \
      --add-prs-to-gate
  else
    python3 model/train_models.py \
      --dataset-path "$dataset"
  fi
done
