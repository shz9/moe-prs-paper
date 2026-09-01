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
dataset_path=${2:-""}

if [[ -n "$dataset_path" ]]; then
  if [[ ! -f "$dataset_path" ]]; then
    echo "Training dataset not found: $dataset_path" >&2
    exit 1
  fi
  datasets=("$dataset_path")
else
  # Backwards-compatible manual mode: train every fold for the analysis.
  shopt -s nullglob
  datasets=(data/harmonized_data/"$analysis_id"/*/fold_*/train_data.pkl)
fi

if [[ ${#datasets[@]} -eq 0 ]]; then
  echo "No cross-validation training datasets found for analysis: $analysis_id" >&2
  exit 1
fi

for dataset in "${datasets[@]}"
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
