#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=04:00:00
#SBATCH --output=./log/gate_plotting/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Loop over the datasets in data/harmonized_data directory,
# find the relevant MoE models for each one, and then
# invoke the plot_pgs_admixture.py script to generate
# the admixture figures for each one:

source env/moe/bin/activate

train_biobank=${1:-"ukbb"}

mapfile -t analyses < <(
  find data/harmonized_data/ -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
)
sex_stratified_analyses=("LOG_TST_SEX" "URT_SEX" "LOG_CRTN_SEX" "WHR_SEX")

echo "> Processing data for models trained on ${train_biobank}..."

# Loop over the analysis datasets:
for analysis in "${analyses[@]}"
do

  if [[ "${sex_stratified_analyses[*]}" =~ "$analysis" ]]; then
      category="Sex"
  else
      category="Ancestry"
  fi

  for dataset in data/harmonized_data/"$analysis"/"$train_biobank"/test_*.pkl
  do
    for model in data/trained_models/"$analysis"/"$train_biobank"/*/Mo*.pkl
    do
      # Check that the model exists before invoking the plotting script:
      if [ ! -f "$model" ]; then
        echo "Model not found: $model"
        continue
      fi
      python3 plotting/plot_pgs_admixture.py --model "$model" \
                                             --dataset "$dataset" \
                                             --group-col "$category" \
                                             --subsample
    done
  done
done
