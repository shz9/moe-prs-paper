#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=08:00:00
#SBATCH --output=./log/model_fit/simulations/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Loop over training datasets in "harmonized_data" directory
# and invoke the training script for each one:

source "env/moe/bin/activate"

sim_scenario=${1:-"sim_1/HEIGHT_MA/ukbb/context_Ancestry_h0.6"}

python3 model/train_models.py --dataset-path "data/harmonized_data_simulations/"$sim_scenario"/train_data.pkl"
