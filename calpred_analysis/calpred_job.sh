#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=04:00:00
#SBATCH --output=./log/calpred/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

set -euo pipefail

analysis_id=${1:?Usage: calpred_analysis/calpred_job.sh <analysis_id> [biobank ...]}

if [[ $# -gt 1 ]]; then
    biobanks=("${@:2}")
else
    biobanks=("ukbb" "cartagene")
fi

module load gcc/12.3 r/4.3.1
export R_LIBS=calpred_analysis/calpred_R_env
source env/moe/bin/activate

mkdir -p figures/calpred

ran_any=0

for bb in "${biobanks[@]}"; do
    dataset="data/harmonized_data/${analysis_id}/${bb}/train_data.pkl"

    if [[ ! -f "$dataset" ]]; then
        echo "Skipping missing dataset: $dataset"
        continue
    fi

    echo "Running CalPred: analysis_id=${analysis_id}, biobank=${bb}"
    python calpred_analysis/fit_calpred.py --dataset "$dataset"
    ran_any=1
done

if [[ "$ran_any" -eq 0 ]]; then
    echo "Error: no train_data.pkl files found for analysis_id=${analysis_id}"
    exit 1
fi
