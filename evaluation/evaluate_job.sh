#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=03:00:00
#SBATCH --output=./log/evaluation/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

analysis_id=${1:-"HEIGHT_MA"}
max_jobs=${2:-2}   # number of datasets to evaluate concurrently
export analysis_id

echo "Job started at: `date`"

source "env/moe/bin/activate"

run_eval() {
    local dataset="$1"
    shift
    local -a eval_args=(--test-data "$dataset" --coarse-ancestry-only)

    if [[ "$analysis_id" == *"_SEX"* ]]; then
        eval_args+=(--cat-group-cols Sex)
    fi
    eval_args+=("$@")

    echo "Evaluating on: $dataset"
    python3 evaluation/evaluate_predictive_performance.py "${eval_args[@]}"
}
export -f run_eval

mapfile -t datasets < <(
    find data/harmonized_data/"$analysis_id" \
        -mindepth 3 -maxdepth 3 -type f -name test_data.pkl | sort
)

if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "No cross-validation test datasets found for analysis: $analysis_id" >&2
    exit 1
fi

printf '%s\n' "${datasets[@]}" | \
    xargs -I{} -P "$max_jobs" bash -c 'run_eval "$@"' _ {}

# CARTaGENE is an external held-out cohort for UKBB-trained models. Load the
# full dataset once, average individual predictions across all UKBB fold
# models, and bootstrap CARTaGENE participants for sampling uncertainty.
external_dataset="data/harmonized_data/$analysis_id/cartagene/full_data.pkl"
if [[ -f "$external_dataset" ]]; then
    mapfile -t ukbb_model_folds < <(
        find data/trained_models/"$analysis_id"/ukbb \
            -mindepth 1 -maxdepth 1 -type d -name 'fold_*' -printf '%f\n' | sort
    )

    if [[ ${#ukbb_model_folds[@]} -gt 0 ]]; then
        run_eval "$external_dataset" --all-model-folds --train-biobank ukbb
    else
        echo "No UKBB fold models found for external CARTaGENE evaluation." >&2
    fi
fi

echo "Job finished at: `date`"
