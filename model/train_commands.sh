#!/bin/bash

mkdir -p ./log/model_fit/

analysis_csv=${1:-""}

if [[ -n "$analysis_csv" ]]; then
  if [[ ! -f "$analysis_csv" ]]; then
    echo "Error: file not found: $analysis_csv"
    exit 1
  fi

  mapfile -t analysis_ids < <(
    awk -F',' '
      NR==1 {
        for (i=1; i<=NF; i++) {
          h=$i
          gsub(/^"|"$/, "", h)
          if (h == "AnalysisID") idx=i
        }
        next
      }
      idx > 0 {
        v=$idx
        gsub(/^"|"$/, "", v)
        if (length(v) > 0) print v
      }
    ' "$analysis_csv" | sort -u
  )

  if [[ ${#analysis_ids[@]} -eq 0 ]]; then
    echo "Error: no AnalysisID values found in $analysis_csv"
    exit 1
  fi
else
  mapfile -t analysis_ids < <(
    find data/harmonized_data/ -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
  )
fi

# Canonical de-duplication before job submission.
mapfile -t analysis_ids < <(printf '%s\n' "${analysis_ids[@]}" | awk 'NF' | sort -u)

for an_id in "${analysis_ids[@]}"
do
  sbatch -J "$an_id" model/train_job.sh "$an_id"
done
