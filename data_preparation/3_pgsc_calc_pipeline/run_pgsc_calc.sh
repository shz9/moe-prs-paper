#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=7GB
#SBATCH --time=08:00:00
#SBATCH --output=./log/pgsc_pipeline/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"

module load python/3.11
module load java
module load apptainer

# Options set by user:
biobank=${1:-"cartagene"}
target_build=${2:-"GRCh38"}
scorefiles=${3:-"data/pgs_weights/*/$target_build/*.txt.gz"}
output_dir=${4:-"data/pgsc_calc_scores/$biobank/"}
min_overlap=${5:-0.75}

# Set nextflow environment variables:
export NXF_ANSI_LOG=false
export NXF_OPTS="-Xms500M -Xmx2G"
export NXF_OFFLINE='true'
export NXF_HOME=$(pwd)/nextflow_home
export NXF_SINGULARITY_CACHEDIR=$(pwd)/pgsc_calc_requirements/singularity_containers/
export NXF_WORK=$(pwd)/nextflow_home/work/"$biobank"

mkdir -p "$NXF_WORK"
mkdir -p "$output_dir"

# Run the pipeline:
"$NXF_HOME"/nextflow run pgscatalog/pgsc_calc \
            -profile singularity \
            -offline \
            -c data_preparation/3_pgsc_calc_pipeline/custom.config \
            --input "pgsc_calc_requirements/samplesheets/${biobank}.csv" \
            --target_build "$target_build" \
            --scorefile "$scorefiles" \
            --outdir "$output_dir" \
            --verbose \
            --min_overlap "$min_overlap" \
            --run_ancestry "$(pwd)/pgsc_calc_requirements/reference_data/pgsc_HGDP+1kGP_v1.tar.zst"

echo "Job finished with exit code $? at: `date`"
