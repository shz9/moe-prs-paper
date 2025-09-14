#!/bin/bash

# !!!! IMPORTANT !!!!
# Run this script from the root of the repository from the login node and BEFORE submitting the pipeline jobs.
# This will set up all the requirements for running the nextflow pipeline offline on the compute nodes.
# !!!!!!!!!!!!!!!!!!!

# Load required modules:
module load python/3.11
module load apptainer

################# Step 1: Grab the singularity containers #################
# Clone the pgsc_calc repository:
[ ! -d "pgsc_calc" ] && git clone https://github.com/PGScatalog/pgsc_calc.git
cd pgsc_calc || exit

export NXF_SINGULARITY_CACHEDIR=../pgsc_calc_requirements/singularity_containers/

# Pull the Singularity containers for the pgsc_calc pipeline:
mkdir -p $NXF_SINGULARITY_CACHEDIR
git grep 'ext.singularity*' conf/modules.config | cut -f 2 -d '=' | xargs -L 2 echo | tr -d ' ' > singularity_images.txt
cat singularity_images.txt | sed 's/oras:\/\///;s/https:\/\///;s/\//-/g;s/$/.img/;s/:/-/' > singularity_image_paths.txt
paste -d '\n' singularity_image_paths.txt singularity_images.txt | while read -r path && read -r url; do
    singularity pull --disable-cache --dir $NXF_SINGULARITY_CACHEDIR "$path" "$url"
done

################# Step 2: Download reference data #################

cd ..
mkdir -p pgsc_calc_requirements/reference_data
# Download the ancestry reference data for HGDP+1kGP:
# If the file doesn't exist, download it:
if [ ! -f pgsc_calc_requirements/reference_data/pgsc_HGDP+1kGP_v1.tar.zst ]; then
    echo "\n\n\n> Downloading the ancestry reference data for HGDP+1kGP..."
    wget https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst -O pgsc_calc_requirements/reference_data/pgsc_HGDP+1kGP_v1.tar.zst
fi

################# Step 3: Setup nextflow environment #################

echo "\n\n\n> Setting up nextflow environment..."

# Set the home directory of nextflow to current directory:
export NXF_HOME=$(pwd)/nextflow_home
mkdir -p "$NXF_HOME"

# Delete any previously setup nextflow environment:
rm -rf "$NXF_HOME"/.nextflow/
cd "$NXF_HOME" || exit

# Download the latest nextflow binary and place it in the $NXF_HOME directory:
curl -s https://get.nextflow.io | bash
chmod +x ./nextflow

# Run the test pipeline to ensure everything is working:
./nextflow run pgscatalog/pgsc_calc -profile test,singularity

cd .. || exit
