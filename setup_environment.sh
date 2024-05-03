#!/bin/bash
# This script creates the python environment for running the scripts
# on the cluster.

mkdir -p env

echo "========================================================"
echo "Setting up environment for MoE project..."

module load StdEnv/2020
module load python/3.8
python --version

# Create environment with latest version of VIPRS:
rm -rf env/moe/
python -m venv env/moe/
source env/moe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "========================================================"
echo "Setting up environment for Hail..."

python -m venv env/hail/
source env/hail/bin/activate
python -m pip install --upgrade pip
python -m pip install hail

echo "Done!"
