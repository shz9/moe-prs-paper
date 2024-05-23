#!/bin/bash
# This script creates the python environment for running the scripts
# on the cluster.

mkdir -p env

echo "========================================================"
echo "Setting up environment for MoE project..."

module load python/3.10
python --version

# Create environment with latest version of VIPRS:
rm -rf env/moe/
python -m venv env/moe/
source env/moe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

deactivate

echo "========================================================"
echo "Setting up environment for Hail..."

rm -rf env/hail/
python -m venv env/hail/
source env/hail/bin/activate
python -m pip install --upgrade pip
python -m pip install hail
python -m pip install onnxruntime

deactivate

echo "Done!"
