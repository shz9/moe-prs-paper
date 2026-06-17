#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --job-name=plot_all_sections
#SBATCH --output=log/plotting.out
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

source env/moe/bin/activate

python plotting/section_1_figures.py
python plotting/section_2_figures.py
python plotting/section_3_figures.py
python plotting/section_4_figures.py
