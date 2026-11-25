#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Run one evaluation script on a CPU node with enough memory for the in-memory dataset.
# Usage: sbatch scripts/slurm_eval_cpu.sh <script.py> [args...]
#SBATCH --job-name=pangea-eval
#SBATCH --partition=day,scavenge
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
export OMP_NUM_THREADS=4
python -W ignore "$@"
