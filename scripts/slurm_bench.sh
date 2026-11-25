#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#SBATCH --job-name=pangea-bench
#SBATCH --partition=gpu_b200,gpu_h200,scavenge_gpu
#SBATCH --constraint="gpu:b200|gpu:h200"
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/bench_%j.out
#SBATCH --error=logs/slurm/bench_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
python -W ignore src/model/bench.py --config runs/v2_M_attn/config.json \
    --configs 384:12:384:4 704:12:512:4 1280:16:512:4 1536:20:512:6 2048:24:512:8 2560:20:512:8
