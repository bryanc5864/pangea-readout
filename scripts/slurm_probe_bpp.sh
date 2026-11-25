#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Gradient norms of the landscape and base-pair losses (src/model/probe_bpp_weight.py).
#SBATCH --job-name=probe-bpp
#SBATCH --partition=gpu,gpu_h100,gpu_b200,gpu_h200,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/probebpp_%j.out
#SBATCH --error=logs/slurm/probebpp_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -W ignore src/model/probe_bpp_weight.py --shards "/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" \
    --bpp-dir /nfs/roberts/scratch/pi_skr2/bjc75/pangea/bpp_v2 --max-shards 64 --steps 300
