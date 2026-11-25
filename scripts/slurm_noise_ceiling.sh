#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# The Monte Carlo noise ceiling on per-channel R-squared (see src/eval/noise_ceiling.py).
#SBATCH --job-name=pangea-noise
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/noise_%j.out
#SBATCH --error=logs/slurm/noise_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
set -e
cd /home/bjc75/other/biophyschem
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
python -W ignore src/eval/noise_ceiling.py --shards "${PANGEA_SCRATCH}/targets/shard_*.h5" \
    --n "${N:-300}" --log-channels-from runs/v2_M_attn/config.json \
    --out results/noise_ceiling_v1.json
