#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# One PANGEA training run, parameterised by environment variables so the scaling study
# (PLAN.md: "train at 2, 10, 50, 200 and 1,000 million parameters and plot per-channel
# accuracy against capacity") is one script with many configurations rather than many
# scripts that drift apart.
#
#   RUN_NAME WIDTH BLOCKS RANK ATTN EPOCHS BETA BATCH LR SPLIT EXTRA
#
# Walltime defaults to one hour: short requests backfill into the partially allocated GPU
# nodes, which is what got the first run scheduled in minutes after hours of waiting at
# eight. Resume is on, so a longer run simply continues across resubmissions.
#SBATCH --job-name=pangea-v2
#SBATCH --partition=gpu,gpu_h100,gpu_b200,gpu_h200,gpu_rtx6000,scavenge_gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/v2_%x_%j.out
#SBATCH --error=logs/slurm/v2_%x_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
mkdir -p runs logs/slurm
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# CODE_ROOT, when set, is a frozen snapshot (scripts/freeze_code.sh): every link of a
# chain then runs the same code even while the working tree changes underneath it.
python -W ignore "${CODE_ROOT:-.}/src/model/train.py" \
    --shards "${SHARDS:-${PANGEA_SCRATCH}/targets/shard_*.h5}" \
    --out "${RUNS_DIR:-runs}/${RUN_NAME}" \
    --width "${WIDTH:-192}" --blocks "${BLOCKS:-12}" --rank "${RANK:-64}" \
    --attn-layers "${ATTN:-0}" --attn-heads "${HEADS:-8}" \
    --epochs "${EPOCHS:-100}" --beta "${BETA:-0.5}" \
    --batch-size "${BATCH:-512}" --lr "${LR:-1e-3}" \
    --split "${SPLIT:-similarity}" --require-cuda ${EXTRA:-}
