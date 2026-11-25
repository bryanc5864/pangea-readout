#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Multi-GPU (data-parallel) training for the top rungs of the scaling ladder. Same
# trainer and arguments as slurm_train_v2.sh; torchrun starts one process per GPU and
# train.py switches to DistributedDataParallel when WORLD_SIZE > 1. Resumes from the
# run's checkpoint like the single-GPU script, so walltime chains and preemption are safe.
#
#SBATCH --job-name=pangea-ddp
#SBATCH --partition=gpu_b200,gpu_h200,gpu_h100,scavenge_gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --output=logs/slurm/ddp_%x_%j.out
#SBATCH --error=logs/slurm/ddp_%x_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
mkdir -p runs logs/slurm
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
NG=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node="${NG}" src/model/train.py \
    --shards "${SHARDS:-${PANGEA_SCRATCH}/targets/shard_*.h5}" \
    --out "${RUNS_DIR:-runs}/${RUN_NAME}" \
    --width "${WIDTH}" --blocks "${BLOCKS}" --rank "${HRANK}" \
    --attn-layers "${ATTN:-0}" --attn-heads "${HEADS:-8}" \
    --epochs "${EPOCHS:-100}" --beta "${BETA:-0.5}" \
    --batch-size "${BATCH:-512}" --lr "${LR:-3e-4}" \
    --split "${SPLIT:-similarity}" --require-cuda ${EXTRA:-}
