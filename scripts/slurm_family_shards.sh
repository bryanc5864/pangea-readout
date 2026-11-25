#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Side shards for a family added after the main corpus (src/data/generate_family_shards.py):
#   SHARDS (glob, quoted), FAMILIES (space-separated), OUT (directory), optional
#   PER_BP_FAMILIES (store per-bp channels only for these); CODE_ROOT a frozen
#   snapshot. Shards are strided over the array's tasks.
#SBATCH --job-name=pangea-fam
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/slurm/fam_%A_%a.out
#SBATCH --error=logs/slurm/fam_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
cd /tmp
export OMP_NUM_THREADS=1
python -I -W ignore "${CODE_ROOT:?}/src/data/generate_family_shards.py" --shards "${SHARDS:?}" \
    --families ${FAMILIES:?} ${PER_BP_FAMILIES:+--per-bp-families $PER_BP_FAMILIES} \
    --out "${OUT:?}" --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-1}"
