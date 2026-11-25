#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Base-pair probability matrices (Family L's own folding model) for every fragment of a
# set of target shards: SHARDS (glob, quoted) and OUT (directory). Shards are split over
# the array tasks; each task folds its shards on all of its cores. CODE_ROOT points at a
# frozen code snapshot (scripts/freeze_code.sh).
#SBATCH --job-name=pangea-bpp
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/bpp_%A_%a.out
#SBATCH --error=logs/slurm/bpp_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
cd /tmp
python -I -W ignore "${CODE_ROOT:?}/src/data/generate_bpp.py" --shards "${SHARDS:?}" --out "${OUT:?}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-1}"
