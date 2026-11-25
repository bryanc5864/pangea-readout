#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Replicate relaxed ensembles for the measured noise ceiling (src/eval/ensemble_replicate.py).
#SBATCH --job-name=ensrep
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --array=0-59
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/slurm/ensrep_%A_%a.out
#SBATCH --error=logs/slurm/ensrep_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
cd /tmp
python -I -W ignore "${CODE_ROOT:?}/src/eval/ensemble_replicate.py" --task "$SLURM_ARRAY_TASK_ID" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-60}" --per-task 2
