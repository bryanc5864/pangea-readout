#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# One chunk of the plan-scale context convergence study (12 sequences per task, its own
# seed); CONDS and LENGTHS select the condition group. Raw cells go to OUTDIR for pooling.
#SBATCH --job-name=pangea-conv
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=logs/slurm/conv_%A_%a.out
#SBATCH --error=logs/slurm/conv_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
cd /home/bjc75/other/biophyschem
SEED=$(( ${SEED_BASE:?} + ${SLURM_ARRAY_TASK_ID:-0} ))
WR="${SLURM_TMPDIR:-/tmp}/conv_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$WR" "${OUTDIR:?}"
python -W ignore "${CODE_ROOT:?}/src/physics/context_convergence.py" --n-sequences 12 \
    --lengths ${LENGTHS:?} --conditions ${CONDS:?} --seed "$SEED" --workers 4 --workroot "$WR" \
    --out "$OUTDIR/summary_${SEED}.json" --save-raw "$OUTDIR/chunk_${SEED}.npz" ${CONV_EXTRA:-}
