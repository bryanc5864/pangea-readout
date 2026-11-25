#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family J oxDNA2 single-strand simulations: task i runs rows [(i+TASK_OFFSET)*ROWS, ...+ROWS) of
# LIBRARY; resumable per fragment. Runs in the oxDNA environment.
#SBATCH --job-name=pangea-famJox
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famJox_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famJox_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate /home/bjc75/project/conda_envs/oxdna
export OMP_NUM_THREADS=1
IDX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"; LO=$(( IDX * ${ROWS:?} )); HI=$(( LO + ROWS - 1 ))
WR="${SLURM_TMPDIR:-/tmp}/famJox_${SLURM_JOB_ID:-$$}_${IDX}"; mkdir -p "$WR"
python -W ignore "${CODE_ROOT:?}/src/physics/run_family_j_oxdna.py" --library "${LIBRARY:?}" --rows "${LO}-${HI}" \
  --out "${OUT:?}" --workroot "$WR" 2>&1 | grep -vE "^INFO|^DEBUG|^\s*$|WARNING|written to|Please cite|DOI|Aggregated|MB/s|^>|^\*\*\*>|seeding"
rc=${PIPESTATUS[0]}; rm -rf "$WR"; exit $rc
