#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family M v3, stage 2 on CPUs: APBS on every built structure; channels. Task i runs rows
# [(i+TASK_OFFSET)*ROWS, ...+ROWS) of WINDOWS whose stage-1 record exists; resumable.
#SBATCH --job-name=pangea-famMsolve
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famMsolve_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famMsolve_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PANGEA_G4_TEMPLATES=/home/bjc75/other/biophyschem/data/g4_templates_v2
IDX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"; LO=$(( IDX * ${ROWS:?} )); HI=$(( LO + ROWS - 1 ))
WR="${SLURM_TMPDIR:-/tmp}/famMs_${SLURM_JOB_ID:-$$}_${IDX}"; mkdir -p "$WR"
python -W ignore "${CODE_ROOT:?}/src/physics/run_family_m3.py" --windows "${WINDOWS:?}" --rows "${LO}-${HI}" \
  --out "${OUT:?}" --stage solve --workroot "$WR"
rc=$?; rm -rf "$WR"; exit $rc
