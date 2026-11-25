#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family M triplex / i-motif / cruciform models: task i runs rows [(i+TASK_OFFSET)*ROWS, ...+ROWS) of
# LIBRARY with FOLD (and WINDOWS for cruciforms); resumable per fragment.
#SBATCH --job-name=pangea-famMf
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famMf_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famMf_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
IDX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"; LO=$(( IDX * ${ROWS:?} )); HI=$(( LO + ROWS - 1 ))
WR="${SLURM_TMPDIR:-/tmp}/famMf_${SLURM_JOB_ID:-$$}_${IDX}"; mkdir -p "$WR"
EXTRA=""; [ -n "${WINDOWS:-}" ] && EXTRA="--windows $WINDOWS"
python -W ignore "${CODE_ROOT:?}/src/physics/run_family_m_folds.py" --library "${LIBRARY:?}" --rows "${LO}-${HI}" \
  --fold "${FOLD:?}" --out "${OUT:?}" --workroot "$WR" $EXTRA
rc=$?; rm -rf "$WR"; exit $rc
