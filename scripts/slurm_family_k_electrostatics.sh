#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family K electrostatics on A-form conformations (RNA duplex and both DNA-RNA hybrids): task i
# runs rows [(i+TASK_OFFSET)*ROWS, ...+ROWS) of LIBRARY (frag_id, sequence); resumable per
# fragment and class.
#SBATCH --job-name=pangea-famK
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famK_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famK_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
IDX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"; LO=$(( IDX * ${ROWS:?} )); HI=$(( LO + ROWS - 1 ))
WR="${SLURM_TMPDIR:-/tmp}/famK_${SLURM_JOB_ID:-$$}_${IDX}"; mkdir -p "$WR"
python -W ignore "${CODE_ROOT:?}/src/physics/run_k_electrostatics.py" --library "${LIBRARY:?}" --rows "${LO}-${HI}" \
  --out "${OUT:?}" --workroot "$WR"
rc=$?; rm -rf "$WR"; exit $rc
