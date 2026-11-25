#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family A production: task i solves manifest rows [i*ROWS, (i+1)*ROWS) on their long windows,
# each fragment at its own conditions (manifest), each condition at its production solute length
# (SOLUTE_JSON from the convergence study). Resumable: fragments with an .npz are skipped.
# SNAPSHOT=k (1-8) runs cgNA+ ensemble snapshot k instead of the ground state.
#SBATCH --job-name=pangea-famA
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famA_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famA_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
# TASK_OFFSET: arrays are capped at 10,000 tasks (max_array_tasks), so a long manifest is
# covered by several arrays, the second starting at task TASK_OFFSET
IDX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"; LO=$(( IDX * ${ROWS:?} )); HI=$(( LO + ROWS - 1 ))
WR="${SLURM_TMPDIR:-/tmp}/famA_${SLURM_JOB_ID:-$$}_${IDX}"; mkdir -p "$WR" "${OUT:?}"
EXTRA=""; [ -n "${SNAPSHOT:-}" ] && EXTRA="--snapshot $SNAPSHOT"
python -W ignore "${CODE_ROOT:?}/src/physics/run_fragment.py" --library "${MANIFEST:?}" --rows "${LO}-${HI}" \
  --out "$OUT" --workroot "$WR" --production-conditions --extended --reference-solve \
  --exact-fine-spacing --min-iterations 20000 --min-tolerance 1.0 \
  --solute-bp-json "${SOLUTE_JSON:?}" --resume $EXTRA
rc=$?; rm -rf "$WR"; exit $rc
