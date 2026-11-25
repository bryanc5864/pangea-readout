#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family M explicit-fold models: task i runs rows [i*ROWS, (i+1)*ROWS) of WINDOWS (TSV with
# frag_id, sequence[, modelled]); resumable per fragment. ONLY_MODELLED=1 skips rows whose
# modelled column is 0.
#SBATCH --job-name=pangea-famM
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famM_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famM_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
export PANGEA_G4_TEMPLATES=/home/bjc75/other/biophyschem/data/g4_templates_v1
IDX="${SLURM_ARRAY_TASK_ID:-0}"; LO=$(( IDX * ${ROWS:?} )); HI=$(( LO + ROWS - 1 ))
WR="${SLURM_TMPDIR:-/tmp}/famM_${SLURM_JOB_ID:-$$}_${IDX}"; mkdir -p "$WR"
EXTRA=""; [ "${ONLY_MODELLED:-0}" = "1" ] && EXTRA="--only-modelled"
python -W ignore "${CODE_ROOT:?}/src/physics/run_family_m.py" --windows "${WINDOWS:?}" --rows "${LO}-${HI}" \
  --out "${OUT:?}" --workroot "$WR" $EXTRA
rc=$?; rm -rf "$WR"; exit $rc
