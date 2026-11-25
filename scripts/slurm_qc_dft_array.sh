#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# QC tiers 2 and 3 over the tetramer contexts: task i runs contexts[i::N] of one geometry
# library (GEOM: tetramer = ground state; tetramer_soft_plus / _soft_minus = +-1 sigma
# along the central step's softest cgNA+ mode). DFT_ARGS selects the quantities, e.g.
# "--ipea --sapt" for the ground state and "" (stacking and H-bonds only) for the others.
#SBATCH --job-name=pangea-dft
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/dft_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/dft_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate /home/bjc75/project/conda_envs/qc
S="${SLURM_TMPDIR:-/tmp}/psi4_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$S"
OUT=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/qc/dft_${GEOM:?}; mkdir -p "$OUT"
python -W ignore "${CODE_ROOT:?}/src/qc/run_psi4.py" --index "/home/bjc75/scratch/pangea/qc/${GEOM}/qc_fragment_index.json" \
  --basis "${BASIS:-def2-svp}" ${DFT_ARGS:-} --threads 8 --memory-gb 40 --scratch "$S" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:?}" \
  --out "$OUT/shard_$(printf %03d "${SLURM_ARRAY_TASK_ID:-0}").json"
rc=$?; rm -rf "$S"; exit $rc
