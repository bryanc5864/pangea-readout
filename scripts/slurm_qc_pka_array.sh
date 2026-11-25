#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Ring-nitrogen pKa shifts in context (Family B), ddX PCM water (domain decomposition: converged in
# its discretisation, unlike GePol PCM; DECISIONS.md), B3LYP-D3BJ/def2-TZVP, task i = contexts[i::N].
#SBATCH --job-name=pangea-pka
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/pka_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/pka_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate /nfs/roberts/scratch/pi_skr2/bjc75/conda_envs/qcdd
S="${SLURM_TMPDIR:-/tmp}/pka_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$S"; cd "$S"
OUT=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/qc/pka_tetramer; mkdir -p "$OUT"
python -W ignore "${CODE_ROOT:?}/src/qc/run_psi4.py" --index /home/bjc75/scratch/pangea/qc/tetramer/qc_fragment_index.json \
  --basis def2-tzvp --no-dft --pka --solvation ddx --ddx-lmax "${DDX_LMAX:-8}" --ddx-n-lebedev "${DDX_NLEB:-434}" \
  --threads 8 --memory-gb 40 --scratch "$S" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:?}" \
  --out "$OUT/shard_$(printf %03d "${SLURM_ARRAY_TASK_ID:-0}").json"
rc=$?; cd /; rm -rf "$S"; exit $rc
