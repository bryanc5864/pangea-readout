#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family B base-pair trimer tier (run_psi4.py --trimer), B3LYP-D3BJ/def2-TZVP; task i = trimers[i::N].
#SBATCH --job-name=pangea-trimer
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/trimer_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/trimer_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate /home/bjc75/project/conda_envs/qc
S="${SLURM_TMPDIR:-/tmp}/tri_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$S"; cd "$S"
OUT=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/qc/trimer_tzvp; mkdir -p "$OUT"
python -W ignore "${CODE_ROOT:?}/src/qc/run_psi4.py" --index /nfs/roberts/scratch/pi_skr2/bjc75/pangea/qc/trimer/qc_fragment_index.json \
  --basis def2-tzvp --no-dft --trimer --threads 8 --memory-gb 52 --scratch "$S" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:?}" --out "$OUT/shard_$(printf %03d "${SLURM_ARRAY_TASK_ID:-0}").json"
rc=$?; cd /; rm -rf "$S"; exit $rc
