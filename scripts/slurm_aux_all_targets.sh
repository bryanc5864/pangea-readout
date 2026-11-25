#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Whole-sequence channels of families B, C (ground state), D, E, F, L and per-bp Family L for
# the auxiliary windows: the data-scaling extension to whole-sequence channels. 512 shards.
#SBATCH --job-name=pangea-auxA
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-511%192
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/slurm/auxA_%A_%a.out
#SBATCH --error=logs/slurm/auxA_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
set -e
cd /home/bjc75/other/biophyschem
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/aux_all
S="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$D/shard_$(printf '%04d' "$S").h5"
[ -s "$OUT" ] && { echo "present: $OUT"; exit 0; }
CODE_ROOT="${CODE_ROOT:-/home/bjc75/other/biophyschem}"
python -W ignore "$CODE_ROOT/src/data/generate_aux_targets.py" --windows /nfs/roberts/scratch/pi_skr2/bjc75/pangea/aux_L/aux_windows.tsv \
    --out "$OUT" --shard "$S" --n-shards 512 --families B C D E F L --per-bp-families L
