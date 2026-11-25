#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family L targets for the auxiliary windows (data-scaling experiment), 256 strided shards.
#SBATCH --job-name=pangea-auxL
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-255%128
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/slurm/auxL_%A_%a.out
#SBATCH --error=logs/slurm/auxL_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
set -e
cd /home/bjc75/other/biophyschem
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/aux_L
S="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$D/shard_$(printf '%04d' "$S").h5"
[ -s "$OUT" ] && { echo "present: $OUT"; exit 0; }
CODE_ROOT="${CODE_ROOT:-/home/bjc75/other/biophyschem}"
python -W ignore "$CODE_ROOT/src/data/generate_aux_targets.py" --windows "$D/aux_windows.tsv" \
    --out "$OUT" --shard "$S" --n-shards 256 --families L
