#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Slurm array job running the electrostatics ground truth over a shard of the fragment
# library. The solver work is embarrassingly parallel, one fragment at a time, so the
# array index selects a contiguous row range of the library TSV.
#
# Usage:
#   sbatch --array=0-99 scripts/slurm_apbs_array.sh <library.tsv> <outdir> <rows_per_task> [stratum] [conditions]
#
# Example, the composition-control stratum across all nine ionic conditions:
#   sbatch --array=0-21 scripts/slurm_apbs_array.sh \
#       /home/bjc75/scratch/pangea/fragments/library_v1.tsv \
#       /home/bjc75/scratch/pangea/apbs/compctrl 30 compctrl
#
#SBATCH --job-name=pangea-pb
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/pb_%A_%a.out
#SBATCH --error=logs/slurm/pb_%A_%a.err

set -uo pipefail

LIBRARY="${1:?library tsv required}"
OUTDIR="${2:?output directory required}"
ROWS_PER_TASK="${3:-50}"
STRATUM="${4:-}"
CONDITIONS="${5:-}"

source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
cd /home/bjc75/other/biophyschem

IDX="${SLURM_ARRAY_TASK_ID:-0}"
LO=$(( IDX * ROWS_PER_TASK ))
HI=$(( LO + ROWS_PER_TASK - 1 ))

# Keep solver scratch on the node's local disk; APBS writes large OpenDX maps that are
# read back immediately and then deleted, so they should never touch shared storage.
WORKROOT="${SLURM_TMPDIR:-/tmp}/pangea_pb_${SLURM_JOB_ID:-$$}_${IDX}"
mkdir -p "$WORKROOT" "$OUTDIR" logs/slurm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname) task=${IDX} rows=${LO}-${HI} cpus=${OMP_NUM_THREADS}"
echo "library=${LIBRARY}"
echo "workroot=${WORKROOT}"

ARGS=(--library "$LIBRARY" --rows "${LO}-${HI}" --out "$OUTDIR" --workroot "$WORKROOT")
[ -n "$STRATUM" ]    && ARGS+=(--stratum "$STRATUM")
[ -n "$CONDITIONS" ] && ARGS+=(--conditions "$CONDITIONS")

python src/physics/run_fragment.py "${ARGS[@]}"
rc=$?

rm -rf "$WORKROOT"
echo "exit=${rc}"
exit "$rc"
