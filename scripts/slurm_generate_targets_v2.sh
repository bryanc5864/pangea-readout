#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Stage-1 targets, second generation: all eight families (B, C, D, E, F, G, H, L), the
# sourced parameters (nearest-neighbour heat capacities, the Z-Hunt table, the DNA fold),
# the two-start minimisation that removes the alpha/gamma crankshaft traps, and the
# relaxed cgNA+ ensemble on a deterministic 10 percent of fragments.
#
# 1024 shards rather than 512 because this runs on `scavenge`, which preempts with
# REQUEUE and a requeued task restarts its shard: the shard size bounds what one
# preemption can cost. Per-fragment cost and the scheduling notes are in
# slurm_generate_targets.sh, whose lessons (OpenMM thread pinning in particular) apply
# unchanged.
#
#SBATCH --job-name=pangea-tgt2
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-1023%192
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/tgt2_%A_%a.out
#SBATCH --error=logs/slurm/tgt2_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
set -e
cd /home/bjc75/other/biophyschem
mkdir -p logs/slurm "${PANGEA_SCRATCH}/targets_v2"

# Keep the numerical libraries single-threaded. Each array task owns one core, and an
# OpenMP runtime that assumes it owns the node will oversubscribe 256 ways.
# OPENMM_CPU_THREADS is the one that matters here and is easy to miss: OpenMM's CPU
# platform ignores OMP_NUM_THREADS and defaults to the node's physical core count, so
# without this each single-core task spawns 48 threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENMM_CPU_THREADS=1

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS=1024
OUT="${PANGEA_SCRATCH}/targets_v2/shard_$(printf '%04d' "${SHARD}").h5"

# A requeued task would otherwise redo work that already landed.
if [ -s "${OUT}" ]; then
  echo "shard ${SHARD} already present at ${OUT}; nothing to do"
  exit 0
fi

echo "host=$(hostname) shard=${SHARD}/${N_SHARDS} out=${OUT}"
# Run from a frozen copy of the code (CODE_ROOT, set at submission) so that edits to the
# working tree during a multi-hour array cannot give shards written by different code.
# The copy carries a sha256 manifest of every source file.
CODE_ROOT="${CODE_ROOT:-/home/bjc75/other/biophyschem}"
echo "code root: ${CODE_ROOT}"
python -W ignore "${CODE_ROOT}/src/data/generate_targets.py" \
    --library "${PANGEA_SCRATCH}/fragments/library_v2.tsv" \
    --out "${OUT}.partial" \
    --shard "${SHARD}" \
    --n-shards "${N_SHARDS}" \
    --n-ensemble 3200 \
    --ensemble-fraction 0.10 \
    --progress-every 25 \
    --workroot "${SLURM_TMPDIR:-/tmp}"

# Rename only on success, so a half-written file is never mistaken for a finished shard.
mv "${OUT}.partial" "${OUT}"
echo "shard ${SHARD} complete"
