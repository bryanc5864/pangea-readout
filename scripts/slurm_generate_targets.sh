#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Stage-1 training targets: the full analytic landscape for every fragment in the
# library. 528 channels across families B, C, D, E, F and L, at two resolutions.
#
# Measured at 61 bp: 1.41 s/fragment for the analytic stack and 4.6 s for the restrained
# minimisation that Family C's torsions and Family E's hydrogen-bond distances need, so
# about 6.1 s/fragment and 325 CPU-hours over the 191,344-fragment library.
#
# Sharded 512 ways rather than 128 because this runs on `scavenge`, which preempts with
# REQUEUE: a requeued task restarts from the beginning, so the shard size sets how much
# work a preemption can cost. 512 shards is about 37 minutes each.
#
# Shards are strided (shard i takes fragments i, i+512, i+1024, ...) rather than
# contiguous, so every shard sees every stratum and a failure in one does not remove a
# whole stratum from the dataset.
#
# **The first attempt produced zero shards in 45 minutes at 256 concurrent.** The cause
# was not the filesystem, which is what it looked like: OpenMM's CPU platform defaults
# its thread count to the node's physical core count (48 here), reads that from the
# hardware rather than from the Slurm allocation, and ignores OMP_NUM_THREADS. Every
# single-core task was spawning 48 threads. `minimize.restrained_minimise` now pins the
# platform's `Threads` property, and OPENMM_CPU_THREADS=1 is exported below as a second
# line of defence.
#
# Pinned to one thread the minimisation is also *faster* -- 4.03 s against 4.84 s with 48
# threads -- because thread overhead on a 2,094-atom system exceeds any benefit. Measured
# cost is 5.45 s/fragment, 290 CPU-hours over the library.
#
# 192 concurrent rather than 256: each task still imports a multi-gigabyte conda
# environment from shared storage, which is inherent, and leaving headroom on shared
# infrastructure is cheap insurance.
#
#SBATCH --job-name=pangea-targets
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-511%192
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/targets_%A_%a.out
#SBATCH --error=logs/slurm/targets_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
set -e
cd /home/bjc75/other/biophyschem
mkdir -p logs/slurm "${PANGEA_SCRATCH}/targets"

# Keep the numerical libraries single-threaded. Each array task owns one core, and an
# OpenMP runtime that assumes it owns the node will oversubscribe 256 ways.
# OPENMM_CPU_THREADS is the one that matters here and is easy to miss: OpenMM's CPU
# platform ignores OMP_NUM_THREADS and defaults to the node's physical core count, so
# without this each single-core task spawns 48 threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENMM_CPU_THREADS=1

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS=512
OUT="${PANGEA_SCRATCH}/targets/shard_$(printf '%04d' "${SHARD}").h5"

# A requeued task would otherwise redo work that already landed.
if [ -s "${OUT}" ]; then
  echo "shard ${SHARD} already present at ${OUT}; nothing to do"
  exit 0
fi

echo "host=$(hostname) shard=${SHARD}/${N_SHARDS} out=${OUT}"
python -W ignore src/data/generate_targets.py \
    --library "${PANGEA_SCRATCH}/fragments/library_v2.tsv" \
    --out "${OUT}.partial" \
    --shard "${SHARD}" \
    --n-shards "${N_SHARDS}" \
    --n-ensemble 200 \
    --progress-every 50 \
    --workroot "${SLURM_TMPDIR:-/tmp}"

# Rename only on success, so a half-written file is never mistaken for a finished shard.
mv "${OUT}.partial" "${OUT}"
echo "shard ${SHARD} complete"
