#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Tier-1 quantum chemistry over the 2,080 reverse-complement-canonical hexamer contexts,
# as an array: task i runs contexts[i::N] with 4 xTB processes. Geometries are built
# once beforehand (src/qc/build_qc_fragments.py; already in $GEOM). Merge with
# src/qc/merge_xtb_shards.py.
#SBATCH --job-name=pangea-qc6
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/qc6a_%A_%a.out
#SBATCH --error=logs/slurm/qc6a_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export XTB_BIN=/home/bjc75/other/biophyschem/scripts/xtb_wrapper.sh OMP_NUM_THREADS=1
cd /home/bjc75/other/biophyschem
GEOM=${GEOM:-/home/bjc75/scratch/pangea/qc/hexamer_v2}
OUT=${OUT:-/home/bjc75/scratch/pangea/qc/hexamer_v2_shards}; mkdir -p "$OUT"
WR="${SLURM_TMPDIR:-/tmp}/qc6_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$WR"
python "${CODE_ROOT:-.}/src/qc/run_xtb.py" --index "$GEOM/qc_fragment_index.json" \
  --out "$OUT/shard_$(printf %03d "${SLURM_ARRAY_TASK_ID:-0}").json" --jobs 4 --threads 1 --vipea \
  --workroot "$WR" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:?}"
rc=$?; rm -rf "$WR"; exit $rc
