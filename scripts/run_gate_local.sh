#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Run the composition-control stratum in parallel shards inside an existing allocation.
#
# WARNING: this session's interactive allocation is cpu=1,mem=8G. Running 8 solvers
# here gets them killed by the cgroup OOM killer (apbs exits -9). Use this ONLY inside
# an salloc sized for the worker count, e.g.
#   salloc -p day -c 32 --mem=64G -t 8:00:00
# and check `scontrol show job $SLURM_JOB_ID | grep AllocTRES` before trusting nproc.
#
# Usage: bash scripts/run_gate_local.sh [n_workers] [conditions]
set -uo pipefail
N_WORKERS="${1:-10}"
CONDITIONS="${2:-NaCl_150mM}"
THREADS="${3:-4}"

source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
cd /home/bjc75/other/biophyschem

LIB=/home/bjc75/scratch/pangea/fragments/library_v1.tsv
OUT=/home/bjc75/scratch/pangea/apbs/compctrl
mkdir -p "$OUT" logs/gate

N=$(awk -F'\t' 'NR>1 && $2=="compctrl"' "$LIB" | wc -l)
PER=$(( (N + N_WORKERS - 1) / N_WORKERS ))
echo "compctrl fragments: $N, workers: $N_WORKERS, rows per worker: $PER"
echo "conditions: $CONDITIONS, threads per worker: $THREADS"

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for ((w=0; w<N_WORKERS; w++)); do
  LO=$(( w * PER ))
  HI=$(( LO + PER - 1 ))
  WORKROOT="/tmp/pangea_gate_$$_${w}"
  mkdir -p "$WORKROOT"
  OMP_NUM_THREADS="$THREADS" python src/physics/run_fragment.py \
      --library "$LIB" --stratum compctrl --rows "${LO}-${HI}" \
      --out "$OUT" --workroot "$WORKROOT" --conditions "$CONDITIONS" \
      > "logs/gate/worker_${w}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} workers"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
rm -rf /tmp/pangea_gate_$$_*
echo "workers finished, $fail reported a nonzero exit"
echo "results: $(ls "$OUT"/*.npz 2>/dev/null | wc -l) npz of $N fragments"
