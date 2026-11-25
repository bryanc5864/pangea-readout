#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# The release ladder: 2M, 10M, 50M, 200M and 1B on every family available, with every
# improvement that passed its pre-registered test -- the strand tie (Family L, and the
# hybrid and single-strand pairs of K and J), the whole-sequence auxiliary windows,
# target revision v2.1 -- as resumable link chains on one frozen code snapshot.
#   usage: scripts/submit_release_ladder.sh <prefix> <code_snapshot> "<extra-dirs>" [rungs]
set -euo pipefail
prefix=$1 snap=$2 extra_dirs=$3; shift 3
rungs=${*:-2M 10M 50M 200M 1B}
cd /home/bjc75/other/biophyschem
COMMON="--global-readout attn_mlp --pair-bias --test-fraction 0.05 --strand-tie --g-interior-trim 5 --extra-dirs $extra_dirs --aux-shards /nfs/roberts/scratch/pi_skr2/bjc75/pangea/aux_all --aux-per-step 512"
BIG="gpu:l40s|gpu:a40|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
HUGE="gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
for r in $rungs; do
  part=""; heads=8; rx=""
  case $r in
    2M)   w=144;  b=8;  a=2; k=144; lr=1e-3; acc=2; n=6;  t=01:00:00; cons=$BIG;;
    10M)  w=272;  b=12; a=4; k=272; lr=1e-3; acc=2; n=10; t=01:00:00; cons=$BIG;;
    50M)  w=640;  b=12; a=4; k=512; lr=6e-4; acc=2; n=14; t=02:00:00; cons=$HUGE;;
    200M) w=1232; b=16; a=4; k=512; lr=3e-4; acc=4; n=14; t=02:00:00; cons=$HUGE
          rx="--ckpt-every-minutes 5 --eval-batch-size 1024";;
    1B)   w=2144; b=26; a=8; k=512; lr=2e-4; acc=8; n=30; t=02:00:00; cons=$HUGE; heads=16
          rx="--ckpt-every-minutes 5 --eval-batch-size 512";;
    *) echo "unknown rung $r"; exit 1;;
  esac
  [ "$cons" = "$HUGE" ] && part="--partition=gpu_h100,gpu_b200,gpu_h200,gpu_rtx6000,scavenge_gpu"
  name=${prefix}_L${r}
  prev=""; ids=""
  for i in $(seq 1 $n); do
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    # FIRST_DEP: a job the first link waits for (e.g. the merged cache build), afterok
    [ $i -eq 1 ] && [ -n "${FIRST_DEP:-}" ] && dep="--dependency=afterok:$FIRST_DEP"
    prev=$(env CODE_ROOT=$snap SHARDS="/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" \
      RUNS_DIR=runs_v3 RUN_NAME=$name WIDTH=$w BLOCKS=$b ATTN=$a RANK=$k LR=$lr HEADS=$heads \
      EPOCHS=100 EXTRA="$COMMON --accum-steps $acc $rx" \
      sbatch --parsable --job-name=$name --time=$t --constraint="$cons" $part $dep \
      scripts/slurm_train_v2.sh)
    ids="$ids $prev"
  done
  echo "$name (code $snap, links $t):$ids" | tee -a results/v3_continuations.txt
done
