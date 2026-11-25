#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Submit a scaling ladder (the v3 rungs, identical widths, depths, ranks, learning rates
# and accumulation) with extra architecture flags, as resumable 1-hour link chains on a
# frozen code snapshot.
#   usage: scripts/submit_ladder.sh <prefix> <code_snapshot> "<extra flags>" [rungs]
#   rungs: any of 2M 10M 50M 200M (default all four)
set -euo pipefail
prefix=$1 snap=$2 extra=$3; shift 3
rungs=${*:-2M 10M 50M 200M}
cd /home/bjc75/other/biophyschem
BASE="--global-readout attn_mlp --pair-bias --test-fraction 0.05"
BIG="gpu:l40s|gpu:a40|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
HUGE="gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
for r in $rungs; do
  rung_extra=""
  case $r in
    2M)   w=144;  b=8;  a=2; k=144; lr=1e-3; acc=1; n=4;  cons=$BIG;;
    10M)  w=272;  b=12; a=4; k=272; lr=1e-3; acc=1; n=6;  cons=$BIG;;
    50M)  w=640;  b=12; a=4; k=512; lr=6e-4; acc=1; n=10; cons=$BIG;;
    200M) w=1232; b=16; a=4; k=512; lr=3e-4; acc=2; n=16; cons=$HUGE
          rung_extra="--ckpt-every-minutes 5 --eval-batch-size 1024";;  # 1-hour links: lose <= 5 min per link
    *) echo "unknown rung $r"; exit 1;;
  esac
  # A multi-partition job whose first partition lacks the constrained GPUs is rejected
  # (BadConstraints), so the large-GPU constraint gets only the partitions that have them.
  part=""; [ "$cons" = "$HUGE" ] && part="--partition=gpu_h100,gpu_b200,gpu_h200,gpu_rtx6000,scavenge_gpu"
  name=${prefix}_L${r}
  prev=""; ids=""
  for i in $(seq 1 $n); do
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    prev=$(env CODE_ROOT=$snap SHARDS="/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" \
      RUNS_DIR=runs_v3 RUN_NAME=$name WIDTH=$w BLOCKS=$b ATTN=$a RANK=$k LR=$lr HEADS=8 \
      EPOCHS=100 EXTRA="$BASE --accum-steps $acc $rung_extra $extra" \
      sbatch --parsable --job-name=$name --time=01:00:00 --constraint="$cons" $part $dep \
      scripts/slurm_train_v2.sh)
    ids="$ids $prev"
  done
  echo "$name (1-hour links, code $snap, extra: $extra):$ids" | tee -a results/v3_continuations.txt
done
