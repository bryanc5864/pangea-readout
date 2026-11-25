#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Continuation links for the v3 runs. The walltimes in submit_v2_experiments.sh were
# sized for B200/H200 speeds, but runs land on whatever GPU frees first: the 2M rung ran
# at 93 s/epoch on an A40 against 12 s on a B200, which puts 100 epochs past its 2 h
# limit. Every run checkpoints and resumes, so each chain gets extra links (afterany on
# its current last link); a link that finds its run complete exits after writing any
# missing final metrics. Parameters are exactly those of the original submission.
set -euo pipefail
cd /home/bjc75/other/biophyschem
export SHARDS="/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" RUNS_DIR=runs_v3
COMMON="--global-readout attn_mlp --pair-bias --test-fraction 0.05"
BIG="gpu:l40s|gpu:a40|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
HUGE="gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
link() {  # name time constraint width blocks attn rank lr heads extra after
  local part=""; [ "$3" = "$HUGE" ] && part="--partition=gpu_h100,gpu_b200,gpu_h200,gpu_rtx6000,scavenge_gpu"
  env RUN_NAME="$1" WIDTH="$4" BLOCKS="$5" ATTN="$6" RANK="$7" LR="$8" HEADS="$9" EPOCHS=100 \
      EXTRA="$COMMON ${10}" sbatch --parsable --job-name="$1" --time="$2" \
      ${3:+--constraint=$3} $part --dependency=afterany:${11} scripts/slurm_train_v2.sh
}
more() {  # n name time constraint width blocks attn rank lr heads extra after
  local n=$1; shift; local prev=${11}; local ids=""
  for i in $(seq 1 "$n"); do prev=$(link "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "$prev"); ids="$ids $prev"; done
  echo "$1:$ids"
}
F2="--train-filter clade:bacterium,plant"; FH="--train-filter species:Homo_sapiens"
A200="--accum-steps 2 --ckpt-every-minutes 30 --eval-batch-size 1024"
A1000="--accum-steps 4 --ckpt-every-minutes 30 --eval-batch-size 512"
more 2 v3_L2M    02:00:00 ""      144 8  2 144 1e-3 8  ""     33999965
more 2 v3_L10M   03:00:00 ""      272 12 4 272 1e-3 8  ""     33999966
more 2 v3_L50M   06:00:00 "$BIG"  640 12 4 512 6e-4 8  ""     33999968
more 1 v3_L200M  08:00:00 "$HUGE" 1232 16 4 512 3e-4 8 "$A200"  33999970
more 1 v3_L1000M 08:00:00 "$HUGE" 2144 26 8 512 2e-4 16 "$A1000" 33999975
j=33999976; for rk in 4 8 16 32 64 128 256; do more 2 v3_rank$rk 03:00:00 "" 272 12 4 $rk 1e-3 8 "" $j; j=$((j+1)); done
more 1 v3_Ukingdom_2M   02:00:00 ""      144 8  2 144 1e-3 8 "$F2" 33999983
more 2 v3_Ukingdom_10M  02:00:00 ""      272 12 4 272 1e-3 8 "$F2" 33999984
more 1 v3_Ukingdom_50M  04:00:00 "$BIG"  640 12 4 512 6e-4 8 "$F2" 33999985
more 1 v3_Ukingdom_200M 06:00:00 "$HUGE" 1232 16 4 512 3e-4 8 "$F2 $A200" 33999987
more 1 v3_Uhuman_2M     02:00:00 ""      144 8  2 144 1e-3 8 "$FH" 33999988
more 2 v3_Uhuman_10M    02:00:00 ""      272 12 4 272 1e-3 8 "$FH" 33999989
more 1 v3_Uhuman_50M    04:00:00 "$BIG"  640 12 4 512 6e-4 8 "$FH" 33999990
more 1 v3_Uhuman_200M   06:00:00 "$HUGE" 1232 16 4 512 3e-4 8 "$FH $A200" 33999992
