#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# The v2 experiment set, on the regenerated targets, every run with a held-out test side
# (5 percent, similarity-grouped) that no training or selection decision sees.
#
#   ladder        PLAN.md's scaling study at 2, 10, 50, 200 and 1,000 million parameters
#   rank sweep    the head bottleneck run to full width at the 10M trunk: the smallest
#                 saturating rank is the plan's estimate of intrinsic dimensionality
#   universality  held-out-kingdom (bacteria + plants) and human-only encoders at each
#                 rung up to 200M: the universality gap against capacity
#
# Widths come from src/model/size_ladder.py on the v2 cache (W2 W10 W50 W200 W1000).
# Design: attention (4 layers, 8 at 1B) with the pair-map bias, the attention/MLP global
# readout, beta-NLL 0.5 -- each chosen by an ablation recorded in DECISIONS.md.
#
# Scheduling: every H200/B200 node was fully allocated when this was written, so the top
# rungs run on one GPU at a time as resumable chains (each link resumes from the run's
# checkpoint; a link that finds the run complete exits), with gradient accumulation so
# they fit the 80-96 GB cards too. GPU memory constraints keep large models off the
# 24-32 GB cards, where they would fail with out-of-memory.
set -euo pipefail
cd /home/bjc75/other/biophyschem
: "${W2:?}" "${W10:?}" "${W50:?}" "${W200:?}" "${W1000:?}"
export SHARDS="/home/bjc75/scratch/pangea/targets_v2/shard_*.h5"
# Run directories live on the group scratch fileset (runs_v3 -> /nfs/roberts/scratch/...):
# the home quota is 125 GiB, and a 1B checkpoint alone is about 16 GB. The targets
# directory above is itself a link to the same fileset.
export RUNS_DIR=runs_v3
COMMON="--global-readout attn_mlp --pair-bias --test-fraction 0.05"
BIG="gpu:l40s|gpu:a40|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
HUGE="gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
r() { local w=$1; echo $(( w < 512 ? w : 512 )); }
one() {   # name time constraint width blocks attn rank lr heads extra [dependency]
  local dep=""; [ -n "${11:-}" ] && dep="--dependency=afterany:${11}"
  # A constraint naming only 80-180 GB GPUs is unsatisfiable in the general `gpu`
  # partition, and a multi-partition job whose first partition cannot satisfy its
  # constraint sat in BadConstraints instead of trying the others -- so those jobs list
  # only the partitions that have such GPUs.
  local part=""; [ "$3" = "$HUGE" ] && part="--partition=gpu_h100,gpu_b200,gpu_h200,gpu_rtx6000,scavenge_gpu"
  env RUN_NAME="$1" WIDTH="$4" BLOCKS="$5" ATTN="$6" RANK="$7" LR="$8" HEADS="$9" EPOCHS=100 \
      EXTRA="$COMMON ${10:-}" sbatch --parsable --job-name="$1" --time="$2" \
      ${3:+--constraint=$3} $part $dep scripts/slurm_train_v2.sh
}
chain() { # links name time constraint width blocks attn rank lr heads extra
  local n=$1; shift; local prev=""
  for i in $(seq 1 "$n"); do prev=$(one "$@" "$prev"); echo -n "$prev "; done
}
echo "ladder:"
echo "  L2M    $(one v3_L2M    02:00:00 ""    $W2  8  2 $(r $W2)  1e-3 8 "")"
echo "  L10M   $(one v3_L10M   03:00:00 ""    $W10 12 4 $(r $W10) 1e-3 8 "")"
echo "  L50M   $(chain 2 v3_L50M   06:00:00 "$BIG"  $W50 12 4 $(r $W50) 6e-4 8 "")"
echo "  L200M  $(chain 2 v3_L200M  08:00:00 "$HUGE" $W200 16 4 $(r $W200) 3e-4 8 "--accum-steps 2 --ckpt-every-minutes 30 --eval-batch-size 1024")"
echo "  L1000M $(chain 5 v3_L1000M 08:00:00 "$HUGE" $W1000 26 8 512 2e-4 16 "--accum-steps 4 --ckpt-every-minutes 30 --eval-batch-size 512")"
echo "rank sweep at the 10M trunk (width $W10):"
for rk in 4 8 16 32 64 128 256; do
  [ "$rk" -lt "$W10" ] && echo "  r$rk    $(one v3_rank$rk 03:00:00 "" $W10 12 4 $rk 1e-3 8 "")"
done
echo "universality (train filter -> every species scored on the shared test side):"
for spec in "kingdom:clade:bacterium,plant" "human:species:Homo_sapiens"; do
  tag=${spec%%:*}; flt=${spec#*:}
  echo "  $tag 2M   $(one v3_U${tag}_2M   02:00:00 ""      $W2  8  2 $(r $W2)  1e-3 8 "--train-filter $flt")"
  echo "  $tag 10M  $(one v3_U${tag}_10M  02:00:00 ""      $W10 12 4 $(r $W10) 1e-3 8 "--train-filter $flt")"
  echo "  $tag 50M  $(one v3_U${tag}_50M  04:00:00 "$BIG"  $W50 12 4 $(r $W50) 6e-4 8 "--train-filter $flt")"
  echo "  $tag 200M $(chain 2 v3_U${tag}_200M 06:00:00 "$HUGE" $W200 16 4 $(r $W200) 3e-4 8 "--train-filter $flt --accum-steps 2 --ckpt-every-minutes 30 --eval-batch-size 1024")"
done
