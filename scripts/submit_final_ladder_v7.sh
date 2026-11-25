#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# The final ladder (v7): 2M, 10M, 50M, 200M and 1B on every physics family the plan names
# and every improvement that passed its test. Inputs: scripts/finalize_v7_inputs.sh. Over v6 it adds
#   * Family W (10,240 bp fragments: window ladder 10-5,000 bp and the wide branch's targets;
#     --wide-shards switches the 8.2 kb wide branch and the pooled window heads on);
#   * side families Kes (Family K A-form electrostatics), Ax (Family A opening and dielectric
#     response), Glib (Family G hydration MD, docking, trimer DFT), Nes (Family N modification
#     electrostatics), FrGr (R-loop energetics, hydrolytic damage rates), Jox (Family J oxDNA2);
#   * Family M regenerated: quadruplexes of every class (v3, GPU MD ensembles), triplexes,
#     i-motifs, cruciforms.
# PROVISIONAL=1 adds --allow-provisional (69 channels with placeholder physics,
# results/provisional_channels.json; off unless the user decides otherwise).
# v6 added over v5:
#   * Family A, electrostatics, on 121 bp long windows (labels on the central 21 bp;
#     --long-shards), with the Family A strand pairs in the tie;
#   * Family M, explicit quadruplex folds (corpus side shards and quadruplex windows);
#   * Family J revision 2 (oxDNA2 stacking channels; famJ2 replaces famJ);
#   * Families J, N, K on the 1.02M auxiliary windows (data-limited fold families);
#   * the quadruplex windows and the rare-event windows as further auxiliary sets;
#   * Family B quantum chemistry ("Bq": DFT/SAPT tetramer table mapped onto every step).
# Family I stays out (held apart for the labelled protein-readout variant, PLAN.md).
#   usage: scripts/submit_final_ladder.sh <prefix> <code_snapshot> [rungs]
#   env:   FIRST_DEP (the merged cache build), LONG_SHARDS (Family A long-shard directory)
set -euo pipefail
prefix=$1 snap=$2; shift 2
rungs=${*:-2M 10M 50M 200M 1B}
cd /home/bjc75/other/biophyschem
G=/nfs/roberts/scratch/pi_skr2/bjc75/pangea
EXTRA_DIRS="$G/targets_v2_famK $G/targets_v2_famJ2 $G/targets_v2_famN $G/targets_v2_famM $G/targets_v2_famBq \
  $G/targets_v2_famKes $G/targets_v2_famAx $G/targets_v2_famGlib $G/targets_v2_famNes $G/targets_v2_famFrGr $G/targets_v2_famJox"
AUX="$G/aux_all,$G/fold_windows/base,$G/rare_windows/base"
# Bq is a lookup over 136 four-base-pair contexts, each seen millions of times in the main
# corpus; repeating it on the 1.08M auxiliary windows would cost ~10 GB of GPU memory for
# no new information, so it is a main-corpus side shard only.
AUX_EXTRA="$G/aux_all_famJNK $G/fold_windows/famJNK $G/rare_windows/famJNK $G/fold_windows/famM"
LONG="${LONG_SHARDS:?set LONG_SHARDS to the Family A long-shard directory}"
COMMON="--global-readout attn_mlp --pair-bias --test-fraction 0.05 --strand-tie --g-interior-trim 5 --extra-dirs $EXTRA_DIRS --aux-shards $AUX --aux-extra-dirs $AUX_EXTRA --aux-extra-optional --aux-holdout fold_g4_:0.1 --aux-per-step 512 --long-shards $LONG --long-per-step 128 --long-weight 1.0 --wide-shards $G/famW/shards --wide-per-step 2 --wide-weight 1.0"
[ "${PROVISIONAL:-0}" = 1 ] && COMMON="$COMMON --allow-provisional"
BIG="gpu:l40s|gpu:a40|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
HUGE="gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
for r in $rungs; do
  part=""; heads=8; rx=""
  case $r in
    2M)   w=144;  b=8;  a=2; k=144; lr=1e-3; acc=2; n=8;  t=01:00:00; cons=$BIG;;
    10M)  w=272;  b=12; a=4; k=272; lr=1e-3; acc=2; n=12; t=01:00:00; cons=$BIG;;
    50M)  w=640;  b=12; a=4; k=512; lr=6e-4; acc=2; n=16; t=02:00:00; cons=$HUGE;;
    200M) w=1232; b=16; a=4; k=512; lr=3e-4; acc=4; n=16; t=02:00:00; cons=$HUGE
          rx="--ckpt-every-minutes 5 --eval-batch-size 1024";;
    1B)   w=2144; b=26; a=8; k=512; lr=2e-4; acc=8; n=34; t=02:00:00; cons=$HUGE; heads=16
          rx="--ckpt-every-minutes 5 --eval-batch-size 512";;
    *) echo "unknown rung $r"; exit 1;;
  esac
  [ "$cons" = "$HUGE" ] && part="--partition=gpu_h100,gpu_b200,gpu_h200,gpu_rtx6000,scavenge_gpu"
  name=${prefix}_L${r}
  prev=""; ids=""
  for i in $(seq 1 $n); do
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
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
