#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Runs once the whole-sequence auxiliary array (aux_all) has finished: checks that every
# one of its 512 shards exists and holds its windows, then (1) submits the pre-registered
# whole-sequence data-scaling run v3_L10M_auxAll on frozen code, and (2) generates the
# base-pair matrices for the aux_all shards (needed by any model that combines the
# whole-sequence auxiliary data with base-pair supervision). Submitted with
# --dependency=afterany on the array, on a CPU partition.
#SBATCH --job-name=after-auxall
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/afterauxall_%j.out
#SBATCH --error=logs/slurm/afterauxall_%j.err
set -uo pipefail
cd /home/bjc75/other/biophyschem
D=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/aux_all
n=$(ls "$D" | grep -c '^shard_.*\.h5$')
echo "aux_all shards present: $n of 512"
if [ "$n" -ne 512 ]; then
    echo "incomplete: not launching (fill in the missing shards, then rerun this script)"
    exit 1
fi
if squeue -u bjc75 -h -o "%j" | grep -qx v3_L10M_auxAll || [ -d runs_v3/v3_L10M_auxAll ]; then
    echo "v3_L10M_auxAll already submitted or started; not submitting again"
    SKIP_TRAIN=1
fi
SNAP=$(cat /home/bjc75/scratch/pangea/code/seeds_LATEST)
BASE="--global-readout attn_mlp --pair-bias --test-fraction 0.05"
AUX="--aux-shards $D --aux-per-step 512 --accum-steps 2"
BIG="gpu:l40s|gpu:a40|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
prev=""; ids=""
for i in $(seq 1 12); do
    [ -n "${SKIP_TRAIN:-}" ] && break
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    prev=$(env CODE_ROOT=$SNAP SHARDS="/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" \
        RUNS_DIR=runs_v3 RUN_NAME=v3_L10M_auxAll WIDTH=272 BLOCKS=12 ATTN=4 RANK=272 LR=1e-3 \
        HEADS=8 EPOCHS=100 EXTRA="$BASE $AUX" sbatch --parsable --job-name=v3_L10M_auxAll \
        --time=01:00:00 --constraint="$BIG" $dep scripts/slurm_train_v2.sh)
    ids="$ids $prev"
done
echo "v3_L10M_auxAll (1-hour links, code $SNAP):$ids" | tee -a results/v3_continuations.txt
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
BSNAP=$(cat /home/bjc75/scratch/pangea/code/bpp_LATEST)
mkdir -p /nfs/roberts/scratch/pi_skr2/bjc75/pangea/bpp_auxAll
(cd /tmp && python -I -W ignore "$BSNAP/src/data/generate_bpp.py" --shards "$D/shard_*.h5" \
    --out /nfs/roberts/scratch/pi_skr2/bjc75/pangea/bpp_auxAll --workers 4 | tail -2)
echo "aux_all base-pair matrices done $(date)"
