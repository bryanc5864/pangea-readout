#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Labels for an auxiliary window set (quadruplex windows, rare-event windows): task i writes
# OUTDIR/base/PREFIX_i.h5 with the auxiliary families (B C D E F L, per-bp L, as aux_all) and
# OUTDIR/famJNK/PREFIX_i.h5 with Families J N K (per-bp J N), as aux_all_famJNK.
#   WINDOWS (TSV), OUTDIR, PREFIX, NSHARDS; CODE_ROOT a frozen snapshot.
#SBATCH --job-name=pangea-wintgt
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/wintgt_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/wintgt_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S="${SLURM_ARRAY_TASK_ID:-0}"
NROWS=$(( $(wc -l < "${WINDOWS:?}") - 1 ))
if [ "$S" -ge "$NROWS" ]; then echo "shard $S empty ($NROWS windows)"; exit 0; fi
mkdir -p "${OUTDIR:?}/base" "$OUTDIR/famJNK"
OUT="$OUTDIR/base/${PREFIX:?}_$(printf '%04d' "$S").h5"
cd /tmp
[ -s "$OUT" ] || python -W ignore "${CODE_ROOT:?}/src/data/generate_aux_targets.py" --windows "$WINDOWS" \
    --out "$OUT" --shard "$S" --n-shards "${NSHARDS:?}" --families B C D E F L --per-bp-families L
python -I -W ignore "$CODE_ROOT/src/data/generate_family_shards.py" --shards "$OUT" --families J N K \
    --per-bp-families J N --out "$OUTDIR/famJNK"
