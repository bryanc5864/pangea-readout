#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Every input of the final ladder (v7) derived from the production runs, step by step; each step
# refuses to run past a missing input (the packers stop on missing solves). Run on a compute node
# (CPU). Steps: bq m a longset kes ax glib nes isc jox w   (or "all").
#   bq       QC tetramer table -> Bq side shards (main corpus)
#   m        Family M side shards: quadruplexes (v3, GPU MD ensembles), triplexes, i-motifs, cruciforms
#   a        Family A long shards (121 bp windows, ensemble) from the production solves
#   longset  Family A long set (205 bp segments, tiled solves; window channels)
#   kes      Family K A-form electrostatics (Kes)          ax    Family A extras (Ax)
#   glib     Family G hydration MD, docking, trimer DFT (Gmd, Kmd, Gdk, Bt)
#   nes      Family N modification electrostatics (Nes)    isc   Family I side chains (held apart)
#   jox      Family J oxDNA2 side shards                   w     Family W long fragments
# (Fr and Gr, sequence-only, are packed already: targets_v2_famFrGr.)
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
C=${CODE_ROOT:?}; G=/nfs/roberts/scratch/pi_skr2/bjc75/pangea; R=/home/bjc75/other/biophyschem
MAIN="/home/bjc75/scratch/pangea/targets_v2/shard_*.h5"
step=${1:-all}
par() {   # run the lookup packer as 8 parallel tasks over the main shards: par <out> <args...>
  local out=$1; shift
  for t in 0 1 2 3 4 5 6 7; do
    python -W ignore $C/src/data/pack_lookup_families.py --shards "$MAIN" --out "$out" "$@" --task $t --n-tasks 8 &
  done
  wait
  ls "$out"/missing_task*.json >/dev/null 2>&1 && { echo "missing solves for $out"; exit 3; } || true
}
if [ "$step" = all ] || [ "$step" = bq ]; then
  python -W ignore $C/src/qc/build_qc_table.py --out $R/results/qc/qc_tetramer_table.json
  export PANGEA_QC_TABLE=$R/results/qc/qc_tetramer_table.json
  python -I -W ignore $C/src/data/generate_family_shards.py --shards "$MAIN" --families Bq --out $G/targets_v2_famBq
fi
if [ "$step" = all ] || [ "$step" = m ]; then
  python -W ignore $C/src/data/pack_family_m.py --shards "$MAIN" --g4 $G/famM3/out/solves --triplex $G/famM_tx/solves \
    --imotif $G/famM_im/solves --cruciform $G/famM_cr/solves --windows $G/famA_prod/windows.tsv --out $G/targets_v2_famM
  python -W ignore $C/src/data/pack_family_m.py --shards "$G/fold_windows/base/fold_*.h5" --g4 $G/famM3/out/solves \
    --triplex $G/famM_tx/solves --imotif $G/famM_im/solves --cruciform $G/famM_cr/solves \
    --windows $G/famA_prod/windows.tsv --out $G/fold_windows/famM
fi
if [ "$step" = all ] || [ "$step" = a ]; then
  python -W ignore $C/src/data/generate_family_a_long_shards.py --shards "$MAIN" --windows $G/famA_prod/windows.tsv \
    --solves $G/famA_prod/solves $G/famA_prod/solves_suppl $G/famA_prod/solves_fix --ensemble $G/famA_prod/ensemble \
    --out $G/famA_prod/long_shards
fi
if [ "$step" = all ] || [ "$step" = longset ]; then
  python -W ignore $C/src/data/generate_family_a_longset_shards.py --segments $G/famA_longset/segments.tsv \
    --solves $G/famA_longset/solves --out $G/famA_longset/shards
fi
[ "$step" = all ] || [ "$step" = kes ] && par $G/targets_v2_famKes --kes $G/famK_prod/solves
[ "$step" = all ] || [ "$step" = ax ] && par $G/targets_v2_famAx --aextras $G/famA_extras/solves
[ "$step" = all ] || [ "$step" = glib ] && par $G/targets_v2_famGlib --md $G/md_hydration --dock $G/dock/scores --trimers $G/qc/trimer_tzvp
[ "$step" = all ] || [ "$step" = nes ] && par $G/targets_v2_famNes --nes $G/famN_es/table
[ "$step" = all ] || [ "$step" = isc ] && par $G/targets_v2_famIsc --isc $G/isc/scores
if [ "$step" = all ] || [ "$step" = jox ]; then
  python -W ignore $C/src/data/pack_family_jox.py --shards "$MAIN" --solves $G/famJox/solves_v2 --out $G/targets_v2_famJox
fi
if [ "$step" = all ] || [ "$step" = w ]; then
  python -W ignore $C/src/data/pack_family_w.py --table $G/famW/fragments.tsv --solves $G/famW/solves --out $G/famW/shards
fi
echo "finalize v7 step '$step' done $(date)"
