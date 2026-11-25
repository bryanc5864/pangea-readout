#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Builds every input of the final ladder that is derived from finished production runs, in
# order, refusing to proceed past a missing input:
#   1. QC table -> Bq side shards (main corpus only: a 136-context lookup needs no more data)
#   2. Family M side shards (main corpus, quadruplex windows) from run_family_m outputs
#   3. Family A long shards (with the ensemble) from the production solves
# Run on a compute node (CPU); each step is idempotent where its tool is.
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
C=${CODE_ROOT:?}; G=/nfs/roberts/scratch/pi_skr2/bjc75/pangea; R=/home/bjc75/other/biophyschem
step=${1:-all}
if [ "$step" = all ] || [ "$step" = bq ]; then
  python -W ignore $C/src/qc/build_qc_table.py --out $R/results/qc/qc_tetramer_table.json
  export PANGEA_QC_TABLE=$R/results/qc/qc_tetramer_table.json
  python -I -W ignore $C/src/data/generate_family_shards.py --shards "/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" --families Bq --out $G/targets_v2_famBq
fi
if [ "$step" = all ] || [ "$step" = m ]; then
  python -W ignore $C/src/data/pack_family_m.py --shards "/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" --solves $G/famM/corpus_solves --out $G/targets_v2_famM
  python -W ignore $C/src/data/pack_family_m.py --shards "$G/fold_windows/base/fold_*.h5" --solves $G/fold_windows/famM_solves --out $G/fold_windows/famM
fi
if [ "$step" = all ] || [ "$step" = a ]; then
  python -W ignore $C/src/data/generate_family_a_long_shards.py --shards "/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" --windows $G/famA_prod/windows.tsv --solves $G/famA_prod/solves $G/famA_prod/solves_suppl $G/famA_prod/solves_fix --ensemble $G/famA_prod/ensemble --out $G/famA_long
fi
echo "finalize step '$step' done $(date)"
