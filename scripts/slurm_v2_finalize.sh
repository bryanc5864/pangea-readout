#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# After the v2 target arrays: verify, QC, build the dataset cache once, then submit the
# v2 experiment set. Submitted with a dependency on every array (afterany), so it runs
# whether or not some tasks failed -- and refuses to go further unless all 1024 shards
# exist and QC reports no problems. Building the cache here, once, matters: every
# training job reads it, and two jobs building it at the same moment would write the
# same file concurrently.
#SBATCH --job-name=pangea-finalize
#SBATCH --partition=day
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=logs/slurm/finalize_%j.out
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
cd /home/bjc75/other/biophyschem
T=/home/bjc75/scratch/pangea/targets_v2
n=$(ls $T/shard_*.h5 2>/dev/null | wc -l)
echo "shards present: $n"
if [ "$n" -lt 1024 ]; then
  missing=$(python3 -c "import glob,re; have={int(re.search(r'shard_(\d+)', p).group(1)) for p in glob.glob('$T/shard_*.h5')}; print(','.join(str(i) for i in range(1024) if i not in have))")
  echo "NOT FINALIZING: missing shards $missing"; exit 1
fi
conda activate "${CONDA_ENVS_PATH}/phys"
python -W ignore src/data/qc_targets.py --shards "$T/shard_*.h5" --out results/qc_targets_v2.json \
    --expect-shards 1024 > /dev/null 2>&1
python3 -c "
import json, sys
d = json.load(open('results/qc_targets_v2.json'))
print('QC problems:', d['problems']); print('QC warnings:', d['warnings'][:10])
s = d['summary']; print('fragments', s['n_fragments'], 'per-bp', s['n_per_bp_channels'], 'global', s['n_global_channels'])
sys.exit(1 if d['problems'] else 0)" || { echo "NOT FINALIZING: QC problems"; exit 1; }
conda activate "${CONDA_ENVS_PATH}/train"
export OMP_NUM_THREADS=4
python -u -W ignore - <<'PY' || { echo "NOT FINALIZING: cache build failed"; exit 1; }
import os, sys
sys.path.insert(0, "src/model")
from data import InMemoryLandscape
ds = InMemoryLandscape("/home/bjc75/scratch/pangea/targets_v2/shard_*.h5")
print("v2 cache:", len(ds), "fragments;", ds.per_bp_np.shape[1], "per-bp and", ds.global_np.shape[1], "global columns")
print("dropped:", ds.transforms["per_bp_dropped"] + ds.transforms["global_dropped"])
print("groups:", ds.group_info)
PY
W2=144 W10=272 W50=640 W200=1232 W1000=2144 bash scripts/submit_v2_experiments.sh \
    | tee results/v3_submission.txt
