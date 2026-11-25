#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Build the transformed, standardised in-memory dataset once, so the scaling study's
# training runs load it in seconds instead of each re-reading 512 shards and re-running the
# MinHash similarity grouping over 191,344 sequences. Training jobs depend on this one.
#SBATCH --job-name=pangea-cache
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/cache_%j.out
#SBATCH --error=logs/slurm/cache_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
export OMP_NUM_THREADS=4
python -W ignore - <<'PY'
import sys, os
sys.path.insert(0, "src/model")
from data import InMemoryLandscape
ds = InMemoryLandscape(os.environ.get("SHARDS") or os.environ["PANGEA_SCRATCH"] + "/targets/shard_*.h5",
                       interior_trim_g=int(os.environ.get("G_TRIM", "0")),
                       extra_dirs=os.environ.get("EXTRA_DIRS", "").split() or None)
print("fragments", len(ds), "| per-bp cols", ds.per_bp_np.shape[1],
      "| global cols", ds.global_np.shape[1])
print("log10:", len(ds.transforms["per_bp_log10"]), "per-bp,",
      len(ds.transforms["global_log10"]), "global")
print("dropped:", ds.transforms["per_bp_dropped"] + ds.transforms["global_dropped"])
print("groups:", ds.group_info)
PY
