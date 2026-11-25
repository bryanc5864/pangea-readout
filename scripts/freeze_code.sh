#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Freeze the code a multi-hour array will run, and prove the frozen copy works.
#
# Copies src/ plus every data file the code reads at runtime from outside src/, writes a
# sha256 manifest, then runs the target generator FROM THE SNAPSHOT on two fragments and
# refuses to publish the snapshot unless both succeed. The first frozen v2 array skipped
# that last step; the snapshot lacked results/qc_tetramer_gfn2.json, every fragment of
# 1024 shards failed, and an hour of 191 cores went into failures nobody saw.
#
# Usage: scripts/freeze_code.sh <tag>      -> prints the snapshot path on success
set -uo pipefail
TAG="${1:?tag required}"
REPO=/home/bjc75/other/biophyschem
# Snapshots live in group scratch: the home fileset has a 500,000-file limit (DECISIONS.md, 2026-11-21).
SNAP="/nfs/roberts/scratch/pi_skr2/bjc75/pangea/code/${TAG}_$(date +%Y%m%d_%H%M)"
# Runtime data read from outside src/ (found by grepping src/ for results/ and data/).
RUNTIME_DATA=(results/qc_tetramer_gfn2.json)

set -e
mkdir -p "$SNAP"
cp -a "$REPO/src" "$SNAP/"
find "$SNAP/src" -name "__pycache__" -type d -prune -exec rm -rf {} +
for f in "${RUNTIME_DATA[@]}"; do
  mkdir -p "$SNAP/$(dirname "$f")"; cp -a "$REPO/$f" "$SNAP/$f"
done
( cd "$SNAP" && { find src -type f -name "*.py" | sort; printf '%s\n' "${RUNTIME_DATA[@]}"; } \
    | xargs sha256sum > MANIFEST.sha256 )
set +e

# Activation runs without -e: the module system returns non-zero on success, which under
# set -e aborts the script silently (an earlier lesson in this project).
source "$REPO/scripts/env.sh" >/dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
set -e
export OMP_NUM_THREADS=1 OPENMM_CPU_THREADS=1
CHECK="$SNAP/selfcheck"; mkdir -p "$CHECK"
( cd /tmp && python -W ignore "$SNAP/src/data/generate_targets.py" \
    --library "${PANGEA_SCRATCH}/fragments/library_v2.tsv" --out "$CHECK/check.h5" \
    --limit 2 --n-ensemble 50 --ensemble-fraction 0 --progress-every 1 \
    --workroot "$CHECK" ) > "$CHECK/log.txt" 2>&1 || { echo "SELF-CHECK RUN FAILED"; tail -5 "$CHECK/log.txt"; exit 1; }
python -I -c "
import h5py, sys
h = h5py.File(sys.argv[1], 'r')
n, f = int(h.attrs['n_fragments']), int(h.attrs['n_failures'])
print(f'self-check: {n} fragments, {f} failures')
sys.exit(0 if (n == 2 and f == 0) else 1)" "$CHECK/check.h5" || { echo "SELF-CHECK HAD FAILURES"; exit 1; }
echo "$SNAP" > "/home/bjc75/scratch/pangea/code/${TAG}_LATEST"
echo "$SNAP"
