#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Build the tetramer-context fragment library and run the GFN2-xTB tier over it.
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENVS_PATH/phys"
export XTB_BIN=/home/bjc75/other/biophyschem/scripts/xtb_wrapper.sh
cd /home/bjc75/other/biophyschem

GEOM=/home/bjc75/scratch/pangea/qc/tetramer
python src/qc/build_qc_fragments.py --out "$GEOM" --context-length 4 2>&1 | tail -3
python src/qc/run_xtb.py --index "$GEOM/qc_fragment_index.json" \
  --out results/qc_tetramer_gfn2.json --jobs 16 --threads 2 --vipea 2>&1 | tail -6
