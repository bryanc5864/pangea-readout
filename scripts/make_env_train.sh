#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# The training environment, kept separate from `phys` on purpose. Label generation needs
# OpenMM, ViennaRNA and cgNA+; training needs none of them and needs CUDA wheels that
# would otherwise have to coexist with that stack. Labels are written to disk between the
# two, so neither environment has to import the other.
#
# CUDA 12.8 is the newest toolkit on this cluster, so the cu128 wheels are the match.
# No `set -e` around the module system: lmod's init returns non-zero on paths that are
# not failures, which silently aborted this script after sourcing env.sh and left it
# exiting 0 with no environment created. Errors are checked explicitly instead.
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
set -e

ENV_PATH="${CONDA_ENVS_PATH}/train"
if [ ! -d "$ENV_PATH" ]; then
  conda create -y -p "$ENV_PATH" python=3.11 pip
fi
conda activate "$ENV_PATH"

python -m pip install --upgrade pip
# Torch first, from the CUDA 12.8 index, so the generic PyPI wheel is never selected.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# Everything else from PyPI.
python -m pip install numpy scipy h5py tqdm

python - <<'PY'
import torch, h5py, numpy, scipy
print("torch   ", torch.__version__)
print("cuda build", torch.version.cuda)
print("h5py    ", h5py.__version__)
print("numpy   ", numpy.__version__)
print("scipy   ", scipy.__version__)
print("cuda available on this node:", torch.cuda.is_available())
PY
