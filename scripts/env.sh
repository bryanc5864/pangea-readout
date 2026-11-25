#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Source this to get modules + conda on PATH in a non-interactive shell.
source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /etc/profile.d/modules.sh
module load miniconda/26.9.4

export PANGEA_ROOT=/home/bjc75/other/biophyschem
export PANGEA_SCRATCH=/home/bjc75/scratch/pangea
export CONDA_ENVS_PATH=/home/bjc75/project/conda_envs
export CONDA_PKGS_DIRS=/tmp/bjc75_conda_pkgs  # local SSD: NFS extraction is pathologically slow
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

# usage: pangea_activate <envname>
pangea_activate () {
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENVS_PATH/$1"
}

# numba caches compiled functions next to their source by default; the cgNA+ package sits on the
# home fileset, whose file-count quota (500k) runs near its limit, and tasks failed with "cannot cache
# function ... no locator available". A node-local cache is always writable.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_$(id -u)}"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null || true
