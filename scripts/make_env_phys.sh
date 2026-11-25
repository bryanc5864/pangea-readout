#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Build the physics environment: Poisson-Boltzmann solver, charge assignment,
# molecular mechanics, and the scientific Python stack.
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh

echo "conda: $(command -v conda)"
echo "envs : ${CONDA_ENVS_PATH}"
echo "pkgs : ${CONDA_PKGS_DIRS}"

conda create -y -p "${CONDA_ENVS_PATH}/phys" -c conda-forge \
  python=3.11 numpy scipy numba pandas pyarrow h5py matplotlib \
  apbs pdb2pqr openmm pdbfixer mdanalysis biopython pyfaidx pysam \
  scikit-learn tqdm pyyaml pytest
rc=$?
echo "CONDA_CREATE_EXIT=${rc}"
exit "${rc}"
