#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Build capped central-step geometries for every reverse-complement-canonical context.
#SBATCH --job-name=pangea-qcbuild
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/qcbuild_%j.out
#SBATCH --error=logs/slurm/qcbuild_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
cd /home/bjc75/other/biophyschem
python "${CODE_ROOT:?}/src/qc/build_qc_fragments.py" --out "${GEOM:?}" --context-length "${CTX_LEN:?}" --flank 3
