#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# One READOUT arm on one featurized dataset (src/readout/train_readout.py), on CPU.
#   FEATURES DATASET ARM [EXTRA]
#SBATCH --job-name=readout
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/readout_%j.out
#SBATCH --error=logs/slurm/readout_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python -W ignore "${CODE_ROOT:-.}/src/readout/train_readout.py" --features "${FEATURES:?}" \
    --dataset "${DATASET:?}" --arm "${ARM:?}" ${EXTRA:-}
