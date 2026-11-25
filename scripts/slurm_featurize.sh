#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Physics landscapes of an assay table from a frozen encoder (src/readout/featurize.py),
# split over the array's tasks. RUN, TABLE, OUT, and EXTRA (further featurize flags).
#SBATCH --job-name=pangea-feat
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/feat_%A_%a.out
#SBATCH --error=logs/slurm/feat_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
python -W ignore "${CODE_ROOT:-.}/src/readout/featurize.py" --run "${RUN:?}" --table "${TABLE:?}" \
    --out "${OUT:?}" --chunk "${SLURM_ARRAY_TASK_ID:-0}" --n-chunks "${SLURM_ARRAY_TASK_COUNT:-1}" ${EXTRA:-}
