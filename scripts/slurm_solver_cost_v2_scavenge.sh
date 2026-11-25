#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Does APBS use the threads it is given, and does solute length change its cost?
# Both have been assumed and neither measured. Every CPU-hour figure in DECISIONS.md
# depends on the first; PLAN.md revision 2's main solver lever depends on the second.
#
#SBATCH --job-name=pangea-cost2s
#SBATCH --partition=scavenge
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/cost2_scav_%j.out
#SBATCH --error=logs/slurm/cost2_scav_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
cd /home/bjc75/other/biophyschem
mkdir -p logs/slurm results
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) alloc=$(scontrol show job "${SLURM_JOB_ID}" | grep -oE 'AllocTRES=[^ ]+')"
python src/physics/solver_cost_v2.py \
    --threads 1 2 4 8 \
    --lengths 21 31 41 61 93 \
    --fine-spacings 0.35 0.5 0.7 \
    --plan-lengths \
    --repeats 2 \
    --out results/solver_cost_v2.json \
    --workroot "${SLURM_TMPDIR:-/tmp}"
