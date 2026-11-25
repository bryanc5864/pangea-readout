#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# The context convergence study PLAN.md revision 2 mandates: how far does the central
# 21 bp potential depend on flank length, at every ionic condition? This sets the
# production fragment lengths and decides which channels can be read as window tracks.
# Pilot scale first, to size the ~2,000-fragment run the plan asks for.
#
#SBATCH --job-name=pangea-conv
#SBATCH --partition=day
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/conv_%j.out
#SBATCH --error=logs/slurm/conv_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
cd /home/bjc75/other/biophyschem
mkdir -p logs/slurm results
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) alloc=$(scontrol show job "${SLURM_JOB_ID}" | grep -oE 'AllocTRES=[^ ]+')"
python src/physics/context_convergence.py \
    --n-sequences 24 \
    --lengths 121 61 41 31 21 \
    --conditions NaCl_010mM NaCl_050mM NaCl_100mM NaCl_150mM NaCl_250mM \
    --out results/context_convergence_pilot.json \
    --workroot "${SLURM_TMPDIR:-/tmp}"
