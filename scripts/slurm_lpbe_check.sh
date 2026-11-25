#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Does the linearised Poisson-Boltzmann equation give the same channels as the nonlinear
# one, and does the agreement improve with ionic strength?
#
# This decides whether high-salt conditions can be kept at all. The nonlinear solver
# diverges at 500 mM regardless of box size or grid spacing (results/highsalt_probe.json),
# while the linearised solver converges there. Linearisation error grows with the
# magnitude of the potential, and screening makes the potential smallest exactly where
# the nonlinear solver fails, so lpbe should be *most* trustworthy at high salt. That is
# an argument, not evidence, so it is measured here on the conditions where both
# equations converge, and the trend across salt is what justifies or refutes using lpbe
# above them.
#
# Runs on a properly sized allocation: this is not safe in a 1 CPU / 8 GB interactive
# session (see DECISIONS.md).
#
#SBATCH --job-name=pangea-lpbe
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/lpbe_%j.out
#SBATCH --error=logs/slurm/lpbe_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
cd /home/bjc75/other/biophyschem
mkdir -p logs/slurm results
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) alloc=$(scontrol show job "${SLURM_JOB_ID}" | grep -oE 'AllocTRES=[^ ]+')"
python src/physics/lpbe_check.py \
    --conditions NaCl_010mM NaCl_050mM NaCl_100mM NaCl_150mM NaCl_250mM NaCl_500mM \
    --n-sequences 4 \
    --out results/lpbe_vs_npbe.json \
    --workroot "${SLURM_TMPDIR:-/tmp}"
