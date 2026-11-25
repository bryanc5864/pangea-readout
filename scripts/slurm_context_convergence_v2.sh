#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# The plan-mandated context convergence study, rerun on the corrected grid.
#
# The first run (33912642) was cancelled part-way. It solved 121 bp fragments cleanly at
# 10 mM and failed every one of them at 50 mM and above, because `write_apbs_input` sized
# the shared mg-auto `dime` from the fine box alone and the coarse grid had degraded to
# 1.66 A at that length -- off the plan's "near 1 angstrom" specification. The nonlinear
# Boltzmann source term scales with ionic strength and overflowed. Without the 121 bp
# reference there is no convergence curve to measure against, so four of five conditions
# were unusable. Its partial output is kept as
# results/context_convergence_pilot_prefix_grid.json.
#
# Two things changed since:
#   * `dime` now satisfies both grid levels, so coarse spacing is 0.93-0.98 A from 21 to
#     201 bp. Verified on the exact failing cases.
#   * The study runs its (sequence, length) cells in a process pool. With the corrected
#     grid a 121 bp solve takes about 121 s against 22 s at 21 bp, so the reference length
#     dominates and the serial version no longer fits a walltime.
#
# **Eight workers, not thirty-two, and 192 GB.** The first parallel attempt (33930809)
# used 32 workers in 128 GB and was killed by the OOM killer: the recorded failure
# reasons were `rc=-9` -- SIGKILL -- on 23 of 24 121 bp solves and 9 of 24 at 61 bp. The
# corrected grid is 14.1M points at 121 bp, and APBS multigrid holds many work arrays per
# point, so 32 concurrent solves do not fit. That run also showed heavy start-up
# contention: in its first condition every length took about 120 s, including 21 bp,
# while the second condition settled to 16-20 s once the pool was pipelined. Both
# effects corrupt the timings the study is partly there to measure.
#
#SBATCH --job-name=pangea-conv2
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=logs/slurm/conv2_%j.out
#SBATCH --error=logs/slurm/conv2_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
set -e
cd /home/bjc75/other/biophyschem
mkdir -p logs/slurm results

# One thread per process: APBS does not use threads (measured), and OpenMM's CPU platform
# would otherwise take the node's full core count per worker.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-8}"
python -W ignore src/physics/context_convergence.py \
    --n-sequences 24 \
    --lengths 121 61 41 31 21 \
    --conditions NaCl_010mM NaCl_050mM NaCl_100mM NaCl_150mM NaCl_250mM \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --out results/context_convergence_pilot_v3.json \
    --workroot "${SLURM_TMPDIR:-/tmp}"
