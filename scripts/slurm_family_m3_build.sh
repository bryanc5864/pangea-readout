#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family M v3, stage 1 on a GPU: quadruplex models of every class with their MD ensembles
# (OpenMM OpenCL; CUDA does not load on the B200 nodes). One GPU, WORKERS concurrent processes
# sharing it (each a slice of this task's ROWS rows of WINDOWS); resumable per fragment.
#SBATCH --job-name=pangea-famMbuild
#SBATCH --partition=gpu_b200,scavenge_gpu,gpu_h200,gpu_h100,gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famMbuild_%A_%a.out
#SBATCH --error=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famMbuild_%A_%a.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
export PANGEA_G4_TEMPLATES=${CODE_ROOT:?}/data/g4_templates_v2
IDX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"; LO=$(( IDX * ${ROWS:?} ))
W=${WORKERS:-3}; PER=$(( (ROWS + W - 1) / W ))
for k in $(seq 0 $((W - 1))); do
  a=$(( LO + k * PER )); b=$(( a + PER - 1 )); [ $b -ge $(( LO + ROWS )) ] && b=$(( LO + ROWS - 1 ))
  [ $a -gt $b ] && continue
  WR="${SLURM_TMPDIR:-/tmp}/famMb_${SLURM_JOB_ID:-$$}_${IDX}_$k"; mkdir -p "$WR"
  python -W ignore "${CODE_ROOT:?}/src/physics/run_family_m3.py" --windows "${WINDOWS:?}" --rows "$a-$b" \
    --out "${OUT:?}" --stage build --platform OpenCL --workroot "$WR" --batch ${BATCH:-64} --block ${BLOCK:-16} > "${OUT}/logs/build_${IDX}_$k.log" 2>&1 &
done
wait
