#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Explicit-solvent MD hydration library (Family G hydration, Family K 2'-OH): task i runs
# contexts[i::N_SHARDS] for each class on one GPU (OpenCL; CUDA when the node supports it).
#SBATCH --job-name=pangea-mdhyd
#SBATCH --partition=gpu,scavenge_gpu,gpu_h100,gpu_h200,gpu_b200
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/mdhyd_%A_%a.out
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
P=OpenCL; python -c "import openmm; s=openmm.System(); s.addParticle(1.0); openmm.Context(s, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName('CUDA'))" 2>/dev/null && P=CUDA
for cls in dna rna hyb; do
  python -W ignore "${CODE_ROOT:?}/src/physics/run_md_hydration.py" --class $cls --shard ${SLURM_ARRAY_TASK_ID:-0} \
    --n-shards ${N_SHARDS:?} --out /nfs/roberts/scratch/pi_skr2/bjc75/pangea/md_hydration --platform $P
done
