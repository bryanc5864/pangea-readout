#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Family W (window ladder and wide-branch targets) on the 10,240 bp fragment table; ROWS per task.
#SBATCH --job-name=pangea-famW
#SBATCH --partition=day,scavenge
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/logs/slurm/famW_%A_%a.out
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
IDX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"; LO=$(( IDX * ${ROWS:?} )); HI=$(( LO + ROWS - 1 ))
cd /tmp && python -W ignore "${CODE_ROOT:?}/src/physics/run_family_w.py" --table "${TABLE:?}" --rows "$LO-$HI" --out "${OUT:?}"
