#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Reverse-complement consistency of the Watson/Crick partner channels (src/eval/rc_consistency.py).
#SBATCH --job-name=rccheck
#SBATCH --partition=devel
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/rccheck_%j.out
#SBATCH --error=logs/slurm/rccheck_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
cd /home/bjc75/other/biophyschem
python -W ignore src/eval/rc_consistency.py --shard /home/bjc75/scratch/pangea/targets_v2/shard_0007.h5 --n 6
