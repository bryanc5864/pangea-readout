#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Sampling-noise ceilings of the 16-snapshot relaxed-ensemble channels.
#SBATCH --job-name=snapceil
#SBATCH --partition=devel
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/snapceil_%j.out
#SBATCH --error=logs/slurm/snapceil_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
cd /home/bjc75/other/biophyschem
python -W ignore src/eval/snapshot_ceiling.py --shards "/home/bjc75/scratch/pangea/targets_v2/shard_*.h5"
