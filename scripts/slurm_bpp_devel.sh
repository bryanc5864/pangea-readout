#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Base-pair matrices for the main shards, then the auxiliary L shards, on the devel
# partition (4 cores; starts immediately where the day and scavenge queues were hours out).
#SBATCH --job-name=pangea-bppdev
#SBATCH --partition=devel
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/bppdev_%j.out
#SBATCH --error=logs/slurm/bppdev_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
pangea_activate phys
set -e
cd /tmp
G="${CODE_ROOT:?}/src/data/generate_bpp.py"
python -I -W ignore "$G" --shards "/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" \
    --out /nfs/roberts/scratch/pi_skr2/bjc75/pangea/bpp_v2 --workers 4 | tail -2
echo "main done $(date)"
python -I -W ignore "$G" --shards "/nfs/roberts/scratch/pi_skr2/bjc75/pangea/aux_L/shard_*.h5" \
    --out /nfs/roberts/scratch/pi_skr2/bjc75/pangea/bpp_auxL --workers 4 | tail -2
echo "aux done $(date)"
