#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Step time and memory of the conv pair bias against the axial pair stack at two rungs.
#SBATCH --job-name=bench-pair
#SBATCH --partition=gpu,gpu_h100,gpu_b200,gpu_h200,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/bjc75/other/biophyschem/logs/slurm/benchpair_%j.out
#SBATCH --error=/home/bjc75/other/biophyschem/logs/slurm/benchpair_%j.err
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENVS_PATH}/train"
cd /home/bjc75/other/biophyschem
C=runs_v3/v3_L10M/config.json
python -W ignore src/model/bench.py --config $C --configs 272:12:272:4 640:12:512:4 --batch-size 512 --pair-module conv
python -W ignore src/model/bench.py --config $C --configs 272:12:272:4 640:12:512:4 --batch-size 512 --pair-module axial --pair-channels 64 --pair-layers 4
python -W ignore src/model/bench.py --config $C --configs 272:12:272:4 640:12:512:4 --batch-size 512 --pair-module axial --pair-channels 128 --pair-layers 8
