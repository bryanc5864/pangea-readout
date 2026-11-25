#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Tier-1 quantum chemistry over hexamer contexts: 2,080 reverse-complement-canonical
# 6-mers, which is the broad context coverage the plan asks the cheap tier to provide.
#SBATCH --job-name=pangea-qc6
#SBATCH --partition=day
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/qc6_%j.out
#SBATCH --error=logs/slurm/qc6_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
export XTB_BIN=/home/bjc75/other/biophyschem/scripts/xtb_wrapper.sh
cd /home/bjc75/other/biophyschem
GEOM=/home/bjc75/scratch/pangea/qc/hexamer
mkdir -p logs/slurm results
python src/qc/build_qc_fragments.py --out "$GEOM" --context-length 6 --flank 3 2>&1 | tail -3
python src/qc/run_xtb.py --index "$GEOM/qc_fragment_index.json" \
  --out results/qc_hexamer_gfn2.json --jobs 32 --threads 1 --vipea 2>&1 | tail -5
