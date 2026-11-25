#!/bin/bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Stage (STEP=stage) or upload+verify (STEP=upload) the Hugging Face deposit, at idle I/O and CPU priority so the
# shared file servers and the node are not loaded. Both steps are restartable: resubmit to continue.
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/tmp/hf/%x_%j.out
set -euo pipefail
G=/nfs/roberts/scratch/pi_skr2/bjc75/pangea
R=/home/bjc75/other/biophyschem
PY=/home/bjc75/.conda/envs/homeward/bin/python   # has huggingface_hub 2.2.0 + hf_xet; used read-only
export HF_HOME=$G/tmp/hf/hf_home HF_XET_CACHE=$G/tmp/hf/hf_home/xet HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME"
case "${STEP:?}" in
  stage)  ionice -c3 nice -n19 $PY -I $R/scripts/hf_stage.py --stage $G/hf_stage --state $G/tmp/hf/stage_state ;;
  upload) ionice -c3 nice -n19 $PY -I $R/scripts/hf_upload.py --stage $G/hf_stage --state $G/tmp/hf/upload_state ;;
esac
