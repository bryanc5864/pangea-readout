#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Two-epoch smoke test of the full Family L configuration (axial pair stack, strand tie,
# base-pair supervision; SMOKE_EXTRA adds e.g. the auxiliary windows) on eight shards.
#SBATCH --job-name=smoke-bpp
#SBATCH --partition=gpu,gpu_h100,gpu_b200,gpu_h200,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/smokebpp_%j.out
#SBATCH --error=logs/slurm/smokebpp_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
OUT=/nfs/roberts/scratch/pi_skr2/bjc75/pangea/smoke_bpp_$SLURM_JOB_ID
python -W ignore src/model/train.py --shards "/home/bjc75/scratch/pangea/targets_v2/shard_*.h5" \
    --max-shards 8 --out "$OUT" --width 272 --blocks 12 --rank 272 --attn-layers 4 \
    --attn-heads 8 --epochs 2 --batch-size 512 --lr 1e-3 --split similarity --test-fraction 0.05 \
    --global-readout attn_mlp --pair-bias --pair-module axial --pair-channels 64 --pair-layers 4 \
    --pair-heads 4 --strand-tie --bpp-dir /nfs/roberts/scratch/pi_skr2/bjc75/pangea/bpp_v2 \
    --require-cuda ${SMOKE_EXTRA:-}
python - "$OUT" <<'PY'
import json, sys
h = [json.loads(l) for l in open(sys.argv[1] + "/history.jsonl")]
for d in h:
    print({k: d.get(k) for k in ("epoch", "train_loss", "train_bpp_loss", "seconds", "per_bp_median_r2", "global_median_r2")},
          "L per-bp", d.get("per_bp_median_r2_by_family", {}).get("L"))
PY
