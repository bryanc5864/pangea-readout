#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# What GPU and which driver? The torch wheels pulled nvidia-*-cu13 packages, so they
# bundle a CUDA 13 runtime; whether that runs here is a property of the driver, not of
# the CUDA modules on the cluster. Checking before the first training job rather than
# discovering it from a job that silently falls back to CPU.
#SBATCH --job-name=pangea-gpuprobe
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/gpuprobe_%j.out
#SBATCH --error=logs/slurm/gpuprobe_%j.err
set -uo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "--- CUDA version the driver supports:"
nvidia-smi | head -4 | tail -1
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/train" 2>/dev/null || { echo "train env missing"; exit 0; }
python - <<'PY' || echo "torch not importable yet"
import torch
print("torch", torch.__version__, "| bundled cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    a = torch.randn(4096, 4096, device="cuda")
    import time; torch.cuda.synchronize(); t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    print("matmul 4096^3 x20: %.2f s -> %.1f TFLOP/s"
          % (time.time()-t, 20*2*4096**3/(time.time()-t)/1e12))
PY
