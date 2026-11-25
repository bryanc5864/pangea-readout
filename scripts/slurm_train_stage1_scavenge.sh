#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
#
# Stage-1 pretraining: fit PANGEA to the analytic landscape.
#
# `PLAN.md`: "Pretraining on analytically computable targets... This stage teaches the
# trunk the local sequence-to-physics map before any expensive label is touched."
#
# Partition: `scavenge_gpu`, which has 95 nodes and no cap, against `gpu` which showed
# 40 of 40 nodes allocated and nothing idle. Scavenge preempts with REQUEUE, and a
# requeued job restarts from the beginning -- acceptable for a first run of a few epochs,
# and the per-epoch checkpoint means a resume could be added if preemption becomes the
# limiting factor rather than queue time.
#
# `gpu_devel` is not used: it allows exactly one job per user (MaxSubmitPU=1), which is
# easy to mistake for a global submit limit.
#
# The installed torch wheels bundle CUDA 13.2 while the cluster's newest module is 12.8,
# so whether CUDA works is a property of the driver rather than of the modules.
# `--require-cuda` turns a silent CPU fallback into an error; a run that quietly drops to
# CPU finishes, reports numbers, and the only symptom is that it was slow.
#
# Walltime is one hour, not eight, and that is a scheduling decision rather than an
# estimate pulled down. 147,700 fragments of 61 bp against a 2M-parameter model is about
# 1,150 batches an epoch; even with the second forward pass that architectural
# reverse-complement equivariance costs, twelve epochs is tens of minutes on any modern
# GPU. An eight-hour request cannot be backfilled into the gaps between other users' jobs,
# and the GPU nodes here sit MIXED rather than idle -- partially allocated, with free
# GPUs that only a short job can slot into. Asking for what the job needs is what gets it
# scheduled. Resume is on by default, so if an hour turns out to be short the requeue
# continues from the last epoch rather than starting over.
#
#SBATCH --job-name=pangea-trains
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/trains_%j.out
#SBATCH --error=logs/slurm/trains_%j.err
set -uo pipefail
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/train"
set -e
cd /home/bjc75/other/biophyschem
mkdir -p logs/slurm runs

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

RUN="${RUN_NAME:-stage1_scav_$(date +%Y%m%d_%H%M)}"
SHARDS="${PANGEA_SCRATCH}/targets/shard_*.h5"

# The dataset is checked on its own terms before anything trains on it. Several failures
# this session were things that reported success while producing nothing, so the gate is
# the artefact rather than the exit code of whatever made it.
python -W ignore src/data/qc_targets.py \
    --shards "${SHARDS}" \
    --out "runs/${RUN}/qc_targets.json" \
    || { echo "TARGET QC FAILED -- not training on this dataset"; exit 1; }

python -W ignore src/model/train.py \
    --shards "${SHARDS}" \
    --out "runs/${RUN}" \
    --width "${WIDTH:-192}" \
    --blocks "${BLOCKS:-12}" \
    --rank "${RANK:-64}" \
    --batch-size "${BATCH:-128}" \
    --epochs "${EPOCHS:-12}" \
    --lr "${LR:-3e-4}" \
    --windows 10 20 50 \
    --workers 3 \
    --require-cuda

echo "training complete: runs/${RUN}"
