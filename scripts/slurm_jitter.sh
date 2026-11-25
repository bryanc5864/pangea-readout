#!/usr/bin/env bash
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
# Grid-jitter replicates of ONE sequence. The spread across these runs is the solver's
# own numerical noise, which is the floor the composition gate measures against: "the
# potential separates composition-matched sequences" only means something relative to it.
#SBATCH --job-name=pangea-jitter
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/jitter_%A_%a.out
#SBATCH --error=logs/slurm/jitter_%A_%a.err
set -uo pipefail
SEQ="${1:?sequence required}"
OUTDIR="${2:?outdir required}"
CONDITIONS="${3:-NaCl_025mM,NaCl_150mM,NaCl_500mM}"
source /home/bjc75/other/biophyschem/scripts/env.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENVS_PATH}/phys"
cd /home/bjc75/other/biophyschem
IDX="${SLURM_ARRAY_TASK_ID:-0}"
WORKROOT="${SLURM_TMPDIR:-/tmp}/pangea_jit_${SLURM_JOB_ID:-$$}_${IDX}"
mkdir -p "$WORKROOT" "$OUTDIR" logs/slurm
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# EXTRA_ARGS: further run_fragment.py flags (minimiser settings, --perturb-A)
read -r -a _extra <<< "${EXTRA_ARGS:-}"
python "${CODE_ROOT:-.}/src/physics/run_fragment.py" --sequence "$SEQ" --frag-id "jitter_${IDX}" \
  --out "$OUTDIR" --workroot "$WORKROOT" --conditions "$CONDITIONS" \
  --grid-jitter 0.175 --jitter-seed "$IDX" "${_extra[@]}"
rc=$?; rm -rf "$WORKROOT"; exit "$rc"
