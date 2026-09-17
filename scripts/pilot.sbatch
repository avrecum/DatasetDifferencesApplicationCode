#!/usr/bin/env bash
#SBATCH --job-name=preference-diff
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=reports/logs/preference-%j.out
#SBATCH --error=reports/logs/preference-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-${PROJECT}/avrecum/.hf}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PREFERENCE_RUN_ROOT=${1:-runs/smoke}
PREFERENCE_PYTHON=${PREFERENCE_PYTHON:-.venv/bin/python}
PREFERENCE_IMAGE_BATCH=${PREFERENCE_IMAGE_BATCH:-16}
PREFERENCE_TEXT_CHUNK=${PREFERENCE_TEXT_CHUNK:-2048}
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used --format=csv -l 2 > "${PREFERENCE_RUN_ROOT}/gpu-utilization-${SLURM_JOB_ID}.csv" &
PREFERENCE_MONITOR_PID=$!
trap 'kill "$PREFERENCE_MONITOR_PID" 2>/dev/null || true' EXIT
for dataset in imagereward pickapic; do
  "${PREFERENCE_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff score --dataset "${dataset}" --output "${PREFERENCE_RUN_ROOT}/${dataset}" --image-batch-size "$PREFERENCE_IMAGE_BATCH" --text-chunk-size "$PREFERENCE_TEXT_CHUNK" --resume
  "${PREFERENCE_PYTHON}" -m preference_diff analyze --dataset "${dataset}" --output "${PREFERENCE_RUN_ROOT}/${dataset}"
  "${PREFERENCE_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff report --dataset "${dataset}" --output "${PREFERENCE_RUN_ROOT}/${dataset}" --heatmaps 4
done
"${PREFERENCE_PYTHON}" -m preference_diff compare --left "${PREFERENCE_RUN_ROOT}/imagereward" --right "${PREFERENCE_RUN_ROOT}/pickapic" --output "${PREFERENCE_RUN_ROOT}/comparison"
