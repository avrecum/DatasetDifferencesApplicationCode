#!/usr/bin/env bash
#SBATCH --job-name=preference-ir
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=reports/logs/imagereward-%j.out
#SBATCH --error=reports/logs/imagereward-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-${PROJECT}/avrecum/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
PREFERENCE_RUN_ROOT=${1:-runs/pilot500/imagereward}
PREFERENCE_IMAGE_BATCH=${PREFERENCE_IMAGE_BATCH:-16}
PREFERENCE_TEXT_CHUNK=${PREFERENCE_TEXT_CHUNK:-2048}
PREFERENCE_SCORE_ARGS=()
if [[ -n "${2:-}" ]]; then
  PREFERENCE_SCORE_ARGS+=(--reuse-scores "$2")
fi
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used --format=csv -l 2 > "${PREFERENCE_RUN_ROOT}/gpu-utilization-${SLURM_JOB_ID}.csv" &
PREFERENCE_MONITOR_PID=$!
trap 'kill "$PREFERENCE_MONITOR_PID" 2>/dev/null || true' EXIT
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff score --output "${PREFERENCE_RUN_ROOT}" --image-batch-size "$PREFERENCE_IMAGE_BATCH" --text-chunk-size "$PREFERENCE_TEXT_CHUNK" --resume "${PREFERENCE_SCORE_ARGS[@]}"
.venv/bin/python -m preference_diff analyze --output "${PREFERENCE_RUN_ROOT}" --bootstrap 2000
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff report --output "${PREFERENCE_RUN_ROOT}" --heatmaps 4
