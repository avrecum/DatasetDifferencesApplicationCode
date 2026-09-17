#!/usr/bin/env bash
#SBATCH --job-name=preference-data
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=reports/logs/dataset-%j.out
#SBATCH --error=reports/logs/dataset-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-${PROJECT}/avrecum/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
PREFERENCE_RUN_ROOT=${1:?Supply a prepared run root}
PREFERENCE_DATASET=${2:?Supply imagereward or pickapic}
PREFERENCE_IMAGE_BATCH=${PREFERENCE_IMAGE_BATCH:-16}
PREFERENCE_TEXT_CHUNK=${PREFERENCE_TEXT_CHUNK:-2048}
PREFERENCE_REPORT_ARGS=()
if [[ -f reports/source_inspection/text_bank_template_diagnostic.json ]]; then
  PREFERENCE_REPORT_ARGS+=(--bank-diagnostic reports/source_inspection/text_bank_template_diagnostic.json)
fi
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used --format=csv -l 2 > "$PREFERENCE_RUN_ROOT/gpu-utilization-${SLURM_JOB_ID}.csv" &
PREFERENCE_MONITOR_PID=$!
trap 'kill "$PREFERENCE_MONITOR_PID" 2>/dev/null || true' EXIT
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff score \
  --dataset "$PREFERENCE_DATASET" --output "$PREFERENCE_RUN_ROOT" \
  --image-batch-size "$PREFERENCE_IMAGE_BATCH" --text-chunk-size "$PREFERENCE_TEXT_CHUNK" --resume
.venv/bin/python -m preference_diff analyze --dataset "$PREFERENCE_DATASET" --output "$PREFERENCE_RUN_ROOT" --bootstrap 2000
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff report \
  --dataset "$PREFERENCE_DATASET" --output "$PREFERENCE_RUN_ROOT" --heatmaps 4 "${PREFERENCE_REPORT_ARGS[@]}"
