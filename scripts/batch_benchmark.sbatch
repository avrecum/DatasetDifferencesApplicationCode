#!/usr/bin/env bash
#SBATCH --job-name=preference-batch
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=reports/logs/batch-%j.out
#SBATCH --error=reports/logs/batch-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-${PROJECT}/avrecum/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
PREFERENCE_REFERENCE=${1:-runs/pilot500/imagereward}
PREFERENCE_CANDIDATE=${2:-runs/batch16_verification/imagereward}
mkdir -p "$PREFERENCE_CANDIDATE"
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used --format=csv -l 2 > "$PREFERENCE_CANDIDATE/gpu-utilization-${SLURM_JOB_ID}.csv" &
PREFERENCE_MONITOR_PID=$!
trap 'kill "$PREFERENCE_MONITOR_PID" 2>/dev/null || true' EXIT
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff score \
  --output "$PREFERENCE_CANDIDATE" --manifest "$PREFERENCE_REFERENCE/manifest" \
  --image-batch-size 16 --text-chunk-size 2048 --resume
.venv/bin/python scripts/check_distributed_scores.py \
  --reference "$PREFERENCE_REFERENCE/scores" --candidate "$PREFERENCE_CANDIDATE/scores" \
  --allow-batch-chunk-change --allow-matching-invalid \
  --output reports/source_inspection/batched_four_gpu_reference_validation.json
