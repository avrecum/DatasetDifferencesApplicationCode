#!/usr/bin/env bash
#SBATCH --job-name=preference-bank
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=reports/logs/bank-%j.out
#SBATCH --error=reports/logs/bank-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-${PROJECT}/avrecum/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used --format=csv -l 2 > reports/logs/bank-gpu-${SLURM_JOB_ID}.csv &
PREFERENCE_MONITOR_PID=$!
trap 'kill "$PREFERENCE_MONITOR_PID" 2>/dev/null || true' EXIT
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff.bank_diagnostics \
  --bank-dir "${HF_HOME}/preference_diff/bank/4edc5ef4ac601a3d08225b53567b6695cbd60b5e" \
  --output reports/source_inspection/text_bank_template_diagnostic.json
if [[ -n "${1:-}" ]]; then
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff score \
    --output "$1" --resume
  .venv/bin/python scripts/check_distributed_scores.py \
    --reference "${2:-runs/pilot100/imagereward/scores}" --candidate "$1/scores" \
    --output reports/source_inspection/four_gpu_reference_validation.json
fi
