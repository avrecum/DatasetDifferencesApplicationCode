#!/usr/bin/env bash
#SBATCH --job-name=preference-ir100
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=reports/logs/imagereward-%j.out
#SBATCH --error=reports/logs/imagereward-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-${PROJECT}/avrecum/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
PREFERENCE_RUN_ROOT=${1:-runs/pilot100/imagereward}
export PREFERENCE_GPU_TEST_MANIFEST="${PREFERENCE_RUN_ROOT}/manifest"
export PREFERENCE_GPU_TEST_BANK="${HF_HOME}/preference_diff/bank/4edc5ef4ac601a3d08225b53567b6695cbd60b5e"
export PREFERENCE_GPU_TEST_REPORT="${PREFERENCE_RUN_ROOT}/gpu_batch_reference_validation.json"
.venv/bin/python -m pytest -q tests/test_scoring.py::test_real_gpu_reference
.venv/bin/python -m preference_diff score --output "${PREFERENCE_RUN_ROOT}" --resume
.venv/bin/python -m preference_diff analyze --output "${PREFERENCE_RUN_ROOT}" --bootstrap 2000
.venv/bin/python -m preference_diff report --output "${PREFERENCE_RUN_ROOT}" --heatmaps 3
