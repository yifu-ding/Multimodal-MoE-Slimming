#!/usr/bin/env bash
set -euo pipefail

# Baseline eval: Kimi-VL-A3B-Instruct on GQA testdev_balanced (no pruning).
#
# Quick smoke test (~500 samples):
#   NUM_SAMPLES=500 bash scripts/eval_baseline_kimi_gqa.sh
#
# Full eval (~12k samples):
#   bash scripts/eval_baseline_kimi_gqa.sh

PREFIX="/home/data/dyf/moe-prune"
export PYTHONPATH=".:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"          # 0 = full testdev_balanced
START_IDX="${START_IDX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SUBSET_SEED="${SUBSET_SEED:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/results/baseline_kimi_gqa}"

EXTRA_ARGS=("$@")

CMD=(
    python scripts/eval_baseline_kimi_gqa.py
    --model_name_or_path "${MODEL_PATH}"
    --output_dir        "${OUTPUT_DIR}"
    --num_samples       "${NUM_SAMPLES}"
    --start_idx         "${START_IDX}"
    --batch_size        "${BATCH_SIZE}"
    --max_new_tokens    "${MAX_NEW_TOKENS}"
)

if [[ -n "${SUBSET_SEED}" ]]; then
    CMD+=(--subset_seed "${SUBSET_SEED}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Model  : ${MODEL_PATH}"
echo "Output : ${OUTPUT_DIR}"
echo "Samples: ${NUM_SAMPLES} (0=full)"
echo "GPU    : ${CUDA_VISIBLE_DEVICES}"
echo ""
"${CMD[@]}"
