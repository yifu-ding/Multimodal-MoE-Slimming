#!/usr/bin/env bash
set -euo pipefail

# Phase 2: Structural channel pruning of Kimi-VL MoE experts.
#
# Requires channel_scores.pt from run_collect_scores.sh.
#
# 20% pruning:
#   PRUNE_RATIO=0.2 CUDA_VISIBLE_DEVICES=0 bash scripts/run_prune.sh
#
# 30% pruning (default):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_prune.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SCORES_PATH="${SCORES_PATH:-${PREFIX}/storage/prune/scores/kimi_gqa/channel_scores.pt}"
PRUNE_RATIO="${PRUNE_RATIO:-0.50}"

# Build a descriptive output dir from the ratio, e.g. "p30" for 0.30
RATIO_TAG="p$(python3 -c "print(str(int(float('${PRUNE_RATIO}')*100)))")"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/pruned_models/kimi_gqa_${RATIO_TAG}}"

EXTRA_ARGS=("$@")

CMD=(
    python src/prune.py
    --model_path    "${MODEL_PATH}"
    --scores_path   "${SCORES_PATH}"
    --prune_ratio   "${PRUNE_RATIO}"
    --output_dir    "${OUTPUT_DIR}"
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Model       : ${MODEL_PATH}"
echo "Scores      : ${SCORES_PATH}"
echo "Prune ratio : ${PRUNE_RATIO}"
echo "Output      : ${OUTPUT_DIR}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo ""
"${CMD[@]}"
