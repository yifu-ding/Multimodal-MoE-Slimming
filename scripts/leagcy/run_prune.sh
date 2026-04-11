#!/usr/bin/env bash
set -euo pipefail

# Phase 2: Structural channel pruning of Kimi-VL MoE experts.
#
# Requires score artifacts from run_collect_scores.sh.
#
# 20% pruning:
#   PRUNE_RATIO=0.2 CUDA_VISIBLE_DEVICES=0 bash scripts/run_prune.sh
#
# 30% pruning (default):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_prune.sh
#
# Coverage-style planning:
#   INTER_METHOD=uniform_coverage INTRA_METHOD=attr_coverage \
#   MODALITY_AWARE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_prune.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SCORES_PATH="${SCORES_PATH:-${PREFIX}/storage/prune/scores/kimi_gqa}"
PRUNE_RATIO="${PRUNE_RATIO:-0.50}"
INTER_METHOD="${INTER_METHOD:-uniform}"
INTRA_METHOD="${INTRA_METHOD:-uniform}"
INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC:-activation}"
ALIGN_INTER="${ALIGN_INTER:-0}"
MIN_PER_EXPERT="${MIN_PER_EXPERT:-0}"
MODALITY_AWARE="${MODALITY_AWARE:-0}"

# Build a descriptive output dir from the ratio, e.g. "p30" for 0.30
RATIO_TAG="p$(python3 -c "print(str(int(float('${PRUNE_RATIO}')*100)))")"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/pruned_models/kimi_gqa_${RATIO_TAG}}"

EXTRA_ARGS=("$@")

CMD=(
    python src/prune.py
    --model_path    "${MODEL_PATH}"
    --scores_path   "${SCORES_PATH}"
    --prune_ratio   "${PRUNE_RATIO}"
    --inter_method  "${INTER_METHOD}"
    --intra_method  "${INTRA_METHOD}"
    --intra_expert_metric "${INTRA_EXPERT_METRIC}"
    --align_inter   "${ALIGN_INTER}"
    --min_per_expert "${MIN_PER_EXPERT}"
    --output_dir    "${OUTPUT_DIR}"
)

if [[ "${MODALITY_AWARE}" == "1" ]]; then
    CMD+=(--modality_aware)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Model       : ${MODEL_PATH}"
echo "Scores      : ${SCORES_PATH}"
echo "Prune ratio : ${PRUNE_RATIO}"
echo "Inter       : ${INTER_METHOD}"
echo "Intra       : ${INTRA_METHOD}"
echo "Metric      : ${INTRA_EXPERT_METRIC}"
echo "Modality    : $([[ "${MODALITY_AWARE}" == "1" ]] && echo "text+visual" || echo "disabled")"
echo "Output      : ${OUTPUT_DIR}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo ""
"${CMD[@]}"
