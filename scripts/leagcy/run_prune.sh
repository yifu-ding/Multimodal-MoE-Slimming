#!/usr/bin/env bash
set -euo pipefail

# Phase 2: Structural channel pruning of Kimi-VL MoE experts.
#
# Requires scores.pt from run_collect_scores.sh.
#
# Examples
# --------
# Uniform 30% (default):
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_prune.sh
#
# Coverage inter-layer + layerwise intra-layer:
#   INTER_METHOD=coverage INTRA_METHOD=layerwise CUDA_VISIBLE_DEVICES=2 bash scripts/run_prune.sh
#
# Coverage inter-layer + coverage intra-layer:
#   INTER_METHOD=coverage INTRA_METHOD=coverage CUDA_VISIBLE_DEVICES=2 bash scripts/run_prune.sh
#
# Global allocation:
#   INTRA_METHOD=global CUDA_VISIBLE_DEVICES=2 bash scripts/run_prune.sh
#
# Custom output dir:
#   OUTPUT_DIR=storage/prune/pruned_models/my_run bash scripts/run_prune.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SCORES_PATH="${SCORES_PATH:-${PREFIX}/storage/prune/scores/kimi_gqa/scores.pt}"
PRUNE_RATIO="${PRUNE_RATIO:-0.50}"

# Pruning - building mask
INTER_METHOD="${INTER_METHOD:-uniform}"     # uniform | coverage | global 
INTRA_METHOD="${INTRA_METHOD:-layerwise}"    # expertwise | layerwise |  coverage
# Layerwise weights source for coverage inter-layer (leave empty for unweighted)
#   uniform : 随便填一个值，fallback 到 uniform 
#   repr_change : use layerwise_repr_change from scores (gradient-free, always available)
#   block_loss  : use layerwise_loss from scores (requires --collect_contrib in Phase 1)
LAYERWISE_WEIGHT_SOURCE="${LAYERWISE_WEIGHT_SOURCE:-block_loss}"
# Pruning - adjusting mask
# Eviction: experts with fewer than this many channels are evicted post-planning
# Set to 0 to disable. Typical: 64 (≈4.5% of I=1408 for Kimi-VL)
EVICT_MIN_CHANNELS="${EVICT_MIN_CHANNELS:-0}"

# Build a descriptive output dir tag, e.g. "p30_uniform_expertwise"
RATIO_TAG="p$(python3 -c "print(str(int(float('${PRUNE_RATIO}')*100)))")"
METHOD_TAG="${INTER_METHOD}_${INTRA_METHOD}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/pruned_models/kimi_gqa_${RATIO_TAG}_${METHOD_TAG}}"

EXTRA_ARGS=("$@")

CMD=(
    python src/prune.py
    --model_path   "${MODEL_PATH}"
    --scores_path  "${SCORES_PATH}"
    --prune_ratio  "${PRUNE_RATIO}"
    --inter_method "${INTER_METHOD}"
    --intra_method "${INTRA_METHOD}"
    --output_dir   "${OUTPUT_DIR}"
)

if [ -n "${LAYERWISE_WEIGHT_SOURCE}" ]; then
    CMD+=(--layerwise_weight_source "${LAYERWISE_WEIGHT_SOURCE}")
fi

if [ "${EVICT_MIN_CHANNELS}" -gt 0 ] 2>/dev/null; then
    CMD+=(--evict_min_channels "${EVICT_MIN_CHANNELS}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Model        : ${MODEL_PATH}"
echo "Scores       : ${SCORES_PATH}"
echo "Prune ratio  : ${PRUNE_RATIO}"
echo "Inter method : ${INTER_METHOD}"
echo "Intra method : ${INTRA_METHOD}"
echo "Layer weights: ${LAYERWISE_WEIGHT_SOURCE:-none}"
echo "Evict min    : ${EVICT_MIN_CHANNELS:-0}"
echo "Output       : ${OUTPUT_DIR}"
echo "GPU          : ${CUDA_VISIBLE_DEVICES}"
echo ""
"${CMD[@]}"
