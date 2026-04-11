#!/usr/bin/env bash
set -euo pipefail

# In-memory prune + eval for Kimi-VL on GQA.
# No pruned checkpoint is saved; only eval outputs are written.
#
# Examples
# --------
# Default 50% layerwise pruning on 100 samples:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_prune_eval_kimi_gqa.sh
#
# Full eval:
#   NUM_SAMPLES=0 CUDA_VISIBLE_DEVICES=1 bash scripts/run_prune_eval_kimi_gqa.sh
#
# Coverage inter-layer + coverage intra-layer:
#   INTER_METHOD=coverage INTRA_METHOD=coverage \
#   LAYERWISE_WEIGHT_SOURCE=block_loss \
#   EXPERTWISE_WEIGHT_SOURCE=expert_out_contrib \
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_prune_eval_kimi_gqa.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SCORES_PATH="${SCORES_PATH:-${PREFIX}/storage/prune/scores/kimi_gqa/scores.pt}"
PRUNE_RATIO="${PRUNE_RATIO:-0.50}"
INTER_METHOD="${INTER_METHOD:-uniform}"     # uniform | coverage | global 
INTRA_METHOD="${INTRA_METHOD:-coverage}"    # expertwise | layerwise |  coverage
# Layerwise weights source for coverage inter-layer (leave empty for unweighted)
#   uniform : 随便填一个值，fallback 到 uniform 
#   repr_change : use layerwise_repr_change from scores (gradient-free, always available)
#   block_loss  : use layerwise_loss from scores (requires --collect_contrib in Phase 1)
LAYERWISE_WEIGHT_SOURCE="${LAYERWISE_WEIGHT_SOURCE:-block_loss}"
# Expertwise weights source for coverage intra-layer
#   expert_out_contrib : use expertwise_weights.attr_coverage from scores payload
#   expert_usage       : use expertwise_weights.usage_coverage from scores payload
#   empty              : uniform anchor across experts
EXPERTWISE_WEIGHT_SOURCE="${EXPERTWISE_WEIGHT_SOURCE:-expert_out_contrib}"
EVICT_MIN_CHANNELS="${EVICT_MIN_CHANNELS:-0}"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
START_IDX="${START_IDX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SUBSET_SEED="${SUBSET_SEED:-}"

# AFFINITY_PATH="${AFFINITY_PATH:-storage/prune/scores/kimi_gqa/affinity.pt}"
AFFINITY_PATH="${AFFINITY_PATH:-}"
AFFINITY_THRESHOLD="${AFFINITY_THRESHOLD:-0.9}"

RATIO_TAG="p$(python3 -c "print(str(int(float('${PRUNE_RATIO}')*100)))")"
METHOD_TAG="${INTER_METHOD}_${INTRA_METHOD}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/results/prune_eval_kimi_gqa_${RATIO_TAG}_${METHOD_TAG}}"

EXTRA_ARGS=("$@")

CMD=(
    python scripts/prune_and_eval_kimi_gqa.py
    --model_path "${MODEL_PATH}"
    --scores_path "${SCORES_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --prune_ratio "${PRUNE_RATIO}"
    --inter_method "${INTER_METHOD}"
    --intra_method "${INTRA_METHOD}"
    --num_samples "${NUM_SAMPLES}"
    --start_idx "${START_IDX}"
    --batch_size "${BATCH_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${LAYERWISE_WEIGHT_SOURCE}" ]]; then
    CMD+=(--layerwise_weight_source "${LAYERWISE_WEIGHT_SOURCE}")
fi

if [[ -n "${EXPERTWISE_WEIGHT_SOURCE}" ]]; then
    CMD+=(--expertwise_weight_source "${EXPERTWISE_WEIGHT_SOURCE}")
fi

if [ "${EVICT_MIN_CHANNELS}" -gt 0 ] 2>/dev/null; then
    CMD+=(--evict_min_channels "${EVICT_MIN_CHANNELS}")
fi

if [[ -n "${SUBSET_SEED}" ]]; then
    CMD+=(--subset_seed "${SUBSET_SEED}")
fi

if [[ -n "${AFFINITY_PATH}" ]]; then
    CMD+=(--affinity_path "${AFFINITY_PATH}")
    CMD+=(--affinity_threshold "${AFFINITY_THRESHOLD}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Model        : ${MODEL_PATH}"
echo "Scores       : ${SCORES_PATH}"
echo "Prune ratio  : ${PRUNE_RATIO}"
echo "Inter method : ${INTER_METHOD}"
echo "Intra method : ${INTRA_METHOD}"
echo "Layer weights: ${LAYERWISE_WEIGHT_SOURCE:-none}"
echo "Expert weights: ${EXPERTWISE_WEIGHT_SOURCE:-none}"
echo "Evict min    : ${EVICT_MIN_CHANNELS:-0}"
echo "Eval output  : ${OUTPUT_DIR}"
echo "Samples      : ${NUM_SAMPLES} (0=full)"
echo "GPU          : ${CUDA_VISIBLE_DEVICES}"
if [[ -n "${AFFINITY_PATH}" ]]; then
    echo "Affinity     : ${AFFINITY_PATH} @ ${AFFINITY_THRESHOLD}"
else
    echo "Affinity     : disabled"
fi
echo ""
"${CMD[@]}"
