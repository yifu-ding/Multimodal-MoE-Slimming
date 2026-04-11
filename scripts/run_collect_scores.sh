#!/usr/bin/env bash
set -euo pipefail

# Phase 1: Collect per-channel importance scores for Kimi-VL MoE experts.
#
# Quick smoke test (weight scoring, no GPU data needed):
#   SCORE_TYPE=weight bash scripts/run_collect_scores.sh
#
# Activation scoring on 128 GQA samples:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Activation + contribution scores (layerwise_loss, expert_out_contrib):
#   COLLECT_CONTRIB=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Full run:
#   CUDA_VISIBLE_DEVICES=0 NUM_SAMPLES=512 bash scripts/run_collect_scores.sh
#
# Modality-aware scoring (auto-computes affinity via preliminary routing-survey pass):
#   MODALITY_AWARE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Saved payload overview
# ----------------------
# The generated scores file now contains:
#   - scores                         : per-layer / per-expert channel scores
#   - layerwise_repr_change         : layerwise weights source = repr_change
#   - layerwise_loss                : layerwise weights source = block_loss
#   - expert_usage                  : raw expert usage signal
#   - expert_out_token_contrib      : raw attr_coverage signal
#   - expertwise_weights:
#       * usage_coverage            : reference-aligned per-expert coverage weights
#       * attr_coverage             : reference-aligned per-expert coverage weights
#
# Downstream prune/eval parameter names
# -------------------------------------
# These parameters are consumed by scripts/run_prune_eval_kimi_gqa.sh
# (not by this collect script itself):
#   - LAYERWISE_WEIGHT_SOURCE=repr_change|block_loss
#   - EXPERTWISE_WEIGHT_SOURCE=expert_out_contrib|expert_usage
#
# Recommended default for coverage intra-layer:
#   EXPERTWISE_WEIGHT_SOURCE=expert_out_contrib
#
# Example end-to-end:
#   COLLECT_CONTRIB=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#   INTER_METHOD=uniform INTRA_METHOD=coverage \
#   EXPERTWISE_WEIGHT_SOURCE=expert_out_contrib \
#   bash scripts/run_prune_eval_kimi_gqa.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
DATASET="${DATASET:-gqa}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
START_IDX="${START_IDX:-0}"
SUBSET_SEED="${SUBSET_SEED:-42}"
SCORE_TYPE="${SCORE_TYPE:-activation}"
EMA="${EMA:-0.9}"
COLLECT_CONTRIB="${COLLECT_CONTRIB:-1}"   # set to 1 to enable gradient-based block contrib
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/scores/kimi_gqa}"
MODALITY_AWARE="${MODALITY_AWARE:-0}"
AFFINITY_MODE="${AFFINITY_MODE:-threshold}"   # threshold | scalar
AFFINITY_THRESHOLD="${AFFINITY_THRESHOLD:-0.9}"

EXTRA_ARGS=("$@")

CMD=(
    python src/collect_scores.py
    --model_name_or_path "${MODEL_PATH}"
    --output_dir         "${OUTPUT_DIR}"
    --dataset            "${DATASET}"
    --num_samples        "${NUM_SAMPLES}"
    --batch_size         "${BATCH_SIZE}"
    --start_idx          "${START_IDX}"
    --subset_seed        "${SUBSET_SEED}"
    --score_type         "${SCORE_TYPE}"
    --ema                "${EMA}"
)

if [ "${COLLECT_CONTRIB}" = "1" ]; then
    CMD+=(--collect_contrib)
fi

if [ "${MODALITY_AWARE}" = "1" ]; then
    CMD+=(--modality_aware --affinity_mode "${AFFINITY_MODE}" --affinity_threshold "${AFFINITY_THRESHOLD}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Model      : ${MODEL_PATH}"
echo "Dataset    : ${DATASET} (${NUM_SAMPLES} samples)"
echo "Score type : ${SCORE_TYPE}"
echo "Output     : ${OUTPUT_DIR}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "Modality aware    : ${MODALITY_AWARE}"
echo "Affinity mode     : ${AFFINITY_MODE}"
echo "Affinity threshold: ${AFFINITY_THRESHOLD}"

echo ""
"${CMD[@]}"
