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
# Activation scoring with text/visual split on 128 GQA samples:
#   MODALITY_AWARE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Full run:
#   CUDA_VISIBLE_DEVICES=0 NUM_SAMPLES=512 bash scripts/run_collect_scores.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
DATASET="${DATASET:-gqa}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START_IDX="${START_IDX:-0}"
SUBSET_SEED="${SUBSET_SEED:-42}"

SCORE_TYPE="${SCORE_TYPE:-activation}"
EMA="${EMA:-0.9}"

MODALITY_AWARE="${MODALITY_AWARE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/scores/kimi_gqa-modal2}"

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

if [[ "${MODALITY_AWARE}" == "1" ]]; then
    CMD+=(--modality_aware)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Model      : ${MODEL_PATH}"
echo "Dataset    : ${DATASET} (${NUM_SAMPLES} samples)"
echo "Score type : ${SCORE_TYPE}"
echo "Modality   : $([[ "${MODALITY_AWARE}" == "1" ]] && echo "text+visual" || echo "disabled")"
echo "Output     : ${OUTPUT_DIR}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo ""
"${CMD[@]}"
