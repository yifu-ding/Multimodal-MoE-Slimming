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
# Activation scoring on 128 COCO2017-Capval samples:
#   DATASET=coco CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Activation scoring on 128 VMMMU samples:
#   DATASET=VMMMU CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Activation scoring with text/visual split on 128 GQA samples:
#   MODALITY_AWARE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Full run:
#   CUDA_VISIBLE_DEVICES=0 NUM_SAMPLES=512 bash scripts/run_collect_scores.sh

source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
DATASET="${DATASET:-gqa}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
START_IDX="${START_IDX:-0}"
SUBSET_SEED="${SUBSET_SEED:-42}"

EMA="${EMA:-0.9}"

case "${DATASET,,}" in
    gqa)
        DATASET="gqa"
        DATASET_TAG="gqa"
        ;;
    coco)
        DATASET="coco"
        DATASET_TAG="coco"
        ;;
    vmmmu|video_mmmu)
        DATASET="video_mmmu"
        DATASET_TAG="video_mmmu"
        ;;
    *)
        echo "Unsupported DATASET=${DATASET}. Supported values: gqa, coco, VMMMU (video_mmmu)." >&2
        exit 1
        ;;
esac

# OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/scores/kimi_${DATASET_TAG}-second_order}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/scores/kimi_${DATASET_TAG}-rell2-$(date +%m%d%H%M)}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.collect_scores_main
    --model_name_or_path "${MODEL_PATH}"
    --output_dir         "${OUTPUT_DIR}"
    --dataset            "${DATASET}"
    --num_samples        "${NUM_SAMPLES}"
    --batch_size         "${BATCH_SIZE}"
    --start_idx          "${START_IDX}"
    --subset_seed        "${SUBSET_SEED}"
    --ema                "${EMA}"
)

# --force if you want to recompute the scores and overwrite the existing ones

CMD+=("${EXTRA_ARGS[@]}")

echo "Model      : ${MODEL_PATH}"
echo "Dataset    : ${DATASET} (${NUM_SAMPLES} samples)"
echo "Output     : ${OUTPUT_DIR}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo ""
"${CMD[@]}"
