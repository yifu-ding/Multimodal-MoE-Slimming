#!/usr/bin/env bash
set -euo pipefail

# Phase 1: freeze a mixed GQA/COCO/M4-Instruct/Video-MMMU calibration set.
# The model processes the full valid sequence. SCORE_TOKENS_PER_SAMPLE is the
# fixed quota used later for sensitivity/loss/Hessian statistics, not truncation.

source scripts/select_least_used_gpu.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${MAES_HF_HOME:-/home/data/dyf/hf_cache}"
export HF_HUB_CACHE="${MAES_HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${MAES_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
SCORE_TOKENS_PER_SAMPLE="${SCORE_TOKENS_PER_SAMPLE:-2048}"
TOTAL_SCORE_TOKENS="${TOTAL_SCORE_TOKENS:-}"
if [[ -n "${TOTAL_SCORE_TOKENS}" ]]; then
    MIN_SAMPLE_TOKENS="${MIN_SAMPLE_TOKENS:-64}"
else
    MIN_SAMPLE_TOKENS="${MIN_SAMPLE_TOKENS:-${SCORE_TOKENS_PER_SAMPLE}}"
fi
FEATURE_LAYER="${FEATURE_LAYER:-0}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-2}"
MIN_MODALITY_TOKENS="${MIN_MODALITY_TOKENS:-8}"
PCA_DIM="${PCA_DIM:-50}"
PERPLEXITY="${PERPLEXITY:-40}"
SEED="${SEED:-42}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${PREFIX}/storage/calibration_manifests/mixed-${NUM_SAMPLES}-seed${SEED}.json}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-8}"
VIDEO_MAX_LONG_SIDE="${VIDEO_MAX_LONG_SIDE:-480}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-}"

case "${MODEL_PATH,,}" in
    kimi-vl-a3b|kimi-vl-a3b-instruct)
        MODEL_PATH="moonshotai/Kimi-VL-A3B-Instruct"
        ;;
    qwen3-vl-30b-a3b|qwen3-vl-30b-a3b-instruct)
        MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct"
        ;;
    deepseek-vl2-small)
        MODEL_PATH="deepseek-ai/deepseek-vl2-small"
        ;;
    internvl3_5-30b-a3b-hf)
        MODEL_PATH="OpenGVLab/InternVL3_5-30B-A3B-HF"
        ;;
esac

CMD=(
    python -m src.calibration.prepare_mixed_calibration
    --model_name_or_path "${MODEL_PATH}"
    --output_manifest "${OUTPUT_MANIFEST}"
    --datasets gqa coco m4_instruct video_mmmu
    --candidate_pool_size "${CANDIDATE_POOL_SIZE}"
    --num_samples "${NUM_SAMPLES}"
    --score_tokens_per_sample "${SCORE_TOKENS_PER_SAMPLE}"
    --min_sample_tokens "${MIN_SAMPLE_TOKENS}"
    --feature_layer "${FEATURE_LAYER}"
    --feature_batch_size "${FEATURE_BATCH_SIZE}"
    --min_modality_tokens "${MIN_MODALITY_TOKENS}"
    --pca_dim "${PCA_DIM}"
    --perplexity "${PERPLEXITY}"
    --seed "${SEED}"
    --num_video_frames "${NUM_VIDEO_FRAMES}"
    --video_max_long_side "${VIDEO_MAX_LONG_SIDE}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
)
if [[ -n "${TOTAL_SCORE_TOKENS}" ]]; then
    CMD+=(--score_token_budget "${TOTAL_SCORE_TOKENS}")
fi
if [[ -n "${DEVICE_MAP}" ]]; then
    CMD+=(--device_map "${DEVICE_MAP}")
fi
CMD+=("$@")

echo "Model      : ${MODEL_PATH}"
echo "Candidates : ${CANDIDATE_POOL_SIZE}"
echo "Selected   : ${NUM_SAMPLES}"
if [[ -n "${TOTAL_SCORE_TOKENS}" ]]; then
    echo "Score quota: ${TOTAL_SCORE_TOKENS} total tokens (variable per sample, min sample length=${MIN_SAMPLE_TOKENS})"
else
    echo "Score quota: ${SCORE_TOKENS_PER_SAMPLE} tokens/sample"
fi
echo "Manifest   : ${OUTPUT_MANIFEST}"
echo "Device map : ${DEVICE_MAP:-cuda:0 (default)}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"
