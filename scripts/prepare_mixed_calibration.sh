#!/usr/bin/env bash
set -euo pipefail

# Phase 1: freeze a mixed GQA/COCO/M4-Instruct/Video-MMMU calibration set.
# The model processes the full valid sequence. SCORE_TOKENS_PER_SAMPLE is the
# fixed quota used later for sensitivity/loss/Hessian statistics, not truncation.

source scripts/select_least_used_gpu.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
SCORE_TOKENS_PER_SAMPLE="${SCORE_TOKENS_PER_SAMPLE:-2048}"
MIN_SAMPLE_TOKENS="${MIN_SAMPLE_TOKENS:-${SCORE_TOKENS_PER_SAMPLE}}"
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
CMD+=("$@")

echo "Model      : ${MODEL_PATH}"
echo "Candidates : ${CANDIDATE_POOL_SIZE}"
echo "Selected   : ${NUM_SAMPLES}"
echo "Score quota: ${SCORE_TOKENS_PER_SAMPLE} tokens/sample"
echo "Manifest   : ${OUTPUT_MANIFEST}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"
