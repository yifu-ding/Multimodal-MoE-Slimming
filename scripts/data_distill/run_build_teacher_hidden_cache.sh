#!/usr/bin/env bash
set -euo pipefail

source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu
PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-${PREFIX}/storage/data_distill/attn_weighted-$(date +%m%d%H%M%S)}/teacher_cache.pt"
TEACHER_LAYER="${TEACHER_LAYER:-0}"
COMPRESSED_LENGTH="${COMPRESSED_LENGTH:-256}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-2048}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-42}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-8}"
VIDEO_MAX_LONG_SIDE="${VIDEO_MAX_LONG_SIDE:-480}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

TEACHER_DATASETS="${TEACHER_DATASETS:-gqa}"
#   使用方式
#   # 均匀随机采样（默认���
#   --compression_mode sample
#   # Attention 加权采样
#   --compression_mode attention_weighted --attn_temperature 1.0
#   # 更 greedy 的 attention 采样（更偏向高 attention token）
#   --compression_mode attention_weighted --attn_temperature 0.5
#   # 旧的 mean pooling
#   --compression_mode pool
COMPRESSION_MODE="${COMPRESSION_MODE:-attention_weighted}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.build_teacher_hidden_cache
    --model_name_or_path "${MODEL_PATH}"
    --output_path "${OUTPUT_PATH}"
    --teacher_layer "${TEACHER_LAYER}"
    --compressed_length "${COMPRESSED_LENGTH}"
    --samples_per_dataset "${SAMPLES_PER_DATASET}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --shuffle_seed "${SHUFFLE_SEED}"
    --num_video_frames "${NUM_VIDEO_FRAMES}"
    --video_max_long_side "${VIDEO_MAX_LONG_SIDE}"
    --save_dtype "${SAVE_DTYPE}"
    --attn_implementation "${ATTN_IMPL}"
    --device_map "${DEVICE_MAP}"
    --teacher_datasets ${TEACHER_DATASETS}
    --compression_mode ${COMPRESSION_MODE}
    --modality_aware_compression
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Model              : ${MODEL_PATH}"
echo "Output             : ${OUTPUT_PATH}"
echo "Teacher layer      : ${TEACHER_LAYER}"
echo "Compressed length  : ${COMPRESSED_LENGTH}"
echo "Samples/dataset    : ${SAMPLES_PER_DATASET}"
echo "Batch size         : ${BATCH_SIZE}"
echo "Video frames       : ${NUM_VIDEO_FRAMES}"
echo "HF_HOME            : ${HF_HOME}"
echo "Device map         : ${DEVICE_MAP}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"
