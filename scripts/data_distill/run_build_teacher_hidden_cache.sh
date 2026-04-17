#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/data_distill/teacher_cache}"
TEACHER_LAYER="${TEACHER_LAYER:-0}"
COMPRESSED_LENGTH="${COMPRESSED_LENGTH:-8}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-42}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-8}"
VIDEO_MAX_LONG_SIDE="${VIDEO_MAX_LONG_SIDE:-480}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.extract.build_teacher_hidden_cache
    --model_name_or_path "${MODEL_PATH}"
    --output_dir "${OUTPUT_DIR}"
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
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Model              : ${MODEL_PATH}"
echo "Output             : ${OUTPUT_DIR}"
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
