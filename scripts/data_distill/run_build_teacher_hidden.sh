#!/usr/bin/env bash
set -euo pipefail

source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu
PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
TEACHER_LAYER="${TEACHER_LAYER:-0}"
COMPRESSED_LENGTH="${COMPRESSED_LENGTH:-2048}"  # sequence length  # 只压缩 seqlen
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-1024}"  # 多少条，这个在cache过程中是不变的（不压缩条数）
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-42}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-8}"
VIDEO_MAX_LONG_SIDE="${VIDEO_MAX_LONG_SIDE:-480}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
CACHE_NEXT_BLOCK_TARGETS="${CACHE_NEXT_BLOCK_TARGETS:-1}"

TEACHER_DATASETS="${TEACHER_DATASETS:-gqa coco m4_instruct}"
#   使用方式
#   # 均匀随机采样（默认）
#   --compression_mode sample
#   # Attention 加权采样
#   --compression_mode attention_weighted --attn_temperature 1.0
#   # 更 greedy 的 attention 采样（更偏向高 attention token）
#   --compression_mode attention_weighted --attn_temperature 0.5
#   # 旧的 mean pooling
#   --compression_mode pool
COMPRESSION_MODE="${COMPRESSION_MODE:-sample}"
ATTN_TEMPERATURE="${ATTN_TEMPERATURE:-1.0}"

read -r -a _teacher_ds_arr <<< "${TEACHER_DATASETS}"
if (( ${#_teacher_ds_arr[@]} > 1 )); then
    TEACHER_DATASETS_PATH_LABEL=mixed
else
    TEACHER_DATASETS_PATH_LABEL="${TEACHER_DATASETS}"
fi

OUTPUT_PATH="${OUTPUT_PATH:-${PREFIX}/storage/data_distill_kimi/${TEACHER_DATASETS_PATH_LABEL}-num_${SAMPLES_PER_DATASET}-token_${COMPRESSED_LENGTH}-${COMPRESSION_MODE}_at${ATTN_TEMPERATURE}-$(date +%m%d%H%M%S)}/teacher_hidden.pt"
LATEST_LINK_DIR="${LATEST_LINK_DIR:-${PREFIX}/storage/data_distill_kimi/${TEACHER_DATASETS_PATH_LABEL}-num_${SAMPLES_PER_DATASET}-token_${COMPRESSED_LENGTH}-${COMPRESSION_MODE}_at${ATTN_TEMPERATURE}-latest}"

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
    --attn_temperature "${ATTN_TEMPERATURE}"
)

if [[ "${CACHE_NEXT_BLOCK_TARGETS}" == "1" ]]; then
    CMD+=(--cache_next_block_targets)
fi

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
echo "Cache next block   : ${CACHE_NEXT_BLOCK_TARGETS}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"

mkdir -p "$(dirname "${LATEST_LINK_DIR}")"
ln -sfn "$(dirname "${OUTPUT_PATH}")" "${LATEST_LINK_DIR}"
echo "Latest link        : ${LATEST_LINK_DIR}"
echo "HIDDEN_PAYLOAD_PATH=${OUTPUT_PATH}"
