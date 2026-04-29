#!/usr/bin/env bash
set -euo pipefail
source scripts/select_least_used_gpu.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
RESUME_FROM="${RESUME_FROM:-}"
LATEST_DISTILLED_LINK_DIR="${LATEST_DISTILLED_LINK_DIR:-${PREFIX}/storage/data_distill_kimi/online-distilled-latest}"

TEACHER_LAYER="${TEACHER_LAYER:-0}"
COMPRESSED_LENGTH="${COMPRESSED_LENGTH:-1024}"
COMPRESSION_MODE="${COMPRESSION_MODE:-sample}"
MODALITY_AWARE_COMPRESSION="${MODALITY_AWARE_COMPRESSION:-1}"
ATTN_TEMPERATURE="${ATTN_TEMPERATURE:-1.0}"

TEACHER_DATASETS=(${TEACHER_DATASETS:-gqa coco m4_instruct})
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-2048}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-64}"
TOKEN_PER_SAMPLE="${TOKEN_PER_SAMPLE:-2048}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-8}"
VIDEO_MAX_LONG_SIDE="${VIDEO_MAX_LONG_SIDE:-480}"

SYNTHETIC_SIZE="${SYNTHETIC_SIZE:-1024}"
SYNTHETIC_BATCH_SIZE="${SYNTHETIC_BATCH_SIZE:-64}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
LR="${LR:-5e-3}"
INIT_STD="${INIT_STD:-0.0}"

LAMBDA_MMD="${LAMBDA_MMD:-1.0}"
LAMBDA_COV="${LAMBDA_COV:-2.0}"
LAMBDA_DIV="${LAMBDA_DIV:-0.002}"
LAMBDA_MEAN="${LAMBDA_MEAN:-1.0}"
LAMBDA_VAR="${LAMBDA_VAR:-2.0}"
LAMBDA_BLOCK="${LAMBDA_BLOCK:-0.25}"

USE_EMA_NORMALIZED_LOSSES="${USE_EMA_NORMALIZED_LOSSES:-1}"  # 不一定？
LOSS_EMA_DECAY="${LOSS_EMA_DECAY:-0.99}"
DIV_WARMUP_STEPS="${DIV_WARMUP_STEPS:-0}"
MMD_SUBSAMPLE="${MMD_SUBSAMPLE:-2048}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-5}"

SEED="${SEED:-42}"
SUBSET_SEED="${SUBSET_SEED:-42}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TRAIN_DTYPE="${TRAIN_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-${TRAIN_DEVICE}}"

WANDB_PROJECT="${WANDB_PROJECT:-maes}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_EVERY_N_STEPS="${WANDB_EVERY_N_STEPS:-10}"
DIVERSITY_ABLATION="${DIVERSITY_ABLATION:-no_div}"  # no_div, full
DISTRIBUTION_ABLATION="${DISTRIBUTION_ABLATION:-mmd_only}"  # moment_only, mmd_only, full   
ABLATION_TAG="div-${DIVERSITY_ABLATION}_dist-${DISTRIBUTION_ABLATION}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-online-distilled-${RUN_STAMP}-${ABLATION_TAG}}"
OUTPUT_PATH="${OUTPUT_PATH:-${PREFIX}/storage/data_distill_kimi/online-distilled-${RUN_STAMP}-${ABLATION_TAG}/distilled_hidden-step${TRAIN_STEPS}.pt}"
RESET_OPTIMIZER_ON_RESUME="${RESET_OPTIMIZER_ON_RESUME:-0}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.distill_synthetic_hidden_online
    --model_name_or_path "${MODEL_PATH}"
    --output_path "${OUTPUT_PATH}"
    --teacher_layer "${TEACHER_LAYER}"
    --compressed_length "${COMPRESSED_LENGTH}"
    --compression_mode "${COMPRESSION_MODE}"
    --attn_temperature "${ATTN_TEMPERATURE}"
    --samples_per_dataset "${SAMPLES_PER_DATASET}"
    --teacher_datasets "${TEACHER_DATASETS[@]}"
    --teacher_batch_size "${TEACHER_BATCH_SIZE}"
    --synthetic_size "${SYNTHETIC_SIZE}"
    --synthetic_batch_size "${SYNTHETIC_BATCH_SIZE}"
    --train_steps "${TRAIN_STEPS}"
    --lr "${LR}"
    --seed "${SEED}"
    --subset_seed "${SUBSET_SEED}"
    --shuffle_seed "${SHUFFLE_SEED}"
    --num_video_frames "${NUM_VIDEO_FRAMES}"
    --video_max_long_side "${VIDEO_MAX_LONG_SIDE}"
    --token_per_sample "${TOKEN_PER_SAMPLE}"
    --device "${TRAIN_DEVICE}"
    --device_map "${DEVICE_MAP}"
    --attn_implementation "${ATTN_IMPL}"
    --train_dtype "${TRAIN_DTYPE}"
    --init_std "${INIT_STD}"
    --lambda_mmd "${LAMBDA_MMD}"
    --lambda_cov "${LAMBDA_COV}"
    --lambda_div "${LAMBDA_DIV}"
    --lambda_mean "${LAMBDA_MEAN}"
    --lambda_var "${LAMBDA_VAR}"
    --lambda_block "${LAMBDA_BLOCK}"
    --diversity_ablation "${DIVERSITY_ABLATION}"
    --distribution_ablation "${DISTRIBUTION_ABLATION}"
    --div_warmup_steps "${DIV_WARMUP_STEPS}"
    --mmd_subsample "${MMD_SUBSAMPLE}"
    --log_interval "${LOG_INTERVAL}"
    --wandb_every_n_steps "${WANDB_EVERY_N_STEPS}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_run_name "${WANDB_RUN_NAME}"
    --wandb_mode "${WANDB_MODE}"
    --loss_ema_decay "${LOSS_EMA_DECAY}"
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"
    --max_checkpoints_to_keep "${MAX_CHECKPOINTS_TO_KEEP}"
)

if [[ -n "${RESUME_FROM}" ]]; then
    CMD+=(--resume_from "${RESUME_FROM}")
fi

if [[ "${RESET_OPTIMIZER_ON_RESUME}" == "1" ]]; then
    CMD+=(--reset_optimizer_on_resume)
fi

if [[ "${MODALITY_AWARE_COMPRESSION}" == "1" ]]; then
    CMD+=(--modality_aware_compression)
fi

if [[ "${USE_EMA_NORMALIZED_LOSSES}" == "1" ]]; then
    CMD+=(--use_ema_normalized_losses)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Model           : ${MODEL_PATH}"
echo "Output          : ${OUTPUT_PATH}"
echo "Resume from     : ${RESUME_FROM:-<none>}"
echo "Teacher layer   : ${TEACHER_LAYER}"
echo "Teacher data    : ${TEACHER_DATASETS[*]}"
echo "Samples/dataset : ${SAMPLES_PER_DATASET}"
echo "Teacher batch   : ${TEACHER_BATCH_SIZE}"
echo "Synthetic M     : ${SYNTHETIC_SIZE}"
echo "Synthetic batch : ${SYNTHETIC_BATCH_SIZE}"
echo "Train steps     : ${TRAIN_STEPS}"
echo "Checkpoint every: ${CHECKPOINT_INTERVAL}"
echo "Keep checkpoints: ${MAX_CHECKPOINTS_TO_KEEP}"
echo "LR              : ${LR}"
echo "Train dtype     : ${TRAIN_DTYPE}"
echo "Compression     : mode=${COMPRESSION_MODE} modality_aware=${MODALITY_AWARE_COMPRESSION} length=${COMPRESSED_LENGTH}"
echo "Lambdas         : mmd=${LAMBDA_MMD} cov=${LAMBDA_COV} mean=${LAMBDA_MEAN} var=${LAMBDA_VAR} div=${LAMBDA_DIV} block=${LAMBDA_BLOCK}"
echo "Ablation        : diversity=${DIVERSITY_ABLATION} distribution=${DISTRIBUTION_ABLATION}"
echo "Wandb           : project=${WANDB_PROJECT} run=${WANDB_RUN_NAME} mode=${WANDB_MODE} every ${WANDB_EVERY_N_STEPS} step(s)"
echo "HF_HOME         : ${HF_HOME}"
echo "Device map      : ${DEVICE_MAP}"
echo "Train device    : ${TRAIN_DEVICE}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"

ln -sfn "$(dirname "${OUTPUT_PATH}")" "${LATEST_DISTILLED_LINK_DIR}"
echo "Latest distilled: ${LATEST_DISTILLED_LINK_DIR}"
