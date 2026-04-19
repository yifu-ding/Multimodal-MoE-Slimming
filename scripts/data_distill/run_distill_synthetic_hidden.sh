#!/usr/bin/env bash
set -euo pipefail
source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH:-${PREFIX}/storage/data_distill/teacher_cache/teacher_hidden_cache.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-$(dirname "${HIDDEN_PAYLOAD_PATH}")/distilled/distilled_hidden.pt}"

SYNTHETIC_SIZE="${SYNTHETIC_SIZE:-1024}"  # 合成 hidden 的 token 数
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-2048}"  # 每步多少条教师序列作为样本参与 loss 计算
TRAIN_STEPS="${TRAIN_STEPS:-6000}"  # 训练多少步
LR="${LR:-1e-2}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
INIT_STD="${INIT_STD:-1e-3}"
LAMBDA_MMD="${LAMBDA_MMD:-1.0}"
LAMBDA_COV="${LAMBDA_COV:-0.1}"
LAMBDA_DIV="${LAMBDA_DIV:-0.1}"
LAMBDA_MEAN="${LAMBDA_MEAN:-0.5}"
LAMBDA_VAR="${LAMBDA_VAR:-0.5}"
MMD_SUBSAMPLE="${MMD_SUBSAMPLE:-2048}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.distill_synthetic_hidden
    --teacher_cache_path "${HIDDEN_PAYLOAD_PATH}"
    --output_path "${OUTPUT_PATH}"
    --synthetic_size "${SYNTHETIC_SIZE}"
    --teacher_batch_size "${TEACHER_BATCH_SIZE}"
    --train_steps "${TRAIN_STEPS}"
    --lr "${LR}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --init_std "${INIT_STD}"
    --lambda_mmd "${LAMBDA_MMD}"
    --lambda_cov "${LAMBDA_COV}"
    --lambda_div "${LAMBDA_DIV}"
    --lambda_mean "${LAMBDA_MEAN}"
    --lambda_var "${LAMBDA_VAR}"
    --mmd_subsample "${MMD_SUBSAMPLE}"
    --log_interval "${LOG_INTERVAL}"
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Teacher cache : ${HIDDEN_PAYLOAD_PATH}"
echo "Output        : ${OUTPUT_PATH}"
echo "Synthetic M   : ${SYNTHETIC_SIZE}"
echo "Train steps   : ${TRAIN_STEPS}"
echo "HF_HOME       : ${HF_HOME}"
echo "Device        : ${DEVICE}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"
