#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

TEACHER_CACHE_PATH="${TEACHER_CACHE_PATH:-${PREFIX}/storage/data_distill/teacher_cache/teacher_hidden_cache.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-${PREFIX}/storage/data_distill/synthetic/synthetic_calib.pt}"
SYNTHETIC_SIZE="${SYNTHETIC_SIZE:-256}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-1024}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
LR="${LR:-1e-2}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
INIT_STD="${INIT_STD:-1e-3}"
LAMBDA_MEAN="${LAMBDA_MEAN:-1.0}"
LAMBDA_VAR="${LAMBDA_VAR:-1.0}"
LAMBDA_POS="${LAMBDA_POS:-1.0}"
LAMBDA_ENERGY="${LAMBDA_ENERGY:-0.5}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.distill.distill_synthetic_hidden
    --teacher_cache_path "${TEACHER_CACHE_PATH}"
    --output_path "${OUTPUT_PATH}"
    --synthetic_size "${SYNTHETIC_SIZE}"
    --teacher_batch_size "${TEACHER_BATCH_SIZE}"
    --train_steps "${TRAIN_STEPS}"
    --lr "${LR}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --init_std "${INIT_STD}"
    --lambda_mean "${LAMBDA_MEAN}"
    --lambda_var "${LAMBDA_VAR}"
    --lambda_pos "${LAMBDA_POS}"
    --lambda_energy "${LAMBDA_ENERGY}"
    --log_interval "${LOG_INTERVAL}"
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Teacher cache : ${TEACHER_CACHE_PATH}"
echo "Output        : ${OUTPUT_PATH}"
echo "Synthetic M   : ${SYNTHETIC_SIZE}"
echo "Train steps   : ${TRAIN_STEPS}"
echo "HF_HOME       : ${HF_HOME}"
echo "Device        : ${DEVICE}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"
