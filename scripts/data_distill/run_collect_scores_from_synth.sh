#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SYNTHETIC_CALIB_PATH="${SYNTHETIC_CALIB_PATH:-${PREFIX}/storage/data_distill/synthetic/synthetic_calib.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/data_distill/scores}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EMA="${EMA:-0.9}"
LOSS_FN="${LOSS_FN:-rel_l2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.runtime.collect_channel_scores_from_synth
    --model_name_or_path "${MODEL_PATH}"
    --synthetic_calib_path "${SYNTHETIC_CALIB_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --ema "${EMA}"
    --loss_fn "${LOSS_FN}"
    --attn_implementation "${ATTN_IMPL}"
    --device_map "${DEVICE_MAP}"
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Model          : ${MODEL_PATH}"
echo "Synthetic calib: ${SYNTHETIC_CALIB_PATH}"
echo "Output         : ${OUTPUT_DIR}"
echo "Batch size     : ${BATCH_SIZE}"
echo "HF_HOME        : ${HF_HOME}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"
