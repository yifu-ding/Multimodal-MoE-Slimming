#!/usr/bin/env bash
set -euo pipefail
source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH:-${PREFIX}/storage/data_distill_kimi/gqa-sample_at1.0-latest/distilled-latest/distilled_hidden.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-${HIDDEN_PAYLOAD_PATH%.pt}-scores.pt}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EMA="${EMA:-0.9}"
LOSS_FN="${LOSS_FN:-rel_l2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.collect_channel_scores_from_synth
    --model_name_or_path "${MODEL_PATH}"
    --input_hidden_path "${HIDDEN_PAYLOAD_PATH}"
    --output_path "${OUTPUT_PATH}"
    --batch_size "${BATCH_SIZE}"
    --ema "${EMA}"
    --loss_fn "${LOSS_FN}"
    --attn_implementation "${ATTN_IMPL}"
    --device_map "${DEVICE_MAP}"
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Model          : ${MODEL_PATH}"
echo "Hidden payload : ${HIDDEN_PAYLOAD_PATH}"
echo "Output         : ${OUTPUT_PATH}"
echo "Batch size     : ${BATCH_SIZE}"
echo "HF_HOME        : ${HF_HOME}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"


# /home/dyf/code/distill/MoDES/storage/data_distill/teacher_cache-attn_weighted/distilled/distilled_hidden.pt
