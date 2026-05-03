#!/usr/bin/env bash
set -euo pipefail
source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

resolve_storage_path() {
    local path="$1"
    local broken_prefix="${PREFIX}/storage"
    local fallback_prefix="/home/data/dyf/MARS-results/storage"
    if [[ -n "${path}" && ! -e "${path}" && "${path}" == "${broken_prefix}"* ]]; then
        local suffix="${path#${broken_prefix}}"
        local candidate="${fallback_prefix}${suffix}"
        if [[ -e "${candidate}" || -d "$(dirname "${candidate}")" ]]; then
            echo "${candidate}"
            return 0
        fi
    fi
    echo "${path}"
}

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH:-}"
HIDDEN_PAYLOAD_PATH="$(resolve_storage_path "${HIDDEN_PAYLOAD_PATH}")"
OUTPUT_PATH="${OUTPUT_PATH:-${HIDDEN_PAYLOAD_PATH%.pt}-scores.pt}"
OUTPUT_PATH="$(resolve_storage_path "${OUTPUT_PATH}")"
BATCH_SIZE="${BATCH_SIZE:-8}"
EMA="${EMA:-0.9}"
LOSS_FN="${LOSS_FN:-rel_l2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

EXTRA_ARGS=("$@")

START_LAYER_INFO=""
if [[ -n "${HIDDEN_PAYLOAD_PATH}" && -f "${HIDDEN_PAYLOAD_PATH}" ]]; then
    START_LAYER_INFO="$(
        HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH}" python - <<'PY'
import os
import torch
from src.calibration.representation_distill.common import resolve_hidden_start_layer

path = os.environ["HIDDEN_PAYLOAD_PATH"]
payload = torch.load(path, map_location="cpu", weights_only=False)
meta = payload.get("metadata", {})
teacher_layer = meta.get("teacher_layer")
teacher_layer_type = meta.get("teacher_layer_type")
if teacher_layer is None:
    print("unknown")
else:
    print(resolve_hidden_start_layer(meta))
PY
    )"
fi

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
echo "Start layer    : ${START_LAYER_INFO:-unknown}"
echo "Output         : ${OUTPUT_PATH}"
echo "Batch size     : ${BATCH_SIZE}"
echo "HF_HOME        : ${HF_HOME}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"


# storage/data_distill/teacher_cache-attn_weighted/distilled/distilled_hidden.pt
