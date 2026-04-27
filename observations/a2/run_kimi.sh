#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
DATASET="${DATASET:-gqa}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START_IDX="${START_IDX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
OBSERVE_LAYERS="${OBSERVE_LAYERS:-5,10,15,20}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
DEVICE_MAP="${DEVICE_MAP:-single_gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/observations/a2/results/kimi}"
SAVE_NAME="${SAVE_NAME:-}"

PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)}"
if [[ -z "${PYTHON}" ]]; then
  echo "需要可导入 torch 的 python，请先激活环境或设置 PYTHON=..." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON}" observations/a2/collect_kimi_channel_kde_data.py
  --model_name_or_path "${MODEL_PATH}"
  --dataset "${DATASET}"
  --batch_size "${BATCH_SIZE}"
  --start_idx "${START_IDX}"
  --num_samples "${NUM_SAMPLES}"
  --observe_layers "${OBSERVE_LAYERS}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --device_map "${DEVICE_MAP}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${SAVE_NAME}" ]]; then
  CMD+=(--save_name "${SAVE_NAME}")
fi

CMD+=("$@")

echo "Model:   ${MODEL_PATH}"
echo "Dataset: ${DATASET}"
echo "Layers:  ${OBSERVE_LAYERS}"
echo "Samples: ${NUM_SAMPLES} (start=${START_IDX})"
echo "Out:     ${OUTPUT_DIR}"
"${CMD[@]}"
