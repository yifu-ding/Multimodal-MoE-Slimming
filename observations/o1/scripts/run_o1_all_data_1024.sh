#!/usr/bin/env bash
# GQA + COCO 各 2048 条，生成 ema_heatmap_*.png / o1_metrics_*.pt / summary_*.json
# 例: CUDA_VISIBLE_DEVICES=0 bash observations/o1/scripts/run_o1_kimi_gqa_coco_2048.sh
set -euo pipefail

PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
SUBSET_SEED="${SUBSET_SEED:-42}"
EMA_THRESHOLD="${EMA_THRESHOLD:-0.5}"
OUT="${OUTPUT_DIR:-${PREFIX}/observations/o1/results/kimi}"

PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)}"
if [[ -z "${PYTHON}" ]]; then
  echo "需要可导入 torch 的 python，请先激活环境或设置 PYTHON=..." >&2
  exit 1
fi

mkdir -p "${OUT}"

CMD=(
  "${PYTHON}" observations/o1/scripts/run_o1.py
  --model_name_or_path "${MODEL_PATH}"
  --datasets gqa coco m4
  --batch_size "${BATCH_SIZE}"
  --num_samples "${NUM_SAMPLES}"
  --output_dir "${OUT}"
  --artifact-suffix "1024sample"
)
CMD+=("$@")

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT}"
echo "Samples per dataset: ${NUM_SAMPLES}"
"${CMD[@]}"
