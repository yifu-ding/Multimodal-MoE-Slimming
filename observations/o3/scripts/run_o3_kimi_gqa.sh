#!/usr/bin/env bash
set -euo pipefail

# Single-GPU wrapper for O3 (Conflict score analysis) on GQA.

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
DATASET="${DATASET:-gqa}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START_IDX="${START_IDX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
HIGH_CONFLICT_THRESHOLD="${HIGH_CONFLICT_THRESHOLD:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/observations/o3/results/kimi_gqa}"
RAW_STATS_PATH="${RAW_STATS_PATH:-}"

EXTRA_ARGS=("$@")

if [[ "${DATASET}" != "gqa" ]]; then
    echo "This wrapper is intended for GQA. Got DATASET=${DATASET}." >&2
    exit 1
fi

CMD=(
    python observations/o3/scripts/run_o3.py
    --model_name_or_path "${MODEL_PATH}"
    --dataset "${DATASET}"
    --batch_size "${BATCH_SIZE}"
    --start_idx "${START_IDX}"
    --num_samples "${NUM_SAMPLES}"
    --output_dir "${OUTPUT_DIR}"
    --high_conflict_threshold "${HIGH_CONFLICT_THRESHOLD}"
)

if [[ -n "${RAW_STATS_PATH}" ]]; then
    CMD+=(--raw_stats_path "${RAW_STATS_PATH}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running O3 with model: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
"${CMD[@]}"
