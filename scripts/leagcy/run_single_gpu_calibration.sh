#!/usr/bin/env bash
set -euo pipefail

# Single-GPU wrapper around the README calibration command.
# Override defaults via env vars, for example:
#   CUDA_VISIBLE_DEVICES=2 MODEL_PATH=storage/models/Kimi-VL-A3B-Instruct bash scripts/run_single_gpu_calibration.sh

PREFIX="/home/data/dyf/moe-prune"
PYTHONPATH=".:${PYTHONPATH:-}"
export PYTHONPATH

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-5678}"
MODEL_PATH="${MODEL_PATH:-${PREFIX}/storage/models/Kimi-VL-A3B-Instruct}"
SAVE_DIR="${SAVE_DIR:-${PREFIX}/storage/importance}"
DATASET="${DATASET:-gqa}"
LOSS_TYPE="${LOSS_TYPE:-kl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
START_IDX="${START_IDX:-0}"
EXTRA_ARGS=("$@")

accelerate launch \
    --num_processes 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    get_layer_importance_ddp.py \
    --model_name_or_path "${MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --dataset "${DATASET}" \
    --loss_type "${LOSS_TYPE}" \
    --batch_size "${BATCH_SIZE}" \
    --start_idx "${START_IDX}" \
    --num_samples "${NUM_SAMPLES}" \
    "${EXTRA_ARGS[@]}"
