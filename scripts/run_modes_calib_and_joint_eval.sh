#!/usr/bin/env bash
set -euo pipefail

# Run MoDES calibration + frontier search + joint MAES pruning eval.
#
# Steps
# -----
# 1. MoDES layer-importance calibration -> LAYER_IMPORTANCE_PATH
# 2. MoDES frontier search             -> TAU_SKIP_PATH
# 3. Joint eval with MAES pruning + MoDES tau-skip
#
# Example
# -------
# USE_LMMS_EVAL=1 \
# SCORES_PATH=/home/dyf/code/distill/MAES/storage/data_distill_kimi/mixed-num_342-token_2048-sample_at1.0-0423143643/teacher_hidden-scores.pt \
# PRUNE_RATIO=0.5 \
# bash scripts/run_modes_calib_and_joint_eval.sh

source scripts/select_least_used_gpu.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}:${PREFIX}/lmms-eval"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
DATASET="${DATASET:-gqa}"
SCORES_PATH="${SCORES_PATH:-}"
PRUNE_RATIO="${PRUNE_RATIO:-0.5}"
SWEEP_TASKS="${SWEEP_TASKS:-coco2017cap mmstar mmbench realworldqa gqa mme textvqa chartqa}"
USE_LMMS_EVAL="${USE_LMMS_EVAL:-1}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-5678}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_gpu_count=$(python - <<'PY'
import os
raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not raw:
    print(0)
else:
    print(len([x for x in raw.split(",") if x.strip()]))
PY
)
    if [[ "${visible_gpu_count}" -gt 0 && "${NUM_PROCESSES}" -gt "${visible_gpu_count}" ]]; then
        echo "[warn] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${visible_gpu_count} GPU(s), but NUM_PROCESSES=${NUM_PROCESSES}. Clamping NUM_PROCESSES to ${visible_gpu_count}."
        NUM_PROCESSES="${visible_gpu_count}"
    fi
fi

MODES_NUM_SAMPLES="${MODES_NUM_SAMPLES:-1024}"
MODES_BATCH_SIZE="${MODES_BATCH_SIZE:-8}"
MODES_START_IDX="${MODES_START_IDX:-0}"
MODES_LOSS_TYPE="${MODES_LOSS_TYPE:-kl}"
MODES_TEMPERATURE="${MODES_TEMPERATURE:-1.0}"
SEARCH_BATCH_SIZE="${SEARCH_BATCH_SIZE:-${MODES_BATCH_SIZE}}"
SEARCH_ATTN_IMPLEMENTATION="${SEARCH_ATTN_IMPLEMENTATION:-flash_attention_2}"

TARGET_SKIP_PROPORTION="${TARGET_SKIP_PROPORTION:-0.8}"
GRID_NUM="${GRID_NUM:-100}"
GRID_MAP="${GRID_MAP:-exp}"
EXP_COEFF="${EXP_COEFF:-10}"
TEXT_TAU_MIN="${TEXT_TAU_MIN:-0.0}"
TEXT_TAU_MAX="${TEXT_TAU_MAX:-0.71}"
VISUAL_TAU_MIN="${VISUAL_TAU_MIN:-0.0}"
VISUAL_TAU_MAX="${VISUAL_TAU_MAX:-0.6}"

EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
START_IDX="${START_IDX:-0}"
SUBSET_SEED="${SUBSET_SEED:-}"

FORCE="${FORCE:-0}"
SKIP_CALIB="${SKIP_CALIB:-0}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"
MODEL_DIR_NAME="${MODEL_PATH##*/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PREFIX}/storage/modes_joint/${MODEL_DIR_NAME}/${DATASET}/${RUN_TAG}}"
LAYER_IMPORTANCE_DIR="${LAYER_IMPORTANCE_DIR:-${OUTPUT_ROOT}/layer_importance}"
TAU_SEARCH_DIR="${TAU_SEARCH_DIR:-${OUTPUT_ROOT}/tau_search}"
JOINT_EVAL_OUTPUT_DIR="${JOINT_EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/joint_eval}"

LAYER_IMPORTANCE_PATH_DEFAULT="${LAYER_IMPORTANCE_DIR}/${DATASET}/${MODEL_DIR_NAME}/${MODES_LOSS_TYPE}_${MODES_START_IDX}_${MODES_NUM_SAMPLES}.pkl"
if [[ "${EXP_COEFF}" != "10" ]]; then
    TAU_SKIP_PATH_DEFAULT="${TAU_SEARCH_DIR}/${DATASET}/${MODEL_DIR_NAME}/${MODES_LOSS_TYPE}_${MODES_START_IDX}_${MODES_NUM_SAMPLES}_${TARGET_SKIP_PROPORTION}_text_${TEXT_TAU_MIN}_${TEXT_TAU_MAX}_visual_${VISUAL_TAU_MIN}_${VISUAL_TAU_MAX}_grid${GRID_NUM}_exp${EXP_COEFF}.pkl"
else
    TAU_SKIP_PATH_DEFAULT="${TAU_SEARCH_DIR}/${DATASET}/${MODEL_DIR_NAME}/${MODES_LOSS_TYPE}_${MODES_START_IDX}_${MODES_NUM_SAMPLES}_${TARGET_SKIP_PROPORTION}_text_${TEXT_TAU_MIN}_${TEXT_TAU_MAX}_visual_${VISUAL_TAU_MIN}_${VISUAL_TAU_MAX}_grid${GRID_NUM}.pkl"
fi

LAYER_IMPORTANCE_PATH="${LAYER_IMPORTANCE_PATH:-${LAYER_IMPORTANCE_PATH_DEFAULT}}"
TAU_SKIP_PATH="${TAU_SKIP_PATH:-${TAU_SKIP_PATH_DEFAULT}}"
EXPERT_IMPORTANCE_PATH="${EXPERT_IMPORTANCE_PATH:-}"

mkdir -p "${OUTPUT_ROOT}" "${LAYER_IMPORTANCE_DIR}" "${TAU_SEARCH_DIR}" "${JOINT_EVAL_OUTPUT_DIR}"

if [[ -z "${SCORES_PATH}" ]]; then
    echo "error: SCORES_PATH is required for joint MAES pruning eval." >&2
    exit 1
fi

run_cmd() {
    echo ""
    echo "CMD: $*"
    echo ""
    "$@"
}

echo "Model                : ${MODEL_PATH}"
echo "Dataset              : ${DATASET}"
echo "Sweep tasks          : ${SWEEP_TASKS}"
echo "MAES scores          : ${SCORES_PATH}"
echo "Prune ratio          : ${PRUNE_RATIO}"
echo "Layer importance out : ${LAYER_IMPORTANCE_PATH}"
echo "Tau search out       : ${TAU_SKIP_PATH}"
echo "Eval output dir      : ${JOINT_EVAL_OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Num processes        : ${NUM_PROCESSES}"

if [[ "${SKIP_CALIB}" != "1" ]]; then
    if [[ -f "${LAYER_IMPORTANCE_PATH}" && "${FORCE}" != "1" ]]; then
        echo "[skip] layer-importance exists: ${LAYER_IMPORTANCE_PATH}"
    else
        run_cmd accelerate launch \
            --num_processes "${NUM_PROCESSES}" \
            --main_process_port "${MAIN_PROCESS_PORT}" \
            get_layer_importance_ddp.py \
            --model_name_or_path "${MODEL_PATH}" \
            --save_dir "${LAYER_IMPORTANCE_DIR}" \
            --dataset "${DATASET}" \
            --loss_type "${MODES_LOSS_TYPE}" \
            --batch_size "${MODES_BATCH_SIZE}" \
            --num_samples "${MODES_NUM_SAMPLES}" \
            --start_idx "${MODES_START_IDX}" \
            --temperature "${MODES_TEMPERATURE}"
    fi
else
    echo "[skip] SKIP_CALIB=1"
fi

if [[ ! -f "${LAYER_IMPORTANCE_PATH}" ]]; then
    echo "error: layer importance file not found: ${LAYER_IMPORTANCE_PATH}" >&2
    exit 1
fi

if [[ "${SKIP_SEARCH}" != "1" ]]; then
    if [[ -f "${TAU_SKIP_PATH}" && "${FORCE}" != "1" ]]; then
        echo "[skip] tau-search exists: ${TAU_SKIP_PATH}"
    else
        search_cmd=(
            accelerate launch
            --num_processes "${NUM_PROCESSES}"
            --main_process_port "${MAIN_PROCESS_PORT}"
            grid_search_tau_ddp.py
            --model_name_or_path "${MODEL_PATH}"
            --save_dir "${TAU_SEARCH_DIR}"
            --dataset "${DATASET}"
            --loss_type "${MODES_LOSS_TYPE}"
            --batch_size "${SEARCH_BATCH_SIZE}"
            --num_samples "${MODES_NUM_SAMPLES}"
            --start_idx "${MODES_START_IDX}"
            --temperature "${MODES_TEMPERATURE}"
            --layer_importance_path "${LAYER_IMPORTANCE_PATH}"
            --target_skip_proportion "${TARGET_SKIP_PROPORTION}"
            --grid_num "${GRID_NUM}"
            --grid_map "${GRID_MAP}"
            --text_tau_min "${TEXT_TAU_MIN}"
            --text_tau_max "${TEXT_TAU_MAX}"
            --visual_tau_min "${VISUAL_TAU_MIN}"
            --visual_tau_max "${VISUAL_TAU_MAX}"
            --exp_coeff "${EXP_COEFF}"
            --attn_implementation "${SEARCH_ATTN_IMPLEMENTATION}"
        )
        if [[ -n "${EXPERT_IMPORTANCE_PATH}" ]]; then
            search_cmd+=(--expert_importance_path "${EXPERT_IMPORTANCE_PATH}")
        fi
        run_cmd "${search_cmd[@]}"
    fi
else
    echo "[skip] SKIP_SEARCH=1"
fi

if [[ ! -f "${TAU_SKIP_PATH}" ]]; then
    echo "error: tau search file not found: ${TAU_SKIP_PATH}" >&2
    exit 1
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
    for TASK in ${SWEEP_TASKS}; do
        TASK_OUTPUT_DIR="${JOINT_EVAL_OUTPUT_DIR}/${TASK}"
        mkdir -p "${TASK_OUTPUT_DIR}"
        echo "[eval] TASK=${TASK} OUTPUT_DIR=${TASK_OUTPUT_DIR}"
        eval_cmd=(
            bash scripts/run_prune_eval_kimi_gqa.sh
        )
        run_cmd env \
            PREFIX="${PREFIX}" \
            PYTHONPATH="${PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
            MODEL_PATH="${MODEL_PATH}" \
            SCORES_PATH="${SCORES_PATH}" \
            PRUNE_RATIO="${PRUNE_RATIO}" \
            TASK="${TASK}" \
            DATASET="${DATASET}" \
            NUM_SAMPLES="${EVAL_NUM_SAMPLES}" \
            BATCH_SIZE="${EVAL_BATCH_SIZE}" \
            START_IDX="${START_IDX}" \
            SUBSET_SEED="${SUBSET_SEED}" \
            USE_LMMS_EVAL="${USE_LMMS_EVAL}" \
            LAYERWISE_LOSS_KEY="${LAYERWISE_LOSS_KEY:-}" \
            TAU_SKIP_PATH="${TAU_SKIP_PATH}" \
            LAYER_IMPORTANCE_PATH="${LAYER_IMPORTANCE_PATH}" \
            EXPERT_IMPORTANCE_PATH="${EXPERT_IMPORTANCE_PATH}" \
            OUTPUT_DIR="${TASK_OUTPUT_DIR}" \
            "${eval_cmd[@]}"
    done
else
    echo "[skip] SKIP_EVAL=1"
fi

echo ""
echo "Done."
echo "LAYER_IMPORTANCE_PATH=${LAYER_IMPORTANCE_PATH}"
echo "TAU_SKIP_PATH=${TAU_SKIP_PATH}"
echo "JOINT_EVAL_OUTPUT_DIR=${JOINT_EVAL_OUTPUT_DIR}"
