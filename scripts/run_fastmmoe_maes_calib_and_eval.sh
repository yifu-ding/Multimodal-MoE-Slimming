#!/usr/bin/env bash
set -euo pipefail

# Run MAES calibration + eval with FastMMoE-enabled DeepSeek-VL2 / InternVL3.5.
#
# FastMMoE is controlled by vendor runtime env vars and is wired into MAES model
# loaders when FASTMMOE_ENABLE=1.
#
# Example:
#   MODEL_NAME=deepseek-vl2-small \
#   TASK=gqa \
#   PRUNE_RATIO=0.5 \
#   bash scripts/run_fastmmoe_maes_calib_and_eval.sh

source scripts/select_least_used_gpu.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}:${PREFIX}/lmms-eval"
export FASTMMOE_ENABLE=1

MODEL_NAME="${MODEL_NAME:-deepseek-vl2-small}"
TASK="${TASK:-gqa}"
DATASET="${DATASET:-${TASK}}"
PRUNE_RATIO="${PRUNE_RATIO:-0.5}"

COLLECT_NUM_SAMPLES="${COLLECT_NUM_SAMPLES:-1024}"
COLLECT_TOKEN_PER_SAMPLE="${COLLECT_TOKEN_PER_SAMPLE:-2048}"
COLLECT_BATCH_SIZE="${COLLECT_BATCH_SIZE:-8}"
COLLECT_START_IDX="${COLLECT_START_IDX:-0}"
COLLECT_SUBSET_SEED="${COLLECT_SUBSET_SEED:-42}"
COLLECT_LOSS_FN="${COLLECT_LOSS_FN:-rel_l2}"

EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_START_IDX="${EVAL_START_IDX:-0}"
EVAL_SUBSET_SEED="${EVAL_SUBSET_SEED:-}"
USE_LMMS_EVAL="${USE_LMMS_EVAL:-0}"

RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PREFIX}/storage/fastmmoe_maes/${MODEL_NAME}/${DATASET}/${RUN_TAG}}"
CALIB_OUTPUT_DIR="${CALIB_OUTPUT_DIR:-${OUTPUT_ROOT}/scores}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/eval}"
SCORES_PATH_DEFAULT="${CALIB_OUTPUT_DIR}/scores.pt"
SCORES_PATH="${SCORES_PATH:-${SCORES_PATH_DEFAULT}}"

SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FORCE="${FORCE:-0}"

REDUCTION_LAYER_IDX="${REDUCTION_LAYER_IDX:-2}"
VISION_EXPERT_REDUCE_FACTOR="${VISION_EXPERT_REDUCE_FACTOR:-0.5}"
TOKEN_MERGE_STRATEGY="${TOKEN_MERGE_STRATEGY:-hybrid}"
TOKEN_MERGE_METHOD="${TOKEN_MERGE_METHOD:-mlerp}"
ROUTING_SIMILARITY_WINDOW_SIZE="${ROUTING_SIMILARITY_WINDOW_SIZE:-3}"

case "${MODEL_NAME}" in
    deepseek-vl2-small)
        export MODEL_PATH="${MODEL_PATH:-deepseek-ai/deepseek-vl2-small}"
        BASE_ALPHA="${BASE_ALPHA:-0.3}"
        MERGE_RATIO="${MERGE_RATIO:-0.05}"
        MERGE_LAYER_LOCS="${MERGE_LAYER_LOCS:-2,5,8}"
        KEEP_TOKEN_RATIO="${KEEP_TOKEN_RATIO:-0.91,0.91,0.91}"
        ;;
    internvl3_5-30b-a3b-hf)
        export MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-30B-A3B-HF}"
        BASE_ALPHA="${BASE_ALPHA:-0.5}"
        MERGE_RATIO="${MERGE_RATIO:-0.025}"
        MERGE_LAYER_LOCS="${MERGE_LAYER_LOCS:-5,8,12}"
        KEEP_TOKEN_RATIO="${KEEP_TOKEN_RATIO:-0.91,0.91,0.91}"
        ;;
    *)
        echo "error: MODEL_NAME must be one of: deepseek-vl2-small, internvl3_5-30b-a3b-hf" >&2
        exit 1
        ;;
esac

export REDUCTION_LAYER_IDX
export VISION_EXPERT_REDUCE_FACTOR
export TOKEN_MERGE_STRATEGY
export TOKEN_MERGE_METHOD
export ROUTING_SIMILARITY_WINDOW_SIZE
export BASE_ALPHA
export MERGE_RATIO
export MERGE_LAYER_LOCS
export KEEP_TOKEN_RATIO

mkdir -p "${OUTPUT_ROOT}" "${CALIB_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

run_cmd() {
    echo ""
    echo "CMD: $*"
    echo ""
    "$@"
}

echo "Model                    : ${MODEL_NAME}"
echo "Model path               : ${MODEL_PATH}"
echo "Dataset / task           : ${DATASET} / ${TASK}"
echo "FastMMoE strategy        : ${TOKEN_MERGE_STRATEGY}"
echo "MAES prune ratio         : ${PRUNE_RATIO}"
echo "Scores output            : ${SCORES_PATH}"
echo "Eval output              : ${EVAL_OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES     : ${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ "${SKIP_COLLECT}" != "1" ]]; then
    if [[ -f "${SCORES_PATH}" && "${FORCE}" != "1" ]]; then
        echo "[skip] scores exist: ${SCORES_PATH}"
    else
        collect_cmd=(
            bash scripts/run_collect_scores.sh
        )
        collect_extra=()
        if [[ "${FORCE}" == "1" ]]; then
            collect_extra+=(--force)
        fi
        run_cmd env \
            PREFIX="${PREFIX}" \
            PYTHONPATH="${PYTHONPATH}" \
            FASTMMOE_ENABLE="${FASTMMOE_ENABLE}" \
            MODEL_PATH="${MODEL_PATH}" \
            DATASET="${DATASET}" \
            NUM_SAMPLES="${COLLECT_NUM_SAMPLES}" \
            TOKEN_PER_SAMPLE="${COLLECT_TOKEN_PER_SAMPLE}" \
            BATCH_SIZE="${COLLECT_BATCH_SIZE}" \
            START_IDX="${COLLECT_START_IDX}" \
            SUBSET_SEED="${COLLECT_SUBSET_SEED}" \
            LOSS_FN="${COLLECT_LOSS_FN}" \
            OUTPUT_DIR="${CALIB_OUTPUT_DIR}" \
            REDUCTION_LAYER_IDX="${REDUCTION_LAYER_IDX}" \
            VISION_EXPERT_REDUCE_FACTOR="${VISION_EXPERT_REDUCE_FACTOR}" \
            TOKEN_MERGE_STRATEGY="${TOKEN_MERGE_STRATEGY}" \
            TOKEN_MERGE_METHOD="${TOKEN_MERGE_METHOD}" \
            ROUTING_SIMILARITY_WINDOW_SIZE="${ROUTING_SIMILARITY_WINDOW_SIZE}" \
            BASE_ALPHA="${BASE_ALPHA}" \
            MERGE_RATIO="${MERGE_RATIO}" \
            MERGE_LAYER_LOCS="${MERGE_LAYER_LOCS}" \
            KEEP_TOKEN_RATIO="${KEEP_TOKEN_RATIO}" \
            "${collect_cmd[@]}" \
            "${collect_extra[@]}"
    fi
else
    echo "[skip] SKIP_COLLECT=1"
fi

if [[ ! -f "${SCORES_PATH}" ]]; then
    echo "error: scores file not found: ${SCORES_PATH}" >&2
    exit 1
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
    run_cmd env \
        PREFIX="${PREFIX}" \
        PYTHONPATH="${PYTHONPATH}" \
        FASTMMOE_ENABLE="${FASTMMOE_ENABLE}" \
        MODEL_NAME="${MODEL_NAME}" \
        MODEL_PATH="${MODEL_PATH}" \
        SCORES_PATH="${SCORES_PATH}" \
        TASK="${TASK}" \
        PRUNE_RATIO="${PRUNE_RATIO}" \
        NUM_SAMPLES="${EVAL_NUM_SAMPLES}" \
        BATCH_SIZE="${EVAL_BATCH_SIZE}" \
        START_IDX="${EVAL_START_IDX}" \
        SUBSET_SEED="${EVAL_SUBSET_SEED}" \
        USE_LMMS_EVAL="${USE_LMMS_EVAL}" \
        OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
        REDUCTION_LAYER_IDX="${REDUCTION_LAYER_IDX}" \
        VISION_EXPERT_REDUCE_FACTOR="${VISION_EXPERT_REDUCE_FACTOR}" \
        TOKEN_MERGE_STRATEGY="${TOKEN_MERGE_STRATEGY}" \
        TOKEN_MERGE_METHOD="${TOKEN_MERGE_METHOD}" \
        ROUTING_SIMILARITY_WINDOW_SIZE="${ROUTING_SIMILARITY_WINDOW_SIZE}" \
        BASE_ALPHA="${BASE_ALPHA}" \
        MERGE_RATIO="${MERGE_RATIO}" \
        MERGE_LAYER_LOCS="${MERGE_LAYER_LOCS}" \
        KEEP_TOKEN_RATIO="${KEEP_TOKEN_RATIO}" \
        bash scripts/run_prune_eval_kimi_gqa.sh
else
    echo "[skip] SKIP_EVAL=1"
fi

echo ""
echo "Done."
echo "SCORES_PATH=${SCORES_PATH}"
echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
