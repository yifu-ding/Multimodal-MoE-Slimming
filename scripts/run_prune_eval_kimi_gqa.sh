#!/usr/bin/env bash
set -euo pipefail

# In-memory prune + eval for Kimi-VL on GQA.
# No pruned checkpoint is saved; only eval outputs are written.
#
# Examples
# --------
# Default 50% pruning on 500 samples:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_prune_eval_kimi_gqa.sh
#
# Full eval:
#   NUM_SAMPLES=0 CUDA_VISIBLE_DEVICES=1 bash scripts/run_prune_eval_kimi_gqa.sh

source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}:${PREFIX}/lmms-eval"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SCORES_PATH="${SCORES_PATH:-}"
# -textvqa gqa chartqa mmstar mmbench mmvet mme realworldqa coco2017cap mvbench egoschema videomme longvideobench video_mmmu}"
TASK="${TASK:-textvqa}"

PRUNE_RATIO="${PRUNE_RATIO:-0.0}"
# INTER_METHOD options (inter-layer planner):
#   uniform
#   u_shaped
#   uniform_coverage
#   loss
#   loss_smooth_<N>      # e.g. loss_smooth_1, loss_smooth_2
#   loss_coverage
#   raw_loss_coverage
INTER_METHOD="${INTER_METHOD:-loss_smooth_2}"
# 在 loss_smooth 的时候会读取，可选：sqrt, cbrt, fourth_root, log, ...
SMOOTH_FN="${SMOOTH_FN:-sqrt}"
# INTRA_METHOD options (intra-layer planner, 基于 EXPERT_METRICS):
#   uniform_*
#   usage_*              # gate_scores.usage
#   router_*             # gate_scores.router
#   true_ablate_*        # expert_scores.true_ablate
#   first_attr_coverage
#   first_attr_fillzero
#   first_attr_fillzero_coverage
#   second_attr_coverage  # expert_scores.second_attr
#   second_attr_fillzero
#   second_attr_fillzero_coverage
INTRA_METHOD="${INTRA_METHOD:-second_attr_coverage}"
# INTRA_EXPERT_METRIC options (must exist in scores payload channel_scores):
# 下列 metric 都有 _text / _visual 后缀版本，开双模态时用 xxx_text + xxx_visual
#   gateup_act
#   3proj_act
#   down_second_order  # 这个是 approx 的
#   3proj_second_order
#   down_saliency
#   3proj_saliency
#   wa
#   3proj_grad
#   down_second_order_exact
#   wg
# 这个 metric 没有双模态版本
#   weight
MODALITY_AWARE="${MODALITY_AWARE:-1}"  # 是否开启双模态
INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC:-gateup_act}"

ALIGN_INTER="${ALIGN_INTER:-0}"
MIN_PER_EXPERT="${MIN_PER_EXPERT:-256}"

# THRESHOLDS_PATH="${THRESHOLDS_PATH:-${PREFIX}/storage/prune/thresholds/kimi_gqa/thresholds.pt}"
THRESHOLDS_PATH="${THRESHOLDS_PATH:-}" 

NUM_SAMPLES="${NUM_SAMPLES:-0}"  # 样本数, 0 表示全量
START_IDX="${START_IDX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SUBSET_SEED="${SUBSET_SEED:-}"

USE_LMMS_EVAL="${USE_LMMS_EVAL:-0}"

MODEL_NAME="${MODEL_NAME:-}"
if [[ -z "${MODEL_NAME}" ]]; then
    MODEL_TAG_RAW="${MODEL_PATH##*/}"
    MODEL_NAME="${MODEL_TAG_RAW,,}"
    case "${MODEL_NAME}" in
        qwen3-vl-30b-a3b-instruct)
            MODEL_NAME="qwen3-vl-30b-a3b"
            ;;
        kimi-vl-a3b-instruct)
            MODEL_NAME="kimi-vl-a3b"
            ;;
        internvl-3.5-gpt-oss-20b-a4b-preview-hf)
            MODEL_NAME="internvl-3.5-20b-a4b"
            ;;
        *)
            MODEL_NAME="${MODEL_NAME%-instruct}"
            ;;
    esac
fi

RATIO_TAG="p$(python3 -c "print(str(int(float('${PRUNE_RATIO}')*100)))")"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/results/prune_eval_${MODEL_NAME}_gqa_${RATIO_TAG}-rell2-$(date +%m%d%H%M)}/logs"

EXTRA_ARGS=("$@")

if [[ "${USE_LMMS_EVAL}" == "1" ]]; then
    # ── lmms-eval mode: use the eval wrapper with built-in task metrics ──
    # Auto-select eval script based on model family
    MODEL_LOWER="${MODEL_PATH,,}"
    if [[ "${MODEL_LOWER}" == *"qwen3"* ]]; then
        EVAL_SCRIPT="eval/qwen3.py"
        MODEL_KEY="qwen3_vl"
    else
        EVAL_SCRIPT="eval/kimi.py"
        MODEL_KEY="kimi_vl"
    fi

    MODEL_ARGS="pretrained=${MODEL_PATH}"
    if [[ -n "${SCORES_PATH}" ]]; then
        MODEL_ARGS+=",scores_path=${SCORES_PATH}"
        MODEL_ARGS+=",prune_ratio=${PRUNE_RATIO}"
        MODEL_ARGS+=",inter_method=${INTER_METHOD}"
        MODEL_ARGS+=",intra_method=${INTRA_METHOD}"
        MODEL_ARGS+=",intra_expert_metric=${INTRA_EXPERT_METRIC}"
        MODEL_ARGS+=",modality_aware=${MODALITY_AWARE}"
        MODEL_ARGS+=",smooth_fn=${SMOOTH_FN}"
        MODEL_ARGS+=",align_inter=${ALIGN_INTER}"
        MODEL_ARGS+=",min_per_expert=${MIN_PER_EXPERT}"
        if [[ -n "${THRESHOLDS_PATH}" ]]; then
            MODEL_ARGS+=",thresholds_path=${THRESHOLDS_PATH}"
        fi
    fi

    LIMIT_ARG=""
    if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
        LIMIT_ARG="--limit ${NUM_SAMPLES}"
    fi

    CMD=(
        python3 "${EVAL_SCRIPT}"
        --model_args "${MODEL_ARGS}"
        --tasks "${TASK}"
        --batch_size "${BATCH_SIZE}"
        --log_samples
        --output_path "${OUTPUT_DIR}"
    )
    if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
        CMD+=(--limit "${NUM_SAMPLES}")
    fi
    CMD+=("${EXTRA_ARGS[@]}")
else
    # ── Legacy mode: use prune_and_eval_kimi_gqa.py ──
    CMD=(
        python3 scripts/prune_and_eval_kimi_gqa.py
        --model_path "${MODEL_PATH}"
        --scores_path "${SCORES_PATH}"
        --task "${TASK}"
        --output_dir "${OUTPUT_DIR}"
        --prune_ratio "${PRUNE_RATIO}"
        --inter_method "${INTER_METHOD}"
        --intra_method "${INTRA_METHOD}"
        --intra_expert_metric "${INTRA_EXPERT_METRIC}"
        --smooth_fn "${SMOOTH_FN}"
        --align_inter "${ALIGN_INTER}"
        --min_per_expert "${MIN_PER_EXPERT}"
        --num_samples "${NUM_SAMPLES}"
        --start_idx "${START_IDX}"
        --batch_size "${BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
    )

    if [[ "${MODALITY_AWARE}" == "1" ]]; then
        CMD+=(--modality_aware)
    fi

    if [[ -n "${SUBSET_SEED}" ]]; then
        CMD+=(--subset_seed "${SUBSET_SEED}")
    fi

    if [[ -n "${THRESHOLDS_PATH}" ]]; then
        CMD+=(--thresholds_path "${THRESHOLDS_PATH}")
    fi

    CMD+=("${EXTRA_ARGS[@]}")
fi

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/stdout_${TIMESTAMP}.log"
{
echo "Model       : ${MODEL_PATH}"
echo "Task        : ${TASK}"
echo "Scores      : ${SCORES_PATH}"
echo "Thresholds  : $([[ -n "${THRESHOLDS_PATH}" ]] && echo "${THRESHOLDS_PATH}" || echo "disabled")"
echo "Prune ratio : ${PRUNE_RATIO}"
echo "Inter       : ${INTER_METHOD}"
echo "Intra       : ${INTRA_METHOD}"
echo "Metric      : ${INTRA_EXPERT_METRIC}"
echo "Modality    : $([[ "${MODALITY_AWARE}" == "1" ]] && echo "text+visual" || echo "disabled")"
echo "Smooth fn   : ${SMOOTH_FN}"
echo "Eval output : ${OUTPUT_DIR}"
echo "Samples     : ${NUM_SAMPLES} (0=full)"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "Use lmms    : ${USE_LMMS_EVAL}"
echo "Log file    : ${LOG_FILE}"
echo ""
"${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"
