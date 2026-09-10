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
TASK="${TASK:-gqa}"  # gqa

PRUNE_RATIO="${PRUNE_RATIO:-0.5}"
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
INTRA_METHOD="${INTRA_METHOD:-second_attr_fillzero_coverage}"
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
SHARED_PROTECT="${SHARED_PROTECT:-1}"  # 双模态时是否保留 shared channels
TEXT_ONLY="${TEXT_ONLY:-0}"  # ablation: 仅使用 text tentative mask
VISUAL_ONLY="${VISUAL_ONLY:-0}"  # ablation: 仅使用 visual tentative mask
NORMALIZE="${NORMALIZE:-0}"  # 是否对 text / visual 分模态做层级归一化
EXPERTWISE_BUDGET_NORMALIZE="${EXPERTWISE_BUDGET_NORMALIZE:-1}"  # 是否先按层归一化 expert raw budget 再分配
USE_EMA="${USE_EMA:-1}"  # 是否使用 EMA affinity 分配 modality budget
# scores payload 里用作 EMA 源的 tensor 键名, 与 ``pipeline.prepare_scores(..., ema_source_key=...)`` 一致
EMA_SOURCE_KEY="${EMA_SOURCE_KEY:-ema_matrix}"
INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC:-gateup_act}"
LAYERWISE_LOSS_KEY="${LAYERWISE_LOSS_KEY:-layerwise_loss}"

ALIGN_INTER="${ALIGN_INTER:-0}"
MIN_PER_EXPERT="${MIN_PER_EXPERT:-128}"

# THRESHOLDS_PATH="${THRESHOLDS_PATH:-${PREFIX}/storage/prune/thresholds/kimi_gqa/thresholds.pt}"
THRESHOLDS_PATH="${THRESHOLDS_PATH:-}" 

NUM_SAMPLES="${NUM_SAMPLES:-0}"  # 样本数, 0 表示全量
START_IDX="${START_IDX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ -z "${MAX_NEW_TOKENS:-}" ]]; then
    if [[ "${TASK}" == "video_mmmu" ]]; then
        MAX_NEW_TOKENS="1024"
    else
        MAX_NEW_TOKENS="32"
    fi
fi
SUBSET_SEED="${SUBSET_SEED:-}"

USE_LMMS_EVAL="${USE_LMMS_EVAL:-0}"
TAU_SKIP_PATH="${TAU_SKIP_PATH:-}"
LAYER_IMPORTANCE_PATH="${LAYER_IMPORTANCE_PATH:-}"
EXPERT_IMPORTANCE_PATH="${EXPERT_IMPORTANCE_PATH:-}"

MODEL_NAME="${MODEL_NAME:-}"
if [[ -n "${MODEL_NAME}" ]]; then
    case "${MODEL_NAME}" in
        deepseek-vl2-small)
            MODEL_PATH="${MODEL_PATH:-deepseek-ai/deepseek-vl2-small}"
            MODEL_FAMILY="deepseek_vl"
            ;;
        kimi-vl-a3b|kimi-vl-a3b-instruct)
            MODEL_NAME="kimi-vl-a3b"
            MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
            MODEL_FAMILY="kimi"
            ;;
        qwen3-vl-30b-a3b|qwen3-vl-30b-a3b-instruct)
            MODEL_NAME="qwen3-vl-30b-a3b"
            MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
            MODEL_FAMILY="qwen3_vl"
            ;;
        internvl3_5-30b-a3b-hf)
            MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-30B-A3B-HF}"
            MODEL_FAMILY="internvl"
            ;;
        gemma-4-26b-a4b)
            MODEL_PATH="${MODEL_PATH:-google/gemma-4-26B-A4B}"
            MODEL_FAMILY="gemma4"
            ;;
        qwen3.5-35b-a3b)
            MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
            MODEL_FAMILY="qwen3_5"
            ;;
        *)
            echo "error: MODEL_NAME must be one of: deepseek-vl2-small kimi-vl-a3b kimi-vl-a3b-instruct qwen3-vl-30b-a3b qwen3-vl-30b-a3b-instruct internvl3_5-30b-a3b-hf gemma-4-26b-a4b qwen3.5-35b-a3b. Got: ${MODEL_NAME}" >&2
            exit 1
            ;;
    esac
fi

if [[ -z "${MODEL_NAME}" ]]; then
    MODEL_TAG_RAW="${MODEL_PATH##*/}"
    MODEL_NAME="${MODEL_TAG_RAW,,}"
fi

if [[ -z "${MODEL_FAMILY:-}" ]]; then
    MODEL_LOWER="${MODEL_PATH,,}"
    case "${MODEL_LOWER}" in
        *deepseek-vl2*)
            MODEL_FAMILY="deepseek_vl"
            ;;
        *kimi-vl*)
            MODEL_FAMILY="kimi"
            ;;
        *qwen3-vl*)
            MODEL_FAMILY="qwen3_vl"
            ;;
        *internvl*)
            MODEL_FAMILY="internvl"
            ;;
        *gemma-4*)
            MODEL_FAMILY="gemma4"
            ;;
        *qwen3.5*)
            MODEL_FAMILY="qwen3_5"
            ;;
        *)
            echo "error: cannot infer model family from MODEL_PATH=${MODEL_PATH}. Please set MODEL_NAME explicitly." >&2
            exit 1
            ;;
    esac
fi

RATIO_TAG="p$(python3 -c "print(str(int(float('${PRUNE_RATIO}')*100)))")"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/results/prune_eval_${MODEL_NAME}_gqa_${RATIO_TAG}-$(date +%m%d%H%M)}/logs"

EXTRA_ARGS=("$@")

if [[ "${USE_LMMS_EVAL}" == "1" ]]; then
    # ── lmms-eval mode: use the eval wrapper with built-in task metrics ──
    # Auto-select eval script based on model family.
    # Today only Kimi-VL and Qwen3-VL have local lmms-eval wrappers with MoDES pruning support.
    case "${MODEL_FAMILY}" in
        qwen3_vl)
            EVAL_SCRIPT="eval/qwen3.py"
            MODEL_KEY="qwen3_vl"
            ;;
        kimi)
            EVAL_SCRIPT="eval/kimi.py"
            MODEL_KEY="kimi_vl"
            ;;
        deepseek_vl|internvl|gemma4|qwen3_5)
            echo "[warn] USE_LMMS_EVAL=1 is not implemented for MODEL_FAMILY=${MODEL_FAMILY}; falling back to legacy prune_and_eval path." >&2
            USE_LMMS_EVAL="0"
            ;;
        *)
            echo "error: unsupported MODEL_FAMILY=${MODEL_FAMILY}" >&2
            exit 1
            ;;
    esac
fi

if [[ "${USE_LMMS_EVAL}" == "1" ]]; then
    MODEL_ARGS="pretrained=${MODEL_PATH}"
    if [[ -n "${SCORES_PATH}" ]]; then
        MODEL_ARGS+=",scores_path=${SCORES_PATH}"
        MODEL_ARGS+=",prune_ratio=${PRUNE_RATIO}"
        MODEL_ARGS+=",inter_method=${INTER_METHOD}"
        MODEL_ARGS+=",intra_method=${INTRA_METHOD}"
        MODEL_ARGS+=",intra_expert_metric=${INTRA_EXPERT_METRIC}"
        MODEL_ARGS+=",modality_aware=${MODALITY_AWARE}"
        MODEL_ARGS+=",shared_protect=${SHARED_PROTECT}"
        MODEL_ARGS+=",text_only=${TEXT_ONLY}"
        MODEL_ARGS+=",visual_only=${VISUAL_ONLY}"
        MODEL_ARGS+=",normalize=${NORMALIZE}"
        MODEL_ARGS+=",expertwise_budget_normalize=${EXPERTWISE_BUDGET_NORMALIZE}"
        MODEL_ARGS+=",use_ema=${USE_EMA}"
        MODEL_ARGS+=",ema_source_key=${EMA_SOURCE_KEY}"
        MODEL_ARGS+=",layerwise_loss_key=${LAYERWISE_LOSS_KEY}"
        MODEL_ARGS+=",smooth_fn=${SMOOTH_FN}"
        MODEL_ARGS+=",align_inter=${ALIGN_INTER}"
        MODEL_ARGS+=",min_per_expert=${MIN_PER_EXPERT}"
        if [[ -n "${THRESHOLDS_PATH}" ]]; then
            MODEL_ARGS+=",thresholds_path=${THRESHOLDS_PATH}"
        fi
    fi
    if [[ -n "${TAU_SKIP_PATH}" ]]; then
        MODEL_ARGS+=",tau_skip_path=${TAU_SKIP_PATH}"
    fi
    if [[ -n "${LAYER_IMPORTANCE_PATH}" ]]; then
        MODEL_ARGS+=",layer_importance_path=${LAYER_IMPORTANCE_PATH}"
    fi
    if [[ -n "${EXPERT_IMPORTANCE_PATH}" ]]; then
        MODEL_ARGS+=",expert_importance_path=${EXPERT_IMPORTANCE_PATH}"
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
        --layerwise_loss_key "${LAYERWISE_LOSS_KEY}"
        --smooth_fn "${SMOOTH_FN}"
        --use_ema "${USE_EMA}"
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

    if [[ "${SHARED_PROTECT}" == "1" ]]; then
        CMD+=(--shared_protect)
    fi

    if [[ "${TEXT_ONLY}" == "1" ]]; then
        CMD+=(--text_only)
    fi

    if [[ "${VISUAL_ONLY}" == "1" ]]; then
        CMD+=(--visual_only)
    fi

    if [[ "${NORMALIZE}" == "1" ]]; then
        CMD+=(--normalize)
    fi

    if [[ "${EXPERTWISE_BUDGET_NORMALIZE}" == "1" ]]; then
        CMD+=(--expertwise_budget_normalize)
    fi

    CMD+=(--ema_source_key "${EMA_SOURCE_KEY}")

    if [[ -n "${SUBSET_SEED}" ]]; then
        CMD+=(--subset_seed "${SUBSET_SEED}")
    fi

    if [[ -n "${TAU_SKIP_PATH}" ]]; then
        CMD+=(--tau_skip_path "${TAU_SKIP_PATH}")
    fi

    if [[ -n "${LAYER_IMPORTANCE_PATH}" ]]; then
        CMD+=(--layer_importance_path "${LAYER_IMPORTANCE_PATH}")
    fi

    if [[ -n "${EXPERT_IMPORTANCE_PATH}" ]]; then
        CMD+=(--expert_importance_path "${EXPERT_IMPORTANCE_PATH}")
    fi

    if [[ -n "${THRESHOLDS_PATH}" ]]; then
        CMD+=(--thresholds_path "${THRESHOLDS_PATH}")
    fi

    CMD+=("${EXTRA_ARGS[@]}")
fi

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/stdout_${TIMESTAMP}.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
{
echo "Model       : ${MODEL_PATH}"
echo "Task        : ${TASK}"
echo "Scores      : ${SCORES_PATH}"
echo "Thresholds  : $([[ -n "${THRESHOLDS_PATH}" ]] && echo "${THRESHOLDS_PATH}" || echo "disabled")"
echo "Prune ratio : ${PRUNE_RATIO}"
echo "Inter       : ${INTER_METHOD}"
echo "Intra       : ${INTRA_METHOD}"
echo "Metric      : ${INTRA_EXPERT_METRIC}"
echo "Modality    : ${MODALITY_AWARE}"
echo "Shared prot : ${SHARED_PROTECT}"
echo "Text only   : ${TEXT_ONLY}"
echo "Visual only : ${VISUAL_ONLY}"
echo "Normalize   : ${NORMALIZE}"
echo "Expert norm : ${EXPERTWISE_BUDGET_NORMALIZE}"
echo "Use EMA     : ${USE_EMA}"
echo "EMA source  : ${EMA_SOURCE_KEY}"
echo "Layer loss  : ${LAYERWISE_LOSS_KEY}"
echo "Smooth fn   : ${SMOOTH_FN}"
echo "Eval output : ${OUTPUT_DIR}"
echo "Samples     : ${NUM_SAMPLES} (0=full)"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "Use lmms    : ${USE_LMMS_EVAL}"
echo "Log file    : ${LOG_FILE}"
echo ""
"${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"
