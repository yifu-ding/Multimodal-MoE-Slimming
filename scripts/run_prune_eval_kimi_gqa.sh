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
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SCORES_PATH="${SCORES_PATH:-${PREFIX}/storage/prune/scores/kimi_gqa-second_order/scores.pt}"

PRUNE_RATIO="${PRUNE_RATIO:-0.50}"
# INTER_METHOD options (inter-layer planner):
#   uniform
#   u_shaped
#   uniform_coverage
#   loss
#   loss_smooth_<N>      # e.g. loss_smooth_1, loss_smooth_2
#   loss_coverage
#   raw_loss_coverage
INTER_METHOD="${INTER_METHOD:-loss_smooth_1}"
# INTRA_METHOD options (intra-layer planner):
#   uniform
#   channel_ranking
#   loss
#   global
#   usage
#   router
#   attr_coverage
#   second_attr_coverage
#   true_ablate
#   true_ablate_coverage
#   loss_coverage  # 没有计算逐个expert loss的话就不支持这个参数
#   usage_coverage
#   router_coverage
INTRA_METHOD="${INTRA_METHOD:-second_attr_coverage}"
# INTRA_EXPERT_METRIC options (must exist in scores payload expert_scores):
#   activation, wa, grad, gateup_act, activation_text, activation_visual, saliency, token_contrib, wg, weight
INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC:-activation}"
ALIGN_INTER="${ALIGN_INTER:-0}"
MIN_PER_EXPERT="${MIN_PER_EXPERT:-128}"

MODALITY_AWARE="${MODALITY_AWARE:-1}"  # 是否开启双模态

NUM_SAMPLES="${NUM_SAMPLES:-500}"  # 样本数
START_IDX="${START_IDX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SUBSET_SEED="${SUBSET_SEED:-}"

RATIO_TAG="p$(python3 -c "print(str(int(float('${PRUNE_RATIO}')*100)))")"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/results/prune_eval_kimi_gqa_${RATIO_TAG}}"

EXTRA_ARGS=("$@")

CMD=(
    python scripts/prune_and_eval_kimi_gqa.py
    --model_path "${MODEL_PATH}"
    --scores_path "${SCORES_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --prune_ratio "${PRUNE_RATIO}"
    --inter_method "${INTER_METHOD}"
    --intra_method "${INTRA_METHOD}"
    --intra_expert_metric "${INTRA_EXPERT_METRIC}"
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

CMD+=("${EXTRA_ARGS[@]}")

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/stdout_${TIMESTAMP}.log"
{
echo "Model       : ${MODEL_PATH}"
echo "Scores      : ${SCORES_PATH}"
echo "Prune ratio : ${PRUNE_RATIO}"
echo "Inter       : ${INTER_METHOD}"
echo "Intra       : ${INTRA_METHOD}"
echo "Metric      : ${INTRA_EXPERT_METRIC}"
echo "Modality    : $([[ "${MODALITY_AWARE}" == "1" ]] && echo "text+visual" || echo "disabled")"
echo "Eval output : ${OUTPUT_DIR}"
echo "Samples     : ${NUM_SAMPLES} (0=full)"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "Log file    : ${LOG_FILE}"
echo ""
"${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"
