#!/usr/bin/env bash
set -euo pipefail

# Calibrate per-expert pruning thresholds for Kimi-VL via blockwise loss.
#
# Prerequisites:
#   - A scores.pt file with activation_text and activation_visual
#     (produced by ``python -m src.calibration.main --modality_aware``).
#
# Examples
# --------
# Default (128 samples, 3 epochs, ratios 0.1–0.7):
#   bash scripts/run_threshold_calibration.sh
#
# Custom scores & ratios:
#   SCORES_PATH=storage/prune/scores/foo/scores.pt \
#   PRUNING_RATIOS="0.3,0.5,0.7" \
#   bash scripts/run_threshold_calibration.sh

source scripts/select_least_used_gpu.sh

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
SCORES_PATH="${SCORES_PATH:-${PREFIX}/storage/prune/scores/debug/scores.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/thresholds/kimi_gqa}"

NUM_SAMPLES="${NUM_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATASET="${DATASET:-gqa}"

PRUNING_RATIOS="${PRUNING_RATIOS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-0.01}"
# PENALTY_LAMBDA 是剪枝率约束项的惩罚系数 λ，控制 loss 中两项的相对权重：
# loss = loss_recon + λ · (keep_ratio - target_keep)²
# loss_recon — blockwise 重建误差（rel_l2），希望越小越好
# λ · (keep_ratio - target_keep)² — 约束项，惩罚实际保留率偏离目标剪枝率
# 如果跑完后发现实际 keep_ratio 偏离目标比较大（ thresholds.pt 里的 actual_keep_ratio 字段），可以增大这个参数。
PENALTY_LAMBDA="${PENALTY_LAMBDA:-100.0}"
# temperature annealing 的参数：
# TAU_START=0.1 — 第 0 个 epoch 使用的初始温度。值越大，sigmoid 曲线越平滑，掩码接近 0.5（软），梯度更稳定但掩码不精确。
# TAU_END=0.01 — 最后一个 epoch 使用的结束温度。值越小，sigmoid 曲线越陡，掩码接近 0/1（硬），更接近真实剪枝效果。
TAU_START="${TAU_START:-0.1}"
TAU_END="${TAU_END:-0.01}"
LOSS_FN="${LOSS_FN:-rel_l2}"
# INTRA_EXPERT_METRIC 只有两个选项： activation 和 channel_second_order
INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC:-channel_second_order}"

# Set to 1 to disable temperature annealing
NO_ANNEAL="${NO_ANNEAL:-0}"

CMD=(
    python -m src.calibration.threshold_calibration
    --model_name_or_path "${MODEL_PATH}"
    --scores_path "${SCORES_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --dataset "${DATASET}"
    --num_samples "${NUM_SAMPLES}"
    --batch_size "${BATCH_SIZE}"
    --pruning_ratios "${PRUNING_RATIOS}"
    --num_epochs "${NUM_EPOCHS}"
    --lr "${LR}"
    --penalty_lambda "${PENALTY_LAMBDA}"
    --tau_start "${TAU_START}"
    --tau_end "${TAU_END}"
    --loss_fn "${LOSS_FN}"
    --intra_expert_metric "${INTRA_EXPERT_METRIC}"
    --force
)

if [[ "${NO_ANNEAL}" == "1" ]]; then
    CMD+=("--no_anneal")
fi

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/stdout_${TIMESTAMP}.log"
{
echo "Model          : ${MODEL_PATH}"
echo "Scores         : ${SCORES_PATH}"
echo "Output         : ${OUTPUT_DIR}"
echo "Pruning ratios : ${PRUNING_RATIOS}"
echo "Epochs         : ${NUM_EPOCHS}"
echo "LR             : ${LR}"
echo "Lambda         : ${PENALTY_LAMBDA}"
echo "Tau            : ${TAU_START} → ${TAU_END} (cosine annealing)"
echo "Loss fn        : ${LOSS_FN}"
echo "Metric         : ${INTRA_EXPERT_METRIC}"
echo "Samples        : ${NUM_SAMPLES}"
echo "GPU            : ${CUDA_VISIBLE_DEVICES}"
echo "Log file       : ${LOG_FILE}"
echo ""
"${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"
