#!/usr/bin/env bash
set -euo pipefail
source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH:-${PREFIX}/storage/data_distill_kimi/gqa-sample_at1.0-latest/teacher_hidden.pt}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d%H%M%S)}"
# storage/data_distill/teacher_cache-attn_weighted/teacher_hidden_cache.pt
OUTPUT_PATH="${OUTPUT_PATH:-$(dirname "${HIDDEN_PAYLOAD_PATH}")/distilled-${RUN_STAMP}/distilled_hidden.pt}"
LATEST_DISTILLED_LINK_DIR="${LATEST_DISTILLED_LINK_DIR:-$(dirname "${HIDDEN_PAYLOAD_PATH}")/distilled-latest}"

# 合成集规模：之前 256 对 2048d 分布的 MMD / cov 估计偏紧，放大到 512
SYNTHETIC_SIZE="${SYNTHETIC_SIZE:-1024}"  # 总规模
SYNTHETIC_BATCH_SIZE="${SYNTHETIC_BATCH_SIZE:-512}"  # 每次优化用的 batch 规模
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-1024}"

# 之前 2000 步明显没收敛（history 首尾 mmd/cov/mean/var 都还在上升）
TRAIN_STEPS="${TRAIN_STEPS:-12000}"
# 权重改大、步数增多后，lr 1e-2 容易震荡；调小到 5e-3
LR="${LR:-2e-3}"
INIT_STD="${INIT_STD:-0.0}"

# 新 loss 权重：
#   - mmd / cov / mean / var 现在是按 modality(text/image/video) 分组再平均，
#     所以 visual token 不会再"主导"损失，整体梯度会更均衡；
#   - cov 之前是全部指标里偏离最大的一项，权重从 0.1 → 0.5；
#   - div 在 loss 首尾实际 dominate 了优化器（其它项都在涨），权重 0.1 → 0.02，
#     并配合 warmup 避免前期扰动。
LAMBDA_MMD="${LAMBDA_MMD:-1.0}"
LAMBDA_COV="${LAMBDA_COV:-2.0}"
LAMBDA_DIV="${LAMBDA_DIV:-0.002}"
LAMBDA_MEAN="${LAMBDA_MEAN:-1.0}"
LAMBDA_VAR="${LAMBDA_VAR:-2.0}"
LAMBDA_BLOCK="${LAMBDA_BLOCK:-0.25}"

USE_EMA_NORMALIZED_LOSSES="${USE_EMA_NORMALIZED_LOSSES:-0}"
LOSS_EMA_DECAY="${LOSS_EMA_DECAY:-0.99}"

# 前 N 步把 lambda_div 从 0 线性提升到设定值，避免 div 在早期把合成 token 吹散
DIV_WARMUP_STEPS="${DIV_WARMUP_STEPS:-100}"
MMD_SUBSAMPLE="${MMD_SUBSAMPLE:-2048}"

LOG_INTERVAL="${LOG_INTERVAL:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"  # 等于 0 则无中间 ckpt

SEED="${SEED:-42}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

if [[ "${DEVICE_MAP}" == cuda:* ]]; then
    TRAIN_DEVICE="cuda"
else
    TRAIN_DEVICE="${DEVICE_MAP}"
fi

WANDB_PROJECT="${WANDB_PROJECT:-maes}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-distilled-${RUN_STAMP}}"


EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.distill_synthetic_hidden
    --teacher_cache_path "${HIDDEN_PAYLOAD_PATH}"
    --output_path "${OUTPUT_PATH}"
    --model_name_or_path "${MODEL_PATH}"
    --synthetic_size "${SYNTHETIC_SIZE}"
    --synthetic_batch_size "${SYNTHETIC_BATCH_SIZE}"
    --teacher_batch_size "${TEACHER_BATCH_SIZE}"
    --train_steps "${TRAIN_STEPS}"
    --lr "${LR}"
    --seed "${SEED}"
    --device "${TRAIN_DEVICE}"
    --init_std "${INIT_STD}"
    --lambda_mmd "${LAMBDA_MMD}"
    --lambda_cov "${LAMBDA_COV}"
    --lambda_div "${LAMBDA_DIV}"
    --lambda_mean "${LAMBDA_MEAN}"
    --lambda_var "${LAMBDA_VAR}"
    --lambda_block "${LAMBDA_BLOCK}"
    --div_warmup_steps "${DIV_WARMUP_STEPS}"
    --mmd_subsample "${MMD_SUBSAMPLE}"
    --log_interval "${LOG_INTERVAL}"
    --attn_implementation "${ATTN_IMPL}"
    --device_map "${DEVICE_MAP}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_run_name "${WANDB_RUN_NAME}"
    --wandb_mode "${WANDB_MODE}"
    --loss_ema_decay "${LOSS_EMA_DECAY}"
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"
)

if [[ "${USE_EMA_NORMALIZED_LOSSES}" == "1" ]]; then
    CMD+=(--use_ema_normalized_losses)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Teacher cache   : ${HIDDEN_PAYLOAD_PATH}"
echo "Output          : ${OUTPUT_PATH}"
echo "Synthetic M     : ${SYNTHETIC_SIZE}"
echo "Synthetic batch : ${SYNTHETIC_BATCH_SIZE}"
echo "Train steps     : ${TRAIN_STEPS}"
echo "LR              : ${LR}"
echo "Lambdas         : mmd=${LAMBDA_MMD}  cov=${LAMBDA_COV}  mean=${LAMBDA_MEAN}  var=${LAMBDA_VAR}  div=${LAMBDA_DIV}  block=${LAMBDA_BLOCK} (warmup=${DIV_WARMUP_STEPS})"
echo "Loss norm       : ema_normalized=${USE_EMA_NORMALIZED_LOSSES}  ema_decay=${LOSS_EMA_DECAY}"
echo "Checkpoint      : every ${CHECKPOINT_INTERVAL} step(s)"
echo "Wandb           : project=${WANDB_PROJECT}  run=${WANDB_RUN_NAME}  mode=${WANDB_MODE}"
echo "HF_HOME         : ${HF_HOME}"
echo "Device map      : ${DEVICE_MAP}"
echo "Train device    : ${TRAIN_DEVICE}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"

ln -sfn "$(dirname "${OUTPUT_PATH}")" "${LATEST_DISTILLED_LINK_DIR}"
echo "Latest distilled: ${LATEST_DISTILLED_LINK_DIR}"
