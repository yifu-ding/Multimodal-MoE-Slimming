#!/usr/bin/env bash
set -euo pipefail
source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export HF_HOME="${HF_HOME:-/home/data/dyf/hf_cache}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH:-${PREFIX}/storage/data_distill_kimi/gqa-sample_at1.0-latest/teacher_hidden.pt}"
# storage/data_distill/teacher_cache-attn_weighted/teacher_hidden_cache.pt
OUTPUT_PATH="${OUTPUT_PATH:-$(dirname "${HIDDEN_PAYLOAD_PATH}")/distilled-$(date +%m%d%H%M%S)/distilled_hidden.pt}"
LATEST_DISTILLED_LINK_DIR="${LATEST_DISTILLED_LINK_DIR:-$(dirname "${HIDDEN_PAYLOAD_PATH}")/distilled-latest}"

# 合成集规模：之前 256 对 2048d 分布的 MMD / cov 估计偏紧，放大到 512
SYNTHETIC_SIZE="${SYNTHETIC_SIZE:-512}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-1024}"

# 之前 2000 步明显没收敛（history 首尾 mmd/cov/mean/var 都还在上升），翻三倍
TRAIN_STEPS="${TRAIN_STEPS:-8000}"
# 权重改大、步数增多后，lr 1e-2 容易震荡；调小到 5e-3
LR="${LR:-5e-3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
INIT_STD="${INIT_STD:-0.0}"

# 新 loss 权重：
#   - mmd / cov / mean / var 现在是按 modality(text/image/video) 分组再平均，
#     所以 visual token 不会再"主导"损失，整体梯度会更均衡；
#   - cov 之前是全部指标里偏离最大的一项，权重从 0.1 → 0.5；
#   - div 在 loss 首尾实际 dominate 了优化器（其它项都在涨），权重 0.1 → 0.02，
#     并配合 warmup 避免前期扰动。
LAMBDA_MMD="${LAMBDA_MMD:-2.0}"
LAMBDA_COV="${LAMBDA_COV:-0.5}"
LAMBDA_DIV="${LAMBDA_DIV:-0.02}"
LAMBDA_MEAN="${LAMBDA_MEAN:-1.0}"
LAMBDA_VAR="${LAMBDA_VAR:-1.0}"
LAMBDA_BLOCK="${LAMBDA_BLOCK:-1.0}"

# 前 N 步把 lambda_div 从 0 线性提升到设定值，避免 div 在早期把合成 token 吹散
DIV_WARMUP_STEPS="${DIV_WARMUP_STEPS:-300}"

MMD_SUBSAMPLE="${MMD_SUBSAMPLE:-2048}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.representation_distill.distill_synthetic_hidden
    --teacher_cache_path "${HIDDEN_PAYLOAD_PATH}"
    --output_path "${OUTPUT_PATH}"
    --model_name_or_path "${MODEL_PATH}"
    --synthetic_size "${SYNTHETIC_SIZE}"
    --teacher_batch_size "${TEACHER_BATCH_SIZE}"
    --train_steps "${TRAIN_STEPS}"
    --lr "${LR}"
    --seed "${SEED}"
    --device "${DEVICE}"
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
)

CMD+=("${EXTRA_ARGS[@]}")

echo "Teacher cache   : ${HIDDEN_PAYLOAD_PATH}"
echo "Output          : ${OUTPUT_PATH}"
echo "Synthetic M     : ${SYNTHETIC_SIZE}"
echo "Train steps     : ${TRAIN_STEPS}"
echo "LR              : ${LR}"
echo "Lambdas         : mmd=${LAMBDA_MMD}  cov=${LAMBDA_COV}  mean=${LAMBDA_MEAN}  var=${LAMBDA_VAR}  div=${LAMBDA_DIV}  block=${LAMBDA_BLOCK} (warmup=${DIV_WARMUP_STEPS})"
echo "HF_HOME         : ${HF_HOME}"
echo "Device          : ${DEVICE}"
echo "CMD: ${CMD[*]}"
echo ""
"${CMD[@]}"

ln -sfn "$(dirname "${OUTPUT_PATH}")" "${LATEST_DISTILLED_LINK_DIR}"
echo "Latest distilled: ${LATEST_DISTILLED_LINK_DIR}"
