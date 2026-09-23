#!/usr/bin/env bash
# EP4 部署效率主表：p=0 default + p=0.3/0.5 x {padded,multi_kernel,single_width,cross_layer}
# 对 Qwen3-VL-30B / InternVL3.5 / Kimi-VL 三个低风险模型，batch=512，prefill-only。
# 幂等：复用 run_qwen_ep4_batch_sweep.sh 自带的 sweep_status.tsv 断点续跑机制，
# 可以安全地重复执行本脚本，已完成的点会被跳过。
set -uo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
cd "${REPO_ROOT}"

STATUS_LOG="${REPO_ROOT}/.automation/efficiency_batch512_main/campaign.log"

run_default() {
    local model="$1" dir="$2"
    local out="${REPO_ROOT}/artifacts/efficiency_figure/${dir}/batch512_main/prune_0"
    local run_exit
    mkdir -p "${out}"
    echo "[campaign] $(date -Is) START default model=${model}" | tee -a "${STATUS_LOG}"
    MODEL="${model}" \
    OUTPUT_ROOT="${out}" \
    START_BATCH_SIZE=512 MAX_BATCH_SIZE=512 \
    MEASURED_BATCHES=4 WARMUP_BATCHES=1 \
    STRATEGIES=default \
    PREFILL_TASK=gqa_prefill PREFILL_MAX_NEW_TOKENS=1 \
    GPU_MEMORY_UTILIZATION=0.90 \
    TOKENS_PER_REQUEST_BUDGET=256 \
        bash scripts/run_qwen_ep4_batch_sweep.sh >> "${STATUS_LOG}" 2>&1
    run_exit=$?
    echo "[campaign] $(date -Is) END default model=${model} exit=${run_exit}" | tee -a "${STATUS_LOG}"
}

run_ratio() {
    local model="$1" dir="$2" ratio="$3"
    local plan="${REPO_ROOT}/artifacts/efficiency_figure/${dir}/random_balanced_ep4_p${ratio}.pt"
    local out="${REPO_ROOT}/artifacts/efficiency_figure/${dir}/batch512_main/prune_${ratio}"
    local run_exit
    mkdir -p "${out}"
    for strategy in padded multi_kernel single_width cross_layer; do
        echo "[campaign] $(date -Is) START p${ratio} model=${model} strategy=${strategy}" | tee -a "${STATUS_LOG}"
        EP4_PLAN="${plan}" \
        MODEL="${model}" \
        OUTPUT_ROOT="${out}" \
        START_BATCH_SIZE=512 MAX_BATCH_SIZE=512 \
        MEASURED_BATCHES=4 WARMUP_BATCHES=1 \
        STRATEGIES="${strategy}" \
        PREFILL_TASK=gqa_prefill PREFILL_MAX_NEW_TOKENS=1 \
        GPU_MEMORY_UTILIZATION=0.90 \
        TOKENS_PER_REQUEST_BUDGET=256 \
            bash scripts/run_qwen_ep4_batch_sweep.sh >> "${STATUS_LOG}" 2>&1
        run_exit=$?
        echo "[campaign] $(date -Is) END p${ratio} model=${model} strategy=${strategy} exit=${run_exit}" | tee -a "${STATUS_LOG}"
    done
}

declare -A MODEL_DIR=(
    ["Qwen/Qwen3-VL-30B-A3B-Instruct"]="qwen3_gqa_ep4"
    ["OpenGVLab/InternVL3_5-30B-A3B-HF"]="internvl3_5_gqa_ep4"
    ["moonshotai/Kimi-VL-A3B-Instruct"]="kimi_vl_gqa_ep4"
)

for model in "Qwen/Qwen3-VL-30B-A3B-Instruct" "OpenGVLab/InternVL3_5-30B-A3B-HF" "moonshotai/Kimi-VL-A3B-Instruct"; do
    dir="${MODEL_DIR[${model}]}"
    run_default "${model}" "${dir}"
    run_ratio "${model}" "${dir}" 30
    run_ratio "${model}" "${dir}" 50
done

echo "[campaign] $(date -Is) ALL DONE" | tee -a "${STATUS_LOG}"
