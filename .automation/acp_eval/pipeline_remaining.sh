#!/usr/bin/env bash
# Waits for the Kimi ACP run's PID to exit (GPUs freed), then runs
# Qwen3-VL-30B and InternVL3.5's ACP p=0.3 evaluations back to back.
set -uo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
cd "${REPO_ROOT}"

KIMI_PID="${KIMI_PID:?Set KIMI_PID to the running Kimi ACP eval PID.}"
LOG_DIR="${REPO_ROOT}/.automation/acp_eval"

echo "[acp-pipeline] $(date -Is) waiting for Kimi PID=${KIMI_PID} to exit"
while kill -0 "${KIMI_PID}" 2>/dev/null; do
    sleep 10
done
echo "[acp-pipeline] $(date -Is) Kimi run exited; GPUs should be free"

TASKS="gqa,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mme,realworldqa"

run_model() {
    local model="$1" plan="$2" out_dir="$3"
    echo "[acp-pipeline] $(date -Is) START ${model}"
    EP4_PLAN="${plan}" \
    MODEL="${model}" \
    MAES_EP4_STRATEGY=padded \
    PRUNING_LABEL=acp \
    OUTPUT_ROOT="${out_dir}" \
    TASKS="${TASKS}" \
    RANDOM_SUBSET_FRACTION=0.5 \
    RANDOM_SUBSET_MIN_SAMPLES=1 \
    RANDOM_SUBSET_SEED=42 \
    GPU_MEMORY_UTILIZATION=0.90 \
        bash scripts/run_vllm_ep4_pruned.sh >> "${LOG_DIR}/$(basename "${out_dir}").log" 2>&1
    echo "[acp-pipeline] $(date -Is) END ${model} exit=$?"
}

run_model \
    "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    "$(realpath runtime/ep4_plans/acp/qwen3-vl-30b-a3b-acp-p30.pt)" \
    "$(realpath .)/results/vllm_acp/qwen3-vl-30b-a3b/ep4-p30-padded"

run_model \
    "OpenGVLab/InternVL3_5-30B-A3B-HF" \
    "$(realpath runtime/ep4_plans/acp/internvl3_5-30b-a3b-acp-p30.pt)" \
    "$(realpath .)/results/vllm_acp/internvl3_5-30b-a3b/ep4-p30-padded"

echo "[acp-pipeline] $(date -Is) ALL DONE"
