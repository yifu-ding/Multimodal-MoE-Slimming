#!/usr/bin/env bash
# Resumable dispatcher: runs each model's ACP p=0.3 padded-strategy eval
# (7 benchmarks, 50% random subset each) in a fixed RUN_DIR so already
# completed tasks (status/<task>.complete) are skipped automatically by
# run_qwen3_vl_vllm_baseline.sh. Safe to re-run; skips fully-done models.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${REPO_ROOT}"

wait_for_gpus() {
    while ! bash "${SCRIPT_DIR}/resource_check.sh"; do
        echo "[pipeline] $(date -Is) GPUs busy; waiting"
        sleep 30
    done
}

for name in "${MODEL_NAMES[@]}"; do
    IFS='|' read -r model_id plan run_dir <<< "$(model_spec "${name}")"
    done_count="$(count_complete "${run_dir}")"
    if (( done_count == ${#TASK_LIST[@]} )); then
        echo "[pipeline] $(date -Is) skip ${name}: all ${#TASK_LIST[@]} tasks already complete"
        continue
    fi

    echo "[pipeline] $(date -Is) ${name}: ${done_count}/${#TASK_LIST[@]} done; resuming"
    wait_for_gpus

    mkdir -p "${run_dir}"
    EP4_PLAN="${plan}" \
    MODEL="${model_id}" \
    MAES_EP4_STRATEGY=padded \
    PRUNING_LABEL=acp \
    OUTPUT_ROOT="$(dirname "${run_dir}")" \
    RUN_DIR="${run_dir}" \
    TASKS="${TASKS}" \
    RANDOM_SUBSET_FRACTION=0.5 \
    RANDOM_SUBSET_MIN_SAMPLES=1 \
    RANDOM_SUBSET_SEED=42 \
    GPU_MEMORY_UTILIZATION=0.90 \
    ENABLE_QWEN3_NATIVE_VIDEO=0 \
        bash scripts/run_vllm_ep4_pruned.sh
    status=$?
    echo "[pipeline] $(date -Is) ${name} run exited with status=${status}"
done

echo "[pipeline] $(date -Is) dispatcher pass complete"
