#!/usr/bin/env bash
# Returns 0 only when all four GPUs are idle AND no sibling eval process is
# alive. A single nvidia-smi sample is not enough: between two tasks of the
# *same* long-running eval, the old vLLM engine tears down and the new one
# has not allocated memory yet, producing a several-second window where every
# GPU reads ~0 MiB used even though the run is very much still in progress.
# Racing that window (as this script originally did) let a second dispatcher
# instance start a colliding run on the same RUN_DIR. Guard against it two
# ways: require several consecutive idle memory samples, and treat any live
# run_*_vllm_baseline.sh / run_vllm_ep4_pruned.sh process as busy regardless
# of instantaneous memory readings.
set -uo pipefail

IDLE_MEMORY_MIB=64
CONSECUTIVE_SAMPLES=3
SAMPLE_INTERVAL_SECONDS=5

gpus_idle_once() {
    mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
    if (( ${#rows[@]} != 4 )); then
        return 1
    fi
    local row used
    for row in "${rows[@]}"; do
        used="${row#*,}"
        used="${used// /}"
        if (( used > IDLE_MEMORY_MIB )); then
            return 1
        fi
    done
    local compute_processes
    compute_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
    [[ -z "${compute_processes}" ]]
}

sibling_eval_alive() {
    pgrep -f "run_qwen3_vl_vllm_baseline\.sh|run_kimi_vl_vllm_baseline\.sh|run_internvl35_vllm_baseline\.sh|run_vllm_ep4_pruned\.sh|run_vllm_ep4_default\.sh" >/dev/null 2>&1
}

if sibling_eval_alive; then
    exit 1
fi

for _ in $(seq 1 "${CONSECUTIVE_SAMPLES}"); do
    gpus_idle_once || exit 1
    sleep "${SAMPLE_INTERVAL_SECONDS}"
done

sibling_eval_alive && exit 1

exit 0
