#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EP4_PLAN="${EP4_PLAN:?Set EP4_PLAN to a validated Qwen EP4 plan.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/efficiency_figure/qwen3_gqa_ep4}"
LIMIT="${LIMIT:-72}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-64}"
STRATEGIES="${STRATEGIES:-padded,multi_kernel,cross_layer}"

mkdir -p "${OUTPUT_ROOT}"

wait_for_clean_four_gpus() {
    while true; do
        mapfile -t rows < <(
            nvidia-smi \
                --query-gpu=index,memory.used \
                --format=csv,noheader,nounits 2>/dev/null || true
        )
        compute_processes="$({
            nvidia-smi \
                --query-compute-apps=pid,gpu_uuid,used_memory,name \
                --format=csv,noheader,nounits 2>/dev/null || true
        })"
        clean=1
        if (( ${#rows[@]} != 4 )); then
            clean=0
        else
            for expected in 0 1 2 3; do
                IFS=',' read -r index used <<< "${rows[expected]}"
                index="${index// /}"
                used="${used// /}"
                if [[ "${index}" != "${expected}" ]] || (( used > GPU_IDLE_MEMORY_MIB )); then
                    clean=0
                fi
            done
        fi
        if [[ -n "${compute_processes}" ]]; then
            clean=0
        fi
        if (( clean == 1 )); then
            printf '[gpu-clean] %s all four GPUs are idle\n' "$(date --iso-8601=seconds)"
            return 0
        fi
        printf '[gpu-wait] %s four GPUs are not clean; retrying in %ss\n' \
            "$(date --iso-8601=seconds)" "${GPU_WAIT_SECONDS}"
        printf '%s\n' "${rows[@]}"
        [[ -z "${compute_processes}" ]] || printf '%s\n' "${compute_processes}"
        sleep "${GPU_WAIT_SECONDS}"
    done
}

IFS=',' read -r -a strategy_list <<< "${STRATEGIES}"
for strategy in "${strategy_list[@]}"; do
    strategy="${strategy// /}"
    case "${strategy}" in
        padded|multi_kernel|cross_layer) ;;
        *) echo "error: unsupported strategy '${strategy}'" >&2; exit 2 ;;
    esac

    wait_for_clean_four_gpus
    run_dir="${OUTPUT_ROOT}/${strategy}"
    mkdir -p "${run_dir}"
    gpu_trace="${run_dir}/gpu_trace.csv"
    batch_trace="${run_dir}/batch_trace.jsonl"
    printf 'timestamp,gpu_index,memory_used_mib,memory_total_mib,gpu_utilization_percent,power_watts\n' > "${gpu_trace}"
    : > "${batch_trace}"
    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits \
        --loop-ms=200 >> "${gpu_trace}" &
    monitor_pid=$!
    cleanup_monitor() {
        kill -TERM "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    }
    trap cleanup_monitor EXIT INT TERM

    set +e
    EP4_PLAN="${EP4_PLAN}" \
    MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct \
    MAES_EP4_STRATEGY="${strategy}" \
    MAES_EFFICIENCY_TRACE="${batch_trace}" \
    MAES_EFFICIENCY_WARMUP_BATCHES="${WARMUP_BATCHES}" \
    TASKS=gqa \
    LIMIT="${LIMIT}" \
    LIGHT_IMAGE_BATCH_SIZE="${BATCH_SIZE}" \
    MAX_NEW_TOKENS=16 \
    GPU_MEMORY_UTILIZATION=0.90 \
    KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES}" \
    MAX_MODEL_LEN=4096 \
    ENABLE_RESPONSE_CACHE=0 \
    FORCE=1 \
    FAIL_FAST=1 \
    RUN_DIR="${run_dir}/eval" \
    BASELINE_LABEL="qwen3_gqa_${strategy}" \
        bash scripts/run_vllm_ep4_pruned.sh
    status=$?
    set -e
    cleanup_monitor
    trap - EXIT INT TERM
    if (( status != 0 )); then
        echo "error: strategy ${strategy} failed with exit code ${status}" >&2
        exit "${status}"
    fi
done

conda run --no-capture-output -n vllm-maes \
    python scripts/collect_qwen_ep4_efficiency.py --input-root "${OUTPUT_ROOT}"
echo "efficiency_results=${OUTPUT_ROOT}"
