#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EP4_PLAN="${EP4_PLAN:?Set EP4_PLAN to a validated Qwen EP4 plan.}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for this pruning ratio.}"
START_BATCH_SIZE="${START_BATCH_SIZE:-8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-512}"
BATCH_GRANULARITY="${BATCH_GRANULARITY:-8}"
MEASURED_BATCHES="${MEASURED_BATCHES:-4}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TOKENS_PER_REQUEST_BUDGET="${TOKENS_PER_REQUEST_BUDGET:-512}"
PREFILL_MAX_NEW_TOKENS="${PREFILL_MAX_NEW_TOKENS:-1}"
PREFILL_TASK="${PREFILL_TASK:-gqa_prefill}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-64}"
STRATEGIES="${STRATEGIES:-padded,multi_kernel,cross_layer}"

for value_name in START_BATCH_SIZE MAX_BATCH_SIZE BATCH_GRANULARITY \
    MEASURED_BATCHES WARMUP_BATCHES TOKENS_PER_REQUEST_BUDGET \
    PREFILL_MAX_NEW_TOKENS; do
    if [[ ! "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ${value_name} must be a positive integer" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}"
STATUS_FILE="${OUTPUT_ROOT}/sweep_status.tsv"
if [[ ! -e "${STATUS_FILE}" ]]; then
    printf 'strategy\tbatch_size\tstatus\texit_code\n' > "${STATUS_FILE}"
fi

wait_for_clean_four_gpus() {
    while true; do
        mapfile -t rows < <(
            nvidia-smi --query-gpu=index,memory.used \
                --format=csv,noheader,nounits 2>/dev/null || true
        )
        compute_processes="$({
            nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,name \
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
        [[ -z "${compute_processes}" ]] || clean=0
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

status_for_point() {
    local strategy="$1"
    local batch_size="$2"
    awk -F '\t' -v strategy="${strategy}" -v batch="${batch_size}" \
        '$1 == strategy && $2 == batch { status=$3 } END { print status }' "${STATUS_FILE}"
}

record_status() {
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "${STATUS_FILE}"
}

run_point() {
    local strategy="$1"
    local batch_size="$2"
    local existing
    existing="$(status_for_point "${strategy}" "${batch_size}")"
    if [[ "${existing}" == "complete" ]]; then
        echo "[skip] strategy=${strategy} batch_size=${batch_size} already complete"
        return 0
    fi

    wait_for_clean_four_gpus
    local point_dir="${OUTPUT_ROOT}/${strategy}/bs_${batch_size}"
    local gpu_trace="${point_dir}/gpu_trace.csv"
    local batch_trace="${point_dir}/batch_trace.jsonl"
    local wrapper_log="${point_dir}/runner.log"
    local limit=$(( batch_size * (WARMUP_BATCHES + MEASURED_BATCHES) ))
    local token_budget=$(( batch_size * TOKENS_PER_REQUEST_BUDGET ))
    (( token_budget >= 16384 )) || token_budget=16384
    mkdir -p "${point_dir}"
    printf 'timestamp,gpu_index,memory_used_mib,memory_total_mib,gpu_utilization_percent,power_watts\n' > "${gpu_trace}"
    : > "${batch_trace}"
    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits --loop-ms=200 >> "${gpu_trace}" &
    local monitor_pid=$!
    cleanup_monitor() {
        kill -TERM "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    }
    trap cleanup_monitor EXIT INT TERM

    echo "[run] strategy=${strategy} batch_size=${batch_size} limit=${limit} max_num_batched_tokens=${token_budget}"
    set +e
    EP4_PLAN="${EP4_PLAN}" \
    MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct \
    MAES_EP4_STRATEGY="${strategy}" \
    MAES_EFFICIENCY_TRACE="${batch_trace}" \
    MAES_EFFICIENCY_WARMUP_BATCHES="${WARMUP_BATCHES}" \
    TASKS="${PREFILL_TASK}" \
    LIMIT="${limit}" \
    LIGHT_IMAGE_BATCH_SIZE="${batch_size}" \
    MAX_NUM_SEQS="${batch_size}" \
    MAX_NUM_BATCHED_TOKENS="${token_budget}" \
    MAX_NEW_TOKENS="${PREFILL_MAX_NEW_TOKENS}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    KV_CACHE_MEMORY_BYTES= \
    MAX_MODEL_LEN=4096 \
    ENABLE_RESPONSE_CACHE=0 \
    FORCE=1 \
    FAIL_FAST=1 \
    RUN_DIR="${point_dir}/eval" \
    BASELINE_LABEL="qwen3_gqa_${strategy}_bs${batch_size}" \
        bash scripts/run_vllm_ep4_pruned.sh 2>&1 | tee "${wrapper_log}"
    local run_status=${PIPESTATUS[0]}
    set -e
    cleanup_monitor
    trap - EXIT INT TERM

    if (( run_status == 0 )); then
        record_status "${strategy}" "${batch_size}" complete 0
        return 0
    fi
    if rg -qi 'CUDA out of memory|OutOfMemoryError|failed with.*out of memory' "${wrapper_log}"; then
        record_status "${strategy}" "${batch_size}" oom "${run_status}"
        return 42
    fi
    record_status "${strategy}" "${batch_size}" failed "${run_status}"
    echo "error: non-OOM failure for strategy=${strategy} batch_size=${batch_size}" >&2
    return "${run_status}"
}

search_strategy() {
    local strategy="$1"
    local batch_size="${START_BATCH_SIZE}"
    local low=0
    local high=0
    while (( batch_size <= MAX_BATCH_SIZE )); do
        if run_point "${strategy}" "${batch_size}"; then
            low="${batch_size}"
            batch_size=$(( batch_size * 2 ))
        else
            local point_status=$?
            if (( point_status != 42 )); then
                return "${point_status}"
            fi
            high="${batch_size}"
            break
        fi
    done
    if (( high == 0 )); then
        record_status "${strategy}" "${MAX_BATCH_SIZE}" capped 0
        return 0
    fi
    while (( high - low > BATCH_GRANULARITY )); do
        local middle=$(( ((low + high) / 2 / BATCH_GRANULARITY) * BATCH_GRANULARITY ))
        (( middle > low )) || middle=$(( low + BATCH_GRANULARITY ))
        if run_point "${strategy}" "${middle}"; then
            low="${middle}"
        else
            local point_status=$?
            if (( point_status != 42 )); then
                return "${point_status}"
            fi
            high="${middle}"
        fi
    done
    echo "[boundary] strategy=${strategy} max_stable_batch=${low} first_oom_batch=${high}"
}

IFS=',' read -r -a strategy_list <<< "${STRATEGIES}"
for strategy in "${strategy_list[@]}"; do
    strategy="${strategy// /}"
    case "${strategy}" in
        padded|multi_kernel|cross_layer) ;;
        *) echo "error: unsupported strategy '${strategy}'" >&2; exit 2 ;;
    esac
    search_strategy "${strategy}"
done

conda run --no-capture-output -n vllm-maes \
    python scripts/collect_qwen_ep4_batch_sweep.py --input-root "${OUTPUT_ROOT}"
echo "batch_sweep_results=${OUTPUT_ROOT}"
