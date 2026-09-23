#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_KEY="${MODEL_KEY:?Set MODEL_KEY to qwen235 or mistral119.}"
STRATEGY="${STRATEGY:?Set STRATEGY.}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREFILL_TOKENS="${PREFILL_TOKENS:-512}"
DECODE_PROMPT_TOKENS="${DECODE_PROMPT_TOKENS:-32}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$(( BATCH_SIZE * PREFILL_TOKENS ))}"
(( MAX_NUM_BATCHED_TOKENS >= 16384 )) || MAX_NUM_BATCHED_TOKENS=16384

case "${MODEL_KEY}" in
    qwen235)
        MODEL_PATH="${MODEL_PATH:-/home/data3/dyf/models/Qwen3-VL-235B-A22B-Instruct-FP8}"
        EP4_PLAN="${EP4_PLAN:-${REPO_ROOT}/artifacts/efficiency_batch64_main/plans/qwen3-vl-235b-p30-seed2603.pt}"
        ;;
    mistral119)
        MODEL_PATH="${MODEL_PATH:-/home/data3/dyf/models/Mistral-Small-4-119B-2603}"
        EP4_PLAN="${EP4_PLAN:-${REPO_ROOT}/artifacts/efficiency_batch64_main/plans/mistral4-119b-p30-seed2603.pt}"
        ;;
    *) echo "error: unsupported MODEL_KEY=${MODEL_KEY}" >&2; exit 2 ;;
esac
case "${STRATEGY}" in
    padded|multi_kernel|single_width|cross_layer) ;;
    *) echo "error: unsupported STRATEGY=${STRATEGY}" >&2; exit 2 ;;
esac
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "error: missing model ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${EP4_PLAN}" ]] || { echo "error: missing plan ${EP4_PLAN}" >&2; exit 2; }

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/artifacts/efficiency_batch64_main/runs/${MODEL_KEY}/${STRATEGY}/bs_${BATCH_SIZE}}"
mkdir -p "${RUN_DIR}"
GPU_TRACE="${RUN_DIR}/gpu_trace.csv"
RESULT_JSON="${RUN_DIR}/result.json"
printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_watts\n' > "${GPU_TRACE}"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits --loop-ms=200 >> "${GPU_TRACE}" &
monitor_pid=$!
cleanup() {
    kill -TERM "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export MAES_EP4_PLAN="$(realpath "${EP4_PLAN}")"
export MAES_EP4_STRATEGY="${STRATEGY}"
export PYTHONPATH="${REPO_ROOT}/runtime/vllm_ep4:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
conda run --no-capture-output -n vllm-maes \
    python scripts/benchmark_ep4_efficiency.py \
    --model-path "${MODEL_PATH}" \
    --output "${RESULT_JSON}" \
    --batch-size "${BATCH_SIZE}" \
    --prefill-tokens "${PREFILL_TOKENS}" \
    --decode-prompt-tokens "${DECODE_PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --measured-runs "${MEASURED_RUNS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" 2>&1 | tee "${RUN_DIR}/runner.log"
cleanup
trap - EXIT INT TERM
printf 'complete\n' > "${RUN_DIR}/COMPLETE"
