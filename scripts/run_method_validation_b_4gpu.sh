#!/usr/bin/env bash
set -euo pipefail

# Collect single-expert HVP/energy/ablation data by sharding MoE layers over GPUs.

PREFIX="${PREFIX:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/dyf/miniconda/envs/vllm-maes/bin/python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
SCORES="${SCORES:-/home/dyf/data/MARS-results/storage/scores/qwen3-mixed-512/scores.pt}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/artifacts/method_validation_b/qwen3-mixed-validation-${VALIDATION_SAMPLES}}"
GPUS="${GPUS:-0,1,2,3}"
SWEEP_LAYER="${SWEEP_LAYER:-0}"
LIVE_LOGS="${LIVE_LOGS:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-0}"
DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-1024}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-4096}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"

export PYTHONPATH="${PREFIX}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/maes-validation-b-mpl}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable does not exist: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${SCORES}" ]]; then
    echo "scores.pt does not exist: ${SCORES}" >&2
    exit 1
fi
if [[ ! "${VALIDATION_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "VALIDATION_SAMPLES must be a positive integer, got: ${VALIDATION_SAMPLES}" >&2
    exit 1
fi
if [[ "${LIVE_LOGS}" != "0" && "${LIVE_LOGS}" != "1" ]]; then
    echo "LIVE_LOGS must be 0 or 1." >&2
    exit 1
fi
for binary_flag in WAIT_FOR_GPUS RUN_SMOKE_TEST RESUME; do
    value="${!binary_flag}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${binary_flag} must be 0 or 1." >&2
        exit 1
    fi
done

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if (( ${#GPU_IDS[@]} != 4 )); then
    echo "GPUS must contain exactly four comma-separated GPU IDs, got: ${GPUS}" >&2
    exit 1
fi
for gpu in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID: ${gpu}" >&2
        exit 1
    fi
done

mapfile -t MOE_LAYERS < <("${PYTHON_BIN}" scripts/discover_moe_layers_from_config.py --model "${MODEL_PATH}")
if (( ${#MOE_LAYERS[@]} == 0 )); then
    echo "No MoE layers discovered for ${MODEL_PATH}." >&2
    exit 1
fi
if [[ " ${MOE_LAYERS[*]} " != *" ${SWEEP_LAYER} "* ]]; then
    echo "SWEEP_LAYER=${SWEEP_LAYER} is not in discovered MoE layers: ${MOE_LAYERS[*]}" >&2
    exit 1
fi

declare -a SHARD_LAYERS PIDS SHARDS
for ((rank = 0; rank < 4; rank++)); do
    SHARD_LAYERS[rank]=""
done
for ((position = 0; position < ${#MOE_LAYERS[@]}; position++)); do
    rank=$((position % 4))
    SHARD_LAYERS[rank]="${SHARD_LAYERS[rank]} ${MOE_LAYERS[position]}"
done

mkdir -p "${OUTPUT_DIR}/logs"

terminate_workers() {
    for pid in "${PIDS[@]-}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    done
}
trap terminate_workers INT TERM HUP EXIT

echo "Model       : ${MODEL_PATH}"
echo "Scores      : ${SCORES}"
echo "GPUs        : ${GPUS}"
echo "MoE layers  : ${MOE_LAYERS[*]}"
echo "Sweep layer : ${SWEEP_LAYER}"
echo "Samples     : ${VALIDATION_SAMPLES} (frozen-manifest prefix)"
echo "Output      : ${OUTPUT_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
    while [[ "${WAIT_FOR_GPUS}" == "1" ]]; do
        all_ready=1
        usage_rows="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
        for gpu in "${GPU_IDS[@]}"; do
            used="$(awk -F, -v target="${gpu}" '$1 + 0 == target {gsub(/ /, "", $2); print $2}' <<<"${usage_rows}")"
            if [[ -z "${used}" ]]; then
                echo "Could not read memory usage for GPU ${gpu}." >&2
                exit 1
            fi
            if (( used > MAX_USED_MIB )); then
                all_ready=0
            fi
        done
        if (( all_ready == 1 )); then
            echo "All requested GPUs are below ${MAX_USED_MIB} MiB used."
            break
        fi
        echo "Waiting for GPUs (required <= ${MAX_USED_MIB} MiB): $(tr '\n' ';' <<<"${usage_rows}")"
        sleep "${GPU_WAIT_SECONDS}"
    done

    if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
        echo "Running one-sample Layer ${SWEEP_LAYER} smoke test on GPU ${GPU_IDS[0]}..."
        env \
            PYTHONUNBUFFERED=1 \
            DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX}" \
            CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
            "${PYTHON_BIN}" -m src.calibration.collect_method_validation_b \
            --scores "${SCORES}" \
            --model-name-or-path "${MODEL_PATH}" \
            --layers "${SWEEP_LAYER}" \
            --sweep-layer "${SWEEP_LAYER}" \
            --max-samples 1 \
            --output "${OUTPUT_DIR}/smoke.pt" \
            --force \
            >"${OUTPUT_DIR}/logs/smoke.log" 2>&1
        "${PYTHON_BIN}" scripts/check_method_validation_b.py \
            --input "${OUTPUT_DIR}/smoke.pt" \
            >>"${OUTPUT_DIR}/logs/smoke.log" 2>&1
        echo "Smoke test passed: ${OUTPUT_DIR}/smoke.pt"
    fi
fi

for ((rank = 0; rank < 4; rank++)); do
    gpu="${GPU_IDS[rank]}"
    layers="${SHARD_LAYERS[rank]# }"
    shard="${OUTPUT_DIR}/shard${rank}.pt"
    log="${OUTPUT_DIR}/logs/shard${rank}.log"
    SHARDS[rank]="${shard}"
    sweep_args=()
    if [[ " ${layers} " == *" ${SWEEP_LAYER} "* ]]; then
        sweep_args=(--sweep-layer "${SWEEP_LAYER}")
    fi
    output_args=()
    if [[ "${RESUME}" == "1" ]]; then
        output_args=(--resume)
    elif [[ "${FORCE}" == "1" ]]; then
        output_args=(--force)
    fi
    echo "Shard ${rank}: GPU ${gpu}, layers=[${layers}], log=${log}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi
    command=(
        setsid env
        PYTHONUNBUFFERED=1
        DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX}"
        CUDA_VISIBLE_DEVICES="${gpu}"
        "${PYTHON_BIN}" -m src.calibration.collect_method_validation_b
        --scores "${SCORES}"
        --model-name-or-path "${MODEL_PATH}"
        --layers ${layers}
        --max-samples "${VALIDATION_SAMPLES}"
        --output "${shard}"
        "${sweep_args[@]}"
        "${output_args[@]}"
    )
    if [[ "${LIVE_LOGS}" == "1" ]]; then
        tee_args=()
        [[ "${RESUME}" == "1" ]] && tee_args=(-a)
        "${command[@]}" > >(tee "${tee_args[@]}" "${log}" | tr '\r' '\n' | awk -v prefix="[B${rank}] " 'NF {print prefix $0; fflush()}') 2>&1 &
    else
        if [[ "${RESUME}" == "1" ]]; then
            "${command[@]}" >>"${log}" 2>&1 &
        else
            "${command[@]}" >"${log}" 2>&1 &
        fi
    fi
    PIDS[rank]=$!
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: workers were not started."
    exit 0
fi

failed=0
for ((rank = 0; rank < 4; rank++)); do
    if wait "${PIDS[rank]}"; then
        echo "Shard ${rank} completed: ${SHARDS[rank]}"
    else
        status=$?
        echo "Shard ${rank} failed with status ${status}: ${OUTPUT_DIR}/logs/shard${rank}.log" >&2
        failed=1
    fi
done
trap - INT TERM HUP EXIT
if (( failed != 0 )); then
    exit 1
fi

merge_args=()
[[ "${FORCE}" == "1" ]] && merge_args=(--force)
"${PYTHON_BIN}" scripts/merge_method_validation_b.py \
    "${SHARDS[@]}" \
    --output "${OUTPUT_DIR}/method_validation_b.pt" \
    "${merge_args[@]}"
"${PYTHON_BIN}" scripts/check_method_validation_b.py \
    --input "${OUTPUT_DIR}/method_validation_b.pt"
"${PYTHON_BIN}" scripts/export_method_validation_b_csv.py \
    --input "${OUTPUT_DIR}/method_validation_b.pt" \
    --output-dir "${OUTPUT_DIR}/data"
"${PYTHON_BIN}" draw/hessian-3d-landscape/plot_method_validation_b.py \
    --data-dir "${OUTPUT_DIR}/data" \
    --output-dir "${OUTPUT_DIR}"
echo "Method-validation B collection complete: ${OUTPUT_DIR}/method_validation_b.pt"
