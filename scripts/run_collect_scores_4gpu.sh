#!/usr/bin/env bash
set -euo pipefail

# Run independent layer shards on four GPUs, then strictly merge scores.pt.
# Each worker loads one complete model on one GPU; this is intended for models
# that fit on a single GPU and whose layer calibrations dominate runtime.

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/maes-matplotlib-cache}"
export HF_HOME="${MAES_HF_HOME:-/home/data/dyf/hf_cache}"
export HF_HUB_CACHE="${MAES_HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${MAES_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
GPUS="${GPUS:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/scores/qwen3-mixed-512-4gpu}"
HESSIAN_PROBE_LAYER="${HESSIAN_PROBE_LAYER:-0}"
DRY_RUN="${DRY_RUN:-0}"
LIVE_LOGS="${LIVE_LOGS:-1}"
DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-1024}"

if [[ "${LIVE_LOGS}" != "0" && "${LIVE_LOGS}" != "1" ]]; then
    echo "LIVE_LOGS must be 0 or 1, got ${LIVE_LOGS}." >&2
    exit 1
fi

if [[ -z "${SELECTION_MANIFEST}" || ! -f "${SELECTION_MANIFEST}" ]]; then
    echo "SELECTION_MANIFEST must point to an existing frozen manifest." >&2
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "GPUS must contain at least two comma-separated GPU IDs." >&2
    exit 1
fi
for gpu in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID in GPUS=${GPUS}: ${gpu}" >&2
        exit 1
    fi
done

mapfile -t MOE_LAYERS < <(
    python scripts/discover_moe_layers_from_config.py --model "${MODEL_PATH}"
)
if (( ${#MOE_LAYERS[@]} == 0 )); then
    echo "No MoE layers discovered for ${MODEL_PATH}." >&2
    exit 1
fi

declare -a SHARD_LAYERS
for ((rank = 0; rank < ${#GPU_IDS[@]}; rank++)); do
    SHARD_LAYERS[rank]=""
done
for ((position = 0; position < ${#MOE_LAYERS[@]}; position++)); do
    rank=$((position % ${#GPU_IDS[@]}))
    SHARD_LAYERS[rank]="${SHARD_LAYERS[rank]} ${MOE_LAYERS[position]}"
done

mkdir -p "${OUTPUT_DIR}/logs"
declare -a PIDS
declare -a SHARD_SCORE_PATHS

terminate_workers() {
    for pid in "${PIDS[@]-}"; do
        # Each worker is a session leader, so terminate its Python descendants too.
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    done
}
trap terminate_workers INT TERM

echo "Model      : ${MODEL_PATH}"
echo "Manifest   : ${SELECTION_MANIFEST}"
echo "GPUs       : ${GPUS}"
echo "Output     : ${OUTPUT_DIR}"
echo "MoE layers : ${MOE_LAYERS[*]}"
echo "Live logs  : ${LIVE_LOGS}"
echo "Decord EOF retries: ${DECORD_EOF_RETRY_MAX}"

for ((rank = 0; rank < ${#GPU_IDS[@]}; rank++)); do
    gpu="${GPU_IDS[rank]}"
    layers="${SHARD_LAYERS[rank]# }"
    shard_dir="${OUTPUT_DIR}/shard${rank}"
    log_path="${OUTPUT_DIR}/logs/shard${rank}.log"
    SHARD_SCORE_PATHS[rank]="${shard_dir}/scores.pt"
    probe_args=(
        HESSIAN_PROBE_LAYER=
        HESSIAN_PROBE_OUT=
        HESSIAN_PROBE_E=
        HESSIAN_PROBE_F=
        HESSIAN_PROBE_VALIDATE=0
    )
    if [[ " ${layers} " == *" ${HESSIAN_PROBE_LAYER} "* ]]; then
        probe_args=(
            HESSIAN_PROBE_LAYER="${HESSIAN_PROBE_LAYER}"
            HESSIAN_PROBE_OUT="${OUTPUT_DIR}/hessian_probe_L${HESSIAN_PROBE_LAYER}.pt"
            HESSIAN_PROBE_E=
            HESSIAN_PROBE_F=
            HESSIAN_PROBE_VALIDATE=0
        )
    fi
    echo "Shard ${rank}: physical GPU ${gpu}, layers=[${layers}], log=${log_path}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi
    worker_command=(
        setsid env
        PYTHONUNBUFFERED=1
        DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX}"
        CUDA_VISIBLE_DEVICES="${gpu}"
        MODEL_PATH="${MODEL_PATH}"
        SELECTION_MANIFEST="${SELECTION_MANIFEST}"
        OUTPUT_DIR="${shard_dir}"
        LAYERS="${layers}"
        "${probe_args[@]}"
        bash scripts/run_collect_scores.sh "$@"
    )
    if [[ "${LIVE_LOGS}" == "1" ]]; then
        "${worker_command[@]}" \
            > >(tee "${log_path}" \
                | tr '\r' '\n' \
                | awk -v prefix="[shard ${rank}] " \
                    'NF { print prefix $0; fflush() }') \
            2>&1 &
    else
        "${worker_command[@]}" >"${log_path}" 2>&1 &
    fi
    PIDS[rank]=$!
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: workers were not started."
    exit 0
fi

failed=0
for ((rank = 0; rank < ${#PIDS[@]}; rank++)); do
    if wait "${PIDS[rank]}"; then
        echo "Shard ${rank} completed: ${SHARD_SCORE_PATHS[rank]}"
    else
        status=$?
        echo "Shard ${rank} failed with status ${status}; see ${OUTPUT_DIR}/logs/shard${rank}.log" >&2
        failed=1
    fi
done
trap - INT TERM

if (( failed != 0 )); then
    echo "At least one score shard failed; final scores.pt was not created." >&2
    exit 1
fi

python scripts/merge_scores.py \
    "${SHARD_SCORE_PATHS[@]}" \
    --strict-metadata \
    --output "${OUTPUT_DIR}/scores.pt"

echo "Four-GPU score collection complete: ${OUTPUT_DIR}/scores.pt"
