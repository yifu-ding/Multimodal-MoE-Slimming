#!/usr/bin/env bash
set -euo pipefail

# Start a text-only OpenAI-compatible judge after the four-GPU EP inference
# process has exited. Qwen2.5 avoids reasoning-tag formatting in judge output.

export HF_HOME="${MAES_HF_HOME:-/home/data/dyf/hf_cache}"
export HF_HUB_CACHE="${MAES_HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${JUDGE_HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${JUDGE_TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm-maes}"
JUDGE_MODEL="${JUDGE_MODEL:-/home/data/dyf/models/Qwen2.5-32B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-mm-judge}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is not available in PATH." >&2
    exit 2
fi
if ! conda env list | awk -v target="${CONDA_ENV_NAME}" '$1 == target { found=1 } END { exit(found ? 0 : 1) }'; then
    echo "error: conda environment '${CONDA_ENV_NAME}' does not exist." >&2
    exit 2
fi
if [[ "${JUDGE_MODEL}" == */* ]] && [[ ! -e "${JUDGE_MODEL}" ]]; then
    MODEL_CACHE_DIR="${HF_HUB_CACHE}/models--${JUDGE_MODEL//\//--}"
    if [[ "${HF_HUB_OFFLINE}" == "1" ]] && [[ ! -d "${MODEL_CACHE_DIR}" ]]; then
        echo "error: judge model is not cached at ${MODEL_CACHE_DIR}." >&2
        echo "       Download it once before starting the offline server:" >&2
        echo "       HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 conda run -n ${CONDA_ENV_NAME} hf download ${JUDGE_MODEL}" >&2
        exit 2
    fi
fi

echo "Starting judge model '${JUDGE_MODEL}' as '${SERVED_MODEL_NAME}'."
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"

exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    vllm serve "${JUDGE_MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
