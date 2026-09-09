#!/usr/bin/env bash
set -euo pipefail

# Unpruned Qwen3-VL-30B-A3B baseline through lmms-eval's vLLM backend.
# This is intentionally separate from every existing MAES prune/eval script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${MAES_HF_HOME:-/home/data/dyf/hf_cache}"
export HF_HUB_CACHE="${MAES_HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${MAES_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-18000000}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm-maes}"
PARALLEL_MODE="${PARALLEL_MODE:-ep4}"

# These names are the lmms-eval task IDs corresponding to MAES's offline suites.
DEFAULT_TASKS="gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,mvbench,videomme,longvideobench_val_v,video_mmmu_local"
TASKS="${TASKS:-${DEFAULT_TASKS}}"

# Keep this identical between baseline and future pruned runs.
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_FRAME_NUM="${MAX_FRAME_NUM:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
LIMIT="${LIMIT:-}"
VERBOSITY="${VERBOSITY:-INFO}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
WORKERS="${WORKERS:-16}"
export WORKERS

case "${PARALLEL_MODE}" in
    ep4)
        TENSOR_PARALLEL_SIZE=4
        ENABLE_EXPERT_PARALLEL=True
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
        ;;
    single)
        TENSOR_PARALLEL_SIZE=1
        ENABLE_EXPERT_PARALLEL=False
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        ;;
    *)
        echo "error: PARALLEL_MODE must be 'ep4' or 'single'; got '${PARALLEL_MODE}'." >&2
        exit 2
        ;;
esac

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES// /}"
if (( ${#VISIBLE_GPU_IDS[@]} < TENSOR_PARALLEL_SIZE )); then
    echo "error: mode=${PARALLEL_MODE} needs ${TENSOR_PARALLEL_SIZE} visible GPU(s)," >&2
    echo "       but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is not available in PATH." >&2
    exit 2
fi

if ! conda env list | awk -v target="${CONDA_ENV_NAME}" '$1 == target { found=1 } END { exit(found ? 0 : 1) }'; then
    echo "error: conda environment '${CONDA_ENV_NAME}' does not exist." >&2
    echo "       Create the pinned vLLM environment before running this baseline." >&2
    exit 2
fi

PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)

if ! "${PYTHON_CMD[@]}" -c 'import lmms_eval, vllm' >/dev/null 2>&1; then
    echo "error: '${CONDA_ENV_NAME}' cannot import both vllm and the local lmms_eval package." >&2
    echo "       PYTHONPATH=${PYTHONPATH}" >&2
    exit 2
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    EAGER_ARG=True
elif [[ "${ENFORCE_EAGER}" == "0" ]]; then
    EAGER_ARG=False
else
    echo "error: ENFORCE_EAGER must be 0 or 1; got '${ENFORCE_EAGER}'." >&2
    exit 2
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vllm_baseline/qwen3-vl-30b-a3b}"
RUN_DIR="${OUTPUT_ROOT}/${PARALLEL_MODE}-${RUN_TIMESTAMP}"
mkdir -p "${RUN_DIR}"
export VIDEO_MMMU_ROOT="${VIDEO_MMMU_ROOT:-${HF_DATASETS_CACHE}/VideoMMMU}"
export VIDEO_MMMU_MEDIA_LOG="${VIDEO_MMMU_MEDIA_LOG:-${RUN_DIR}/video_mmmu_media_paths.jsonl}"

MODEL_ARGS="model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},enable_expert_parallel=${ENABLE_EXPERT_PARALLEL},dtype=bfloat16,enforce_eager=${EAGER_ARG},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},max_frame_num=${MAX_FRAME_NUM},max_new_tokens=${MAX_NEW_TOKENS},trust_remote_code=True,disable_log_stats=False"

# GPT-dependent tasks run in output-only mode after the regular benchmark pass.
# This keeps the four inference GPUs dedicated to the evaluated model. Their
# JSONL predictions can be scored after the model process has exited.
IFS=',' read -r -a REQUESTED_TASKS <<< "${TASKS}"
LOCAL_TASKS=()
DEFERRED_JUDGE_TASKS=()
for task in "${REQUESTED_TASKS[@]}"; do
    task="${task//[[:space:]]/}"
    [[ -z "${task}" ]] && continue
    case "${task}" in
        mmvet|mmbench_en_dev)
            DEFERRED_JUDGE_TASKS+=("${task}")
            ;;
        *)
            LOCAL_TASKS+=("${task}")
            ;;
    esac
done

join_by_comma() {
    local IFS=,
    echo "$*"
}

build_command() {
    local task_csv="$1"
    local predict_only="$2"
    CMD=(
        "${PYTHON_CMD[@]}" -m lmms_eval
        --model vllm
        --model_args "${MODEL_ARGS}"
        --tasks "${task_csv}"
        --include_path "${REPO_ROOT}/eval/vllm_tasks"
        --batch_size "${BATCH_SIZE}"
        --output_path "${RUN_DIR}"
        --verbosity "${VERBOSITY}"
        --show_config
    )
    if [[ -n "${LIMIT}" ]]; then
        CMD+=(--limit "${LIMIT}")
    fi
    if [[ "${LOG_SAMPLES}" == "1" ]] || [[ "${predict_only}" == "1" ]]; then
        CMD+=(--log_samples --log_samples_suffix "qwen3_vl_vllm_${PARALLEL_MODE}")
    fi
    if [[ "${predict_only}" == "1" ]]; then
        CMD+=(--predict_only)
    fi
}

LOCAL_TASKS_CSV="$(join_by_comma "${LOCAL_TASKS[@]}")"
DEFERRED_TASKS_CSV="$(join_by_comma "${DEFERRED_JUDGE_TASKS[@]}")"

{
    echo "run_timestamp=${RUN_TIMESTAMP}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git branch --show-current)"
    echo "model=${MODEL}"
    echo "parallel_mode=${PARALLEL_MODE}"
    echo "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    echo "enable_expert_parallel=${ENABLE_EXPERT_PARALLEL}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "tasks=${TASKS}"
    echo "locally_scored_tasks=${LOCAL_TASKS_CSV:-none}"
    echo "deferred_judge_tasks=${DEFERRED_TASKS_CSV:-none}"
    echo "batch_size=${BATCH_SIZE}"
    echo "limit=${LIMIT:-full}"
    echo "dtype=bfloat16"
    echo "enforce_eager=${EAGER_ARG}"
    echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_frame_num=${MAX_FRAME_NUM}"
    echo "hf_home=${HF_HOME}"
} | tee "${RUN_DIR}/run_config.txt"

"${PYTHON_CMD[@]}" -c 'import platform, torch, transformers, vllm; print("python=" + platform.python_version()); print("torch=" + torch.__version__); print("transformers=" + transformers.__version__); print("vllm=" + vllm.__version__); print("torch_cuda=" + str(torch.version.cuda))' \
    | tee "${RUN_DIR}/versions.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    | tee "${RUN_DIR}/gpu_before.txt"

START_EPOCH="$(date +%s)"
RUN_STATUS=0

run_stage() {
    local stage_name="$1"
    local task_csv="$2"
    local predict_only="$3"
    local stage_log="${RUN_DIR}/${stage_name}.log"

    [[ -z "${task_csv}" ]] && return 0
    build_command "${task_csv}" "${predict_only}"
    {
        printf '%s_command=' "${stage_name}"
        printf '%q ' "${CMD[@]}"
        printf '\n'
    } | tee -a "${RUN_DIR}/run_config.txt"

    set +e
    "${CMD[@]}" 2>&1 | tee "${stage_log}"
    local stage_status=${PIPESTATUS[0]}
    set -e
    if (( stage_status != 0 )); then
        RUN_STATUS=${stage_status}
        return "${stage_status}"
    fi
}

if ! run_stage "local_metrics" "${LOCAL_TASKS_CSV}" 0; then
    echo "error: locally scored benchmark stage failed; deferred generation was not started." >&2
elif ! run_stage "deferred_predictions" "${DEFERRED_TASKS_CSV}" 1; then
    echo "error: deferred judge prediction stage failed." >&2
fi

END_EPOCH="$(date +%s)"

{
    echo "exit_code=${RUN_STATUS}"
    echo "wall_time_seconds=$((END_EPOCH - START_EPOCH))"
} | tee "${RUN_DIR}/run_summary.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    | tee "${RUN_DIR}/gpu_after.txt"

echo "Results: ${RUN_DIR}"
if [[ -n "${DEFERRED_TASKS_CSV}" ]] && (( RUN_STATUS == 0 )); then
    echo "Deferred predictions are ready. Stop this process, start the local judge, then score:"
    echo "  conda run --no-capture-output -n ${CONDA_ENV_NAME} python scripts/judge_vllm_predictions.py --predictions-dir ${RUN_DIR}"
fi
exit "${RUN_STATUS}"
