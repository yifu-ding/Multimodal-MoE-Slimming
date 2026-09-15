#!/usr/bin/env bash
set -euo pipefail

# Unpruned Qwen3-VL-30B-A3B baseline through lmms-eval's vLLM backend.
# This is intentionally separate from every existing MAES prune/eval script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

"${SCRIPT_DIR}/apply_lmms_eval_vllm_patch.sh"

export HF_HOME="${MAES_HF_HOME:-/home/data/dyf/hf_cache}"
export HF_HUB_CACHE="${MAES_HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${MAES_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-18000000}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm-maes}"
PARALLEL_MODE="${PARALLEL_MODE:-ep4}"
BASELINE_LABEL="${BASELINE_LABEL:-qwen3_vl_vllm}"
ENABLE_QWEN3_NATIVE_VIDEO="${ENABLE_QWEN3_NATIVE_VIDEO:-1}"
LIMIT_MM_PER_PROMPT_JSON="${LIMIT_MM_PER_PROMPT_JSON:-}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_qwen3_vl_vllm_baseline.sh}"

# These names are the lmms-eval task IDs corresponding to MAES's offline suites.
DEFAULT_TASKS="${DEFAULT_TASKS:-gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,mvbench,videomme,longvideobench_val_v,video_mmmu_local}"
TASKS="${TASKS:-${DEFAULT_TASKS}}"

# Keep this identical between baseline and future pruned runs.
BATCH_SIZE="${BATCH_SIZE:-8}"
LIGHT_IMAGE_BATCH_SIZE="${LIGHT_IMAGE_BATCH_SIZE:-32}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-16}"
VIDEO_BATCH_SIZE="${VIDEO_BATCH_SIZE:-8}"
VIDEOMME_BATCH_SIZE="${VIDEOMME_BATCH_SIZE:-1}"
VIDEO_MMMU_BATCH_SIZE="${VIDEO_MMMU_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
VIDEOMME_MAX_MODEL_LEN="${VIDEOMME_MAX_MODEL_LEN:-262144}"
VIDEO_MMMU_MAX_MODEL_LEN="${VIDEO_MMMU_MAX_MODEL_LEN:-262144}"
VIDEOMME_MAX_NUM_BATCHED_TOKENS="${VIDEOMME_MAX_NUM_BATCHED_TOKENS:-262144}"
VIDEO_MMMU_MAX_NUM_BATCHED_TOKENS="${VIDEO_MMMU_MAX_NUM_BATCHED_TOKENS:-262144}"
MAX_FRAME_NUM="${MAX_FRAME_NUM:-32}"
VIDEO_NFRAMES="${VIDEO_NFRAMES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
IMAGE_FIRST="${IMAGE_FIRST:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
LIMIT="${LIMIT:-}"
VERBOSITY="${VERBOSITY:-INFO}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
WORKERS="${WORKERS:-16}"
FORCE="${FORCE:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
TASK_MAX_ATTEMPTS="${TASK_MAX_ATTEMPTS:-2}"
ENABLE_RESPONSE_CACHE="${ENABLE_RESPONSE_CACHE:-1}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-1800}"
WATCHDOG_POLL_SECONDS="${WATCHDOG_POLL_SECONDS:-30}"
export WORKERS
export QWEN3_VIDEOMME_FPS="${QWEN3_VIDEOMME_FPS:-2}"
export QWEN3_VIDEOMME_MAX_FRAMES="${QWEN3_VIDEOMME_MAX_FRAMES:-2048}"
export QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME="${QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME:-128}"
export QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME="${QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME:-640}"
export QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS="${QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS:-224000}"
export QWEN3_VIDEOMMMU_FPS="${QWEN3_VIDEOMMMU_FPS:-2}"
export QWEN3_VIDEOMMMU_MAX_FRAMES="${QWEN3_VIDEOMMMU_MAX_FRAMES:-2048}"
export QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME="${QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME:-768}"
export QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS="${QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS:-224000}"

for binary_flag in IMAGE_FIRST LOG_SAMPLES FORCE FAIL_FAST ENABLE_QWEN3_NATIVE_VIDEO ENABLE_RESPONSE_CACHE; do
    if [[ "${!binary_flag}" != "0" && "${!binary_flag}" != "1" ]]; then
        echo "error: ${binary_flag} must be 0 or 1; got '${!binary_flag}'." >&2
        exit 2
    fi
done
for batch_var in BATCH_SIZE LIGHT_IMAGE_BATCH_SIZE IMAGE_BATCH_SIZE VIDEO_BATCH_SIZE VIDEOMME_BATCH_SIZE VIDEO_MMMU_BATCH_SIZE; do
    if [[ ! "${!batch_var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ${batch_var} must be a positive integer; got '${!batch_var}'." >&2
        exit 2
    fi
done
for positive_int_var in VIDEOMME_MAX_MODEL_LEN VIDEO_MMMU_MAX_MODEL_LEN \
    VIDEOMME_MAX_NUM_BATCHED_TOKENS VIDEO_MMMU_MAX_NUM_BATCHED_TOKENS \
    STALL_TIMEOUT_SECONDS WATCHDOG_POLL_SECONDS QWEN3_VIDEOMME_MAX_FRAMES \
    QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME \
    QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS QWEN3_VIDEOMMMU_MAX_FRAMES \
    QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS; do
    if [[ ! "${!positive_int_var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ${positive_int_var} must be a positive integer; got '${!positive_int_var}'." >&2
        exit 2
    fi
done
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]] && [[ ! "${KV_CACHE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: KV_CACHE_MEMORY_BYTES must be empty or a positive integer; got '${KV_CACHE_MEMORY_BYTES}'." >&2
    exit 2
fi
for optional_positive_int_var in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
    if [[ -n "${!optional_positive_int_var}" ]] && [[ ! "${!optional_positive_int_var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ${optional_positive_int_var} must be empty or a positive integer; got '${!optional_positive_int_var}'." >&2
        exit 2
    fi
done
if [[ -n "${VIDEO_NFRAMES}" ]] && { [[ ! "${VIDEO_NFRAMES}" =~ ^[1-9][0-9]*$ ]] || (( VIDEO_NFRAMES % 2 != 0 )); }; then
    echo "error: VIDEO_NFRAMES must be empty or a positive even integer; got '${VIDEO_NFRAMES}'." >&2
    exit 2
fi
if [[ ! "${TASK_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: TASK_MAX_ATTEMPTS must be a positive integer; got '${TASK_MAX_ATTEMPTS}'." >&2
    exit 2
fi
if [[ ! "${QWEN3_VIDEOMME_FPS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${QWEN3_VIDEOMME_FPS}" == "0" ]]; then
    echo "error: QWEN3_VIDEOMME_FPS must be positive; got '${QWEN3_VIDEOMME_FPS}'." >&2
    exit 2
fi
if [[ ! "${QWEN3_VIDEOMMMU_FPS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${QWEN3_VIDEOMMMU_FPS}" == "0" ]]; then
    echo "error: QWEN3_VIDEOMMMU_FPS must be positive; got '${QWEN3_VIDEOMMMU_FPS}'." >&2
    exit 2
fi
if (( QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME > QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME )); then
    echo "error: VideoMME minimum tokens per frame cannot exceed the maximum." >&2
    exit 2
fi

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

if [[ "${IMAGE_FIRST}" == "1" ]]; then
    IMAGE_FIRST_ARG=True
else
    IMAGE_FIRST_ARG=False
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vllm_baseline/qwen3-vl-30b-a3b}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${PARALLEL_MODE}-${RUN_TIMESTAMP}}"
TASK_OUTPUT_ROOT="${RUN_DIR}/tasks"
TASK_LOG_ROOT="${RUN_DIR}/logs"
TASK_STATUS_ROOT="${RUN_DIR}/status"
mkdir -p "${RUN_DIR}" "${TASK_OUTPUT_ROOT}" "${TASK_LOG_ROOT}" "${TASK_STATUS_ROOT}"
export VIDEO_MMMU_ROOT="${VIDEO_MMMU_ROOT:-${HF_DATASETS_CACHE}/VideoMMMU}"
export VIDEO_MMMU_MEDIA_LOG="${VIDEO_MMMU_MEDIA_LOG:-${RUN_DIR}/video_mmmu_media_paths.jsonl}"
export EGOSCHEMA_ROOT="${EGOSCHEMA_ROOT:-${HF_DATASETS_CACHE}/egoschema}"

# A RUN_DIR may intentionally be reused to fill in missing tasks. Preserve all
# run-level metadata from earlier invocations instead of replacing it.
unique_invocation_path() {
    local canonical="$1"
    if [[ ! -e "${canonical}" ]]; then
        printf '%s' "${canonical}"
        return
    fi
    local stem="${canonical%.*}"
    local extension="${canonical##*.}"
    printf '%s.%s.%s.%s' "${stem}" "${RUN_TIMESTAMP}" "$$" "${extension}"
}

preserve_existing_marker() {
    local marker="$1"
    [[ -e "${marker}" ]] || return 0
    mv "${marker}" "${marker}.${RUN_TIMESTAMP}.$$.previous"
}

RUN_CONFIG_FILE="$(unique_invocation_path "${RUN_DIR}/run_config.txt")"
VERSIONS_FILE="$(unique_invocation_path "${RUN_DIR}/versions.txt")"
GPU_BEFORE_FILE="$(unique_invocation_path "${RUN_DIR}/gpu_before.txt")"
TASK_STATUS_FILE="$(unique_invocation_path "${RUN_DIR}/task_status.tsv")"
RUN_SUMMARY_FILE="$(unique_invocation_path "${RUN_DIR}/run_summary.txt")"
GPU_AFTER_FILE="$(unique_invocation_path "${RUN_DIR}/gpu_after.txt")"

model_args_for_task() {
    local task_name="$1"
    local task_max_model_len="${MAX_MODEL_LEN}"
    if [[ "${task_name}" == "videomme_qwen3_vllm" ]]; then
        task_max_model_len="${VIDEOMME_MAX_MODEL_LEN}"
    elif [[ "${task_name}" == "video_mmmu_local" ]]; then
        task_max_model_len="${VIDEO_MMMU_MAX_MODEL_LEN}"
    fi
    local scheduler_args=""
    local cache_args=""
    local multimodal_args=""
    if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
        cache_args=",kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}"
    fi
    if [[ -n "${MAX_NUM_SEQS}" ]]; then
        scheduler_args+=",max_num_seqs=${MAX_NUM_SEQS}"
    fi
    if [[ -n "${LIMIT_MM_PER_PROMPT_JSON}" ]]; then
        multimodal_args=",limit_mm_per_prompt=${LIMIT_MM_PER_PROMPT_JSON}"
    fi
    if [[ -n "${VIDEO_NFRAMES}" ]]; then
        case "${task_name}" in
            videomme|videomme_qwen3_vllm|longvideobench_val_v|video_mmmu_local|egoschema|egoschema_subset|egoschema_subset_local|mvbench|mvbench_available_2600|mvbench_available_3800)
                multimodal_args+=",nframes=${VIDEO_NFRAMES}"
                ;;
        esac
    fi
    if [[ "${task_name}" == "videomme_qwen3_vllm" ]]; then
        scheduler_args+=",max_num_batched_tokens=${VIDEOMME_MAX_NUM_BATCHED_TOKENS}"
    elif [[ "${task_name}" == "video_mmmu_local" ]] && task_uses_native_video "${task_name}"; then
        scheduler_args+=",max_num_batched_tokens=${VIDEO_MMMU_MAX_NUM_BATCHED_TOKENS}"
    elif [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
        scheduler_args+=",max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    fi
    printf '%s' "model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},enable_expert_parallel=${ENABLE_EXPERT_PARALLEL},dtype=bfloat16,enforce_eager=${EAGER_ARG},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${task_max_model_len}${scheduler_args}${cache_args}${multimodal_args},max_frame_num=${MAX_FRAME_NUM},max_new_tokens=${MAX_NEW_TOKENS},image_first=${IMAGE_FIRST_ARG},trust_remote_code=True,disable_log_stats=False"
}

task_uses_native_video() {
    local task_name="$1"
    [[ "${ENABLE_QWEN3_NATIVE_VIDEO}" == "1" ]] || return 1
    case "${task_name}" in
        videomme_qwen3_vllm|video_mmmu_local) return 0 ;;
        *) return 1 ;;
    esac
}

response_cache_batch_size_for() {
    local task_name="$1"
    local inference_batch_size="$2"
    case "${task_name}" in
        egoschema|egoschema_subset|egoschema_subset_local|mvbench|mvbench_available_2600|mvbench_available_3800|videomme|videomme_qwen3_vllm|longvideobench_val_v|video_mmmu_local)
            # A malformed or exceptionally long video must not discard other
            # completed responses from the same inference call.
            echo 1
            ;;
        *)
            echo "${inference_batch_size}"
            ;;
    esac
}

# GPT-dependent tasks run in output-only mode after the regular benchmark pass.
# This keeps the four inference GPUs dedicated to the evaluated model. Their
# JSONL predictions can be scored after the model process has exited.
IFS=',' read -r -a REQUESTED_TASKS <<< "${TASKS}"
LOCAL_TASKS=()
DEFERRED_JUDGE_TASKS=()
declare -A SEEN_REQUESTED_TASKS=()
for task in "${REQUESTED_TASKS[@]}"; do
    task="${task//[[:space:]]/}"
    [[ -z "${task}" ]] && continue
    if [[ -n "${SEEN_REQUESTED_TASKS[${task}]:-}" ]]; then
        echo "[deduplicate] ignoring repeated task ${task}"
        continue
    fi
    SEEN_REQUESTED_TASKS["${task}"]=1
    case "${task}" in
        mmvet|mmbench_en_dev)
            DEFERRED_JUDGE_TASKS+=("${task}")
            ;;
        videomme)
            if [[ "${ENABLE_QWEN3_NATIVE_VIDEO}" == "1" ]]; then
                echo "[protocol] mapping videomme to videomme_qwen3_vllm"
                LOCAL_TASKS+=("videomme_qwen3_vllm")
            else
                LOCAL_TASKS+=("videomme")
            fi
            ;;
        egoschema_subset)
            echo "[protocol] mapping egoschema_subset to local parquet/media task"
            LOCAL_TASKS+=("egoschema_subset_local")
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
    local task_name="$1"
    local predict_only="$2"
    local task_output_dir="$3"
    local task_limit="$4"
    local task_batch_size="$5"
    local task_model_args="$6"
    local cache_root="$7"
    local heartbeat_dir="$8"
    local cache_batch_size="$9"
    CMD=(
        env
        "LMMS_CACHE_RUN_ID=${BASELINE_LABEL}-${PARALLEL_MODE}-${task_name}"
        "LMMS_CACHE_WRITE_THROUGH_BATCH_SIZE=${cache_batch_size}"
        "LMMS_CACHE_CHECKPOINT_INTERVAL=1"
        "MAES_EFFICIENCY_TRACE=${MAES_EFFICIENCY_TRACE:-}"
        "MAES_EFFICIENCY_WARMUP_BATCHES=${MAES_EFFICIENCY_WARMUP_BATCHES:-0}"
        bash "${SCRIPT_DIR}/run_with_stall_watchdog.sh"
        --heartbeat-dir "${heartbeat_dir}"
        --stall-timeout-seconds "${STALL_TIMEOUT_SECONDS}"
        --poll-seconds "${WATCHDOG_POLL_SECONDS}"
        --
        "${PYTHON_CMD[@]}" -m lmms_eval
        --model vllm
        --model_args "${task_model_args}"
        --tasks "${task_name}"
        --include_path "${REPO_ROOT}/eval/vllm_tasks"
        --batch_size "${task_batch_size}"
        --output_path "${task_output_dir}"
        --verbosity "${VERBOSITY}"
        --show_config
    )
    if [[ "${ENABLE_RESPONSE_CACHE}" == "1" ]]; then
        CMD+=(--use_cache "${cache_root}")
    fi
    if task_uses_native_video "${task_name}"; then
        # The vllm registry prefers its chat implementation. The Qwen3 native
        # video adapters are implemented in models/simple/vllm.py.
        CMD+=(--force_simple)
    fi
    if [[ -n "${task_limit}" ]]; then
        CMD+=(--limit "${task_limit}")
    fi
    if [[ "${LOG_SAMPLES}" == "1" ]] || [[ "${predict_only}" == "1" ]]; then
        CMD+=(--log_samples --log_samples_suffix "${BASELINE_LABEL}_${PARALLEL_MODE}_${task_name}")
    fi
    if [[ "${predict_only}" == "1" ]]; then
        CMD+=(--predict_only)
    fi
}

task_batch_size_for() {
    local task_name="$1"
    case "${task_name}" in
        gqa|gqa_prefill|coco2017_cap_val_local|mme)
            echo "${LIGHT_IMAGE_BATCH_SIZE}"
            ;;
        textvqa_val|chartqa|mmstar|mmbench_en_dev_static_local|mmbench_en_dev|mmvet|realworldqa)
            echo "${IMAGE_BATCH_SIZE}"
            ;;
        videomme_qwen3_vllm)
            echo "${VIDEOMME_BATCH_SIZE}"
            ;;
        video_mmmu_local)
            echo "${VIDEO_MMMU_BATCH_SIZE}"
            ;;
        egoschema|egoschema_subset|egoschema_subset_local|mvbench|mvbench_available_2600|mvbench_available_3800|videomme|longvideobench_val_v)
            echo "${VIDEO_BATCH_SIZE}"
            ;;
        *)
            echo "${BATCH_SIZE}"
            ;;
    esac
}

LOCAL_TASKS_CSV="$(join_by_comma "${LOCAL_TASKS[@]}")"
DEFERRED_TASKS_CSV="$(join_by_comma "${DEFERRED_JUDGE_TASKS[@]}")"
printf -v RESUME_COMMAND \
    'RUN_DIR=%q TASKS=%q MODEL=%q CONDA_ENV_NAME=%q PARALLEL_MODE=%q CUDA_VISIBLE_DEVICES=%q BATCH_SIZE=%q LIGHT_IMAGE_BATCH_SIZE=%q IMAGE_BATCH_SIZE=%q VIDEO_BATCH_SIZE=%q VIDEOMME_BATCH_SIZE=%q VIDEO_MMMU_BATCH_SIZE=%q GPU_MEMORY_UTILIZATION=%q MAX_MODEL_LEN=%q VIDEOMME_MAX_MODEL_LEN=%q VIDEO_MMMU_MAX_MODEL_LEN=%q VIDEOMME_MAX_NUM_BATCHED_TOKENS=%q VIDEO_MMMU_MAX_NUM_BATCHED_TOKENS=%q QWEN3_VIDEOMME_FPS=%q QWEN3_VIDEOMME_MAX_FRAMES=%q QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME=%q QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME=%q QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS=%q QWEN3_VIDEOMMMU_FPS=%q QWEN3_VIDEOMMMU_MAX_FRAMES=%q QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME=%q QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS=%q MAX_FRAME_NUM=%q VIDEO_NFRAMES=%q MAX_NEW_TOKENS=%q IMAGE_FIRST=%q ENFORCE_EAGER=%q LIMIT=%q LOG_SAMPLES=%q ENABLE_QWEN3_NATIVE_VIDEO=%q ENABLE_RESPONSE_CACHE=%q LIMIT_MM_PER_PROMPT_JSON=%q STALL_TIMEOUT_SECONDS=%q WATCHDOG_POLL_SECONDS=%q TASK_MAX_ATTEMPTS=%q BASELINE_LABEL=%q bash %q' \
    "${RUN_DIR}" "${TASKS}" "${MODEL}" "${CONDA_ENV_NAME}" "${PARALLEL_MODE}" "${CUDA_VISIBLE_DEVICES}" \
    "${BATCH_SIZE}" "${LIGHT_IMAGE_BATCH_SIZE}" "${IMAGE_BATCH_SIZE}" "${VIDEO_BATCH_SIZE}" "${VIDEOMME_BATCH_SIZE}" "${VIDEO_MMMU_BATCH_SIZE}" \
    "${GPU_MEMORY_UTILIZATION}" "${MAX_MODEL_LEN}" "${VIDEOMME_MAX_MODEL_LEN}" "${VIDEO_MMMU_MAX_MODEL_LEN}" \
    "${VIDEOMME_MAX_NUM_BATCHED_TOKENS}" "${VIDEO_MMMU_MAX_NUM_BATCHED_TOKENS}" \
    "${QWEN3_VIDEOMME_FPS}" "${QWEN3_VIDEOMME_MAX_FRAMES}" "${QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME}" \
    "${QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME}" "${QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS}" \
    "${QWEN3_VIDEOMMMU_FPS}" "${QWEN3_VIDEOMMMU_MAX_FRAMES}" \
    "${QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME}" "${QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS}" \
    "${MAX_FRAME_NUM}" "${VIDEO_NFRAMES}" "${MAX_NEW_TOKENS}" "${IMAGE_FIRST}" "${ENFORCE_EAGER}" \
    "${LIMIT}" "${LOG_SAMPLES}" "${ENABLE_QWEN3_NATIVE_VIDEO}" "${ENABLE_RESPONSE_CACHE}" "${LIMIT_MM_PER_PROMPT_JSON}" \
    "${STALL_TIMEOUT_SECONDS}" "${WATCHDOG_POLL_SECONDS}" "${TASK_MAX_ATTEMPTS}" "${BASELINE_LABEL}" "${RUNNER_SCRIPT}"
if [[ -n "${MAES_EP4_PLAN:-}" ]]; then
    printf -v RESUME_COMMAND 'EP4_PLAN=%q MAES_EP4_PLAN=%q PYTHONPATH=%q MODEL=%q %s' \
        "${MAES_EP4_PLAN}" "${MAES_EP4_PLAN}" "${PYTHONPATH}" "${MODEL}" "${RESUME_COMMAND}"
fi

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
    echo "baseline_label=${BASELINE_LABEL}"
    echo "qwen3_native_video=${ENABLE_QWEN3_NATIVE_VIDEO}"
    echo "locally_scored_tasks=${LOCAL_TASKS_CSV:-none}"
    echo "deferred_judge_tasks=${DEFERRED_TASKS_CSV:-none}"
    echo "fallback_batch_size=${BATCH_SIZE}"
    echo "light_image_batch_size=${LIGHT_IMAGE_BATCH_SIZE}"
    echo "image_batch_size=${IMAGE_BATCH_SIZE}"
    echo "video_batch_size=${VIDEO_BATCH_SIZE}"
    echo "videomme_batch_size=${VIDEOMME_BATCH_SIZE}"
    echo "video_mmmu_batch_size=${VIDEO_MMMU_BATCH_SIZE}"
    echo "limit=${LIMIT:-full}"
    echo "dtype=bfloat16"
    echo "enforce_eager=${EAGER_ARG}"
    echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    echo "kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES:-auto}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_num_seqs=${MAX_NUM_SEQS:-auto}"
    echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-auto}"
    echo "videomme_max_model_len=${VIDEOMME_MAX_MODEL_LEN}"
    echo "videomme_max_num_batched_tokens=${VIDEOMME_MAX_NUM_BATCHED_TOKENS}"
    echo "videomme_fps=${QWEN3_VIDEOMME_FPS}"
    echo "videomme_max_frames=${QWEN3_VIDEOMME_MAX_FRAMES}"
    echo "videomme_min_tokens_per_frame=${QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME}"
    echo "videomme_max_tokens_per_frame=${QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME}"
    echo "videomme_total_video_tokens=${QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS}"
    echo "video_mmmu_max_model_len=${VIDEO_MMMU_MAX_MODEL_LEN}"
    echo "video_mmmu_max_num_batched_tokens=${VIDEO_MMMU_MAX_NUM_BATCHED_TOKENS}"
    echo "video_mmmu_fps=${QWEN3_VIDEOMMMU_FPS}"
    echo "video_mmmu_max_frames=${QWEN3_VIDEOMMMU_MAX_FRAMES}"
    echo "video_mmmu_max_tokens_per_frame=${QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME}"
    echo "video_mmmu_total_video_tokens=${QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS}"
    echo "max_frame_num=${MAX_FRAME_NUM}"
    echo "video_nframes=${VIDEO_NFRAMES:-default}"
    echo "fallback_max_new_tokens=${MAX_NEW_TOKENS}"
    echo "image_first=${IMAGE_FIRST_ARG}"
    echo "hf_home=${HF_HOME}"
    echo "run_dir=${RUN_DIR}"
    echo "maes_ep4_plan=${MAES_EP4_PLAN:-disabled}"
    echo "maes_efficiency_trace=${MAES_EFFICIENCY_TRACE:-disabled}"
    echo "maes_efficiency_warmup_batches=${MAES_EFFICIENCY_WARMUP_BATCHES:-0}"
    echo "response_cache=${ENABLE_RESPONSE_CACHE}"
    echo "limit_mm_per_prompt=${LIMIT_MM_PER_PROMPT_JSON:-default}"
    echo "stall_timeout_seconds=${STALL_TIMEOUT_SECONDS}"
    echo "watchdog_poll_seconds=${WATCHDOG_POLL_SECONDS}"
    echo "force=${FORCE}"
    echo "fail_fast=${FAIL_FAST}"
    echo "task_max_attempts=${TASK_MAX_ATTEMPTS}"
    echo "resume_with=${RESUME_COMMAND}"
} | tee "${RUN_CONFIG_FILE}"

"${PYTHON_CMD[@]}" -c 'import platform, torch, transformers, vllm; print("python=" + platform.python_version()); print("torch=" + torch.__version__); print("transformers=" + transformers.__version__); print("vllm=" + vllm.__version__); print("torch_cuda=" + str(torch.version.cuda))' \
    | tee "${VERSIONS_FILE}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    | tee "${GPU_BEFORE_FILE}"

START_EPOCH="$(date +%s)"
RUN_STATUS=0
COMPLETED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
STOP_REQUESTED=0
printf 'task\tstatus\texit_code\tbatch_size\twall_time_seconds\toutput_dir\n' > "${TASK_STATUS_FILE}"

run_task_once() {
    local task_name="$1"
    local predict_only="$2"
    local safe_task="${task_name//\//_}"
    local task_output_dir="${TASK_OUTPUT_ROOT}/${safe_task}"
    local task_log="${TASK_LOG_ROOT}/${safe_task}.log"
    local complete_marker="${TASK_STATUS_ROOT}/${safe_task}.complete"
    local failed_marker="${TASK_STATUS_ROOT}/${safe_task}.failed"
    local task_limit="${LIMIT}"
    local task_batch_size cache_batch_size task_model_args
    local task_start task_end task_wall task_status task_signature
    local cache_root="${RUN_DIR}/response_cache/${safe_task}"
    local heartbeat_dir="${RUN_DIR}/watchdog/${safe_task}/${RUN_TIMESTAMP}-$$"

    task_batch_size="$(task_batch_size_for "${task_name}")"
    cache_batch_size="$(response_cache_batch_size_for "${task_name}" "${task_batch_size}")"
    task_model_args="$(model_args_for_task "${task_name}")"

    # MME scores paired yes/no questions and cannot aggregate an odd prefix.
    if [[ "${task_name}" == "mme" && "${task_limit}" =~ ^[0-9]+$ ]]; then
        local numeric_limit=$((10#${task_limit}))
        if (( numeric_limit % 2 != 0 )); then
            task_limit="$((numeric_limit + 1))"
            echo "warning: MME requires pairs; using LIMIT=${task_limit} instead of LIMIT=${LIMIT}."
        fi
    fi

    task_signature="$({
        printf '%s\n' "${task_name}" "${predict_only}" "${task_model_args}"
        printf '%s\n' "batch_size=${task_batch_size}" "limit=${task_limit}" "log_samples=${LOG_SAMPLES}"
        printf '%s\n' "response_cache=${ENABLE_RESPONSE_CACHE}" "cache_write_batch_size=${cache_batch_size}"
        printf '%s\n' "limit_mm_per_prompt=${LIMIT_MM_PER_PROMPT_JSON:-default}"
        if task_uses_native_video "${task_name}"; then
            printf '%s\n' "model_backend=simple" "native_video=1"
        fi
        if [[ "${task_name}" == "videomme_qwen3_vllm" ]]; then
            printf '%s\n' \
                "fps=${QWEN3_VIDEOMME_FPS}" \
                "max_frames=${QWEN3_VIDEOMME_MAX_FRAMES}" \
                "min_tokens_per_frame=${QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME}" \
                "max_tokens_per_frame=${QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME}" \
                "total_video_tokens=${QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS}"
        elif [[ "${task_name}" == "video_mmmu_local" ]]; then
            printf '%s\n' \
                "fps=${QWEN3_VIDEOMMMU_FPS}" \
                "max_frames=${QWEN3_VIDEOMMMU_MAX_FRAMES}" \
                "max_tokens_per_frame=${QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME}" \
                "total_video_tokens=${QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS}"
        fi
    } | sha256sum | awk '{print $1}')"

    if [[ "${FORCE}" != "1" && -f "${complete_marker}" ]] &&
        grep -qxF "signature=${task_signature}" "${complete_marker}"; then
        echo "[skip] ${task_name}: matching completed result at ${task_output_dir}"
        printf '%s\tskipped\t0\t%s\t0\t%s\n' "${task_name}" "${task_batch_size}" "${task_output_dir}" >> "${TASK_STATUS_FILE}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        return 0
    fi

    if [[ -f "${complete_marker}" ]]; then
        echo "warning: ${task_name} has a completion marker for different settings; rerunning."
    fi
    preserve_existing_marker "${complete_marker}"
    preserve_existing_marker "${failed_marker}"
    mkdir -p "${task_output_dir}"
    task_log="$(unique_invocation_path "${task_log}")"
    build_command "${task_name}" "${predict_only}" "${task_output_dir}" "${task_limit}" "${task_batch_size}" "${task_model_args}" "${cache_root}" "${heartbeat_dir}" "${cache_batch_size}"
    {
        printf 'task_%s_command=' "${safe_task}"
        printf '%q ' "${CMD[@]}"
        printf '\n'
    } | tee -a "${RUN_CONFIG_FILE}"

    echo "[run] ${task_name}: batch_size=${task_batch_size} cache_batch_size=${cache_batch_size} output=${task_output_dir} log=${task_log}"
    task_start="$(date +%s)"
    set +e
    "${CMD[@]}" 2>&1 | tee "${task_log}"
    task_status=${PIPESTATUS[0]}
    set -e
    # lmms-eval currently logs top-level evaluation exceptions but may still
    # return zero, so treat its explicit error marker as a failed task.
    if (( task_status == 0 )) && grep -q "Error during evaluation:" "${task_log}"; then
        task_status=1
    fi
    if (( task_status == 0 )) &&
        ! find "${task_output_dir}" -type f \( -name '*_results.json' -o -name '*_samples_*.jsonl' \) -print -quit | grep -q .; then
        echo "error: ${task_name} exited successfully but produced no result or sample file." >&2
        task_status=1
    fi
    task_end="$(date +%s)"
    task_wall=$((task_end - task_start))

    if (( task_status == 0 )); then
        {
            echo "task=${task_name}"
            echo "signature=${task_signature}"
            echo "completed_at=$(date --iso-8601=seconds)"
            echo "batch_size=${task_batch_size}"
            echo "wall_time_seconds=${task_wall}"
            echo "output_dir=${task_output_dir}"
        } > "${complete_marker}"
        printf '%s\tcomplete\t0\t%s\t%s\t%s\n' "${task_name}" "${task_batch_size}" "${task_wall}" "${task_output_dir}" >> "${TASK_STATUS_FILE}"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        echo "[complete] ${task_name}: ${task_wall}s"
        return 0
    fi

    {
        echo "task=${task_name}"
        echo "signature=${task_signature}"
        echo "failed_at=$(date --iso-8601=seconds)"
        echo "exit_code=${task_status}"
        echo "batch_size=${task_batch_size}"
        echo "wall_time_seconds=${task_wall}"
        echo "log=${task_log}"
    } > "${failed_marker}"
    printf '%s\tfailed\t%s\t%s\t%s\t%s\n' "${task_name}" "${task_status}" "${task_batch_size}" "${task_wall}" "${task_output_dir}" >> "${TASK_STATUS_FILE}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    echo "error: ${task_name} failed with exit code ${task_status}; see ${task_log}." >&2
    return "${task_status}"
}

run_task() {
    local task_name="$1"
    local predict_only="$2"
    local attempt=1
    local task_status=0

    while (( attempt <= TASK_MAX_ATTEMPTS )); do
        echo "[attempt] ${task_name}: ${attempt}/${TASK_MAX_ATTEMPTS}"
        if run_task_once "${task_name}" "${predict_only}"; then
            return 0
        else
            task_status=$?
        fi

        if (( task_status == 130 || task_status == 143 || attempt == TASK_MAX_ATTEMPTS )); then
            return "${task_status}"
        fi

        echo "warning: ${task_name} attempt ${attempt}/${TASK_MAX_ATTEMPTS} failed; retrying once with the persisted response cache." >&2
        attempt=$((attempt + 1))
    done

    return "${task_status}"
}

for task_name in "${LOCAL_TASKS[@]}"; do
    if run_task "${task_name}" 0; then
        task_status=0
    else
        task_status=$?
    fi
    if (( task_status != 0 )); then
        (( RUN_STATUS == 0 )) && RUN_STATUS=${task_status}
        if (( task_status == 130 || task_status == 143 )); then
            STOP_REQUESTED=1
            echo "Interrupted; completed task markers were preserved for resume." >&2
            break
        fi
        [[ "${FAIL_FAST}" == "1" ]] && break
    fi
done

if (( STOP_REQUESTED == 0 )) && { (( RUN_STATUS == 0 )) || [[ "${FAIL_FAST}" != "1" ]]; }; then
    for task_name in "${DEFERRED_JUDGE_TASKS[@]}"; do
        if run_task "${task_name}" 1; then
            task_status=0
        else
            task_status=$?
        fi
        if (( task_status != 0 )); then
            (( RUN_STATUS == 0 )) && RUN_STATUS=${task_status}
            if (( task_status == 130 || task_status == 143 )); then
                STOP_REQUESTED=1
                echo "Interrupted; completed task markers were preserved for resume." >&2
                break
            fi
            [[ "${FAIL_FAST}" == "1" ]] && break
        fi
    done
fi

END_EPOCH="$(date +%s)"

{
    echo "exit_code=${RUN_STATUS}"
    echo "wall_time_seconds=$((END_EPOCH - START_EPOCH))"
    echo "completed_this_invocation=${COMPLETED_COUNT}"
    echo "skipped_completed=${SKIPPED_COUNT}"
    echo "failed_this_invocation=${FAILED_COUNT}"
    echo "task_status_file=${TASK_STATUS_FILE}"
} | tee "${RUN_SUMMARY_FILE}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    | tee "${GPU_AFTER_FILE}"

echo "Results: ${RUN_DIR}"
echo "Resume this run (matching completed tasks will be skipped):"
echo "  ${RESUME_COMMAND}"
if [[ -n "${DEFERRED_TASKS_CSV}" ]] && (( RUN_STATUS == 0 )); then
    echo "Deferred predictions are ready. Stop this process, start the local judge, then score:"
    echo "  conda run --no-capture-output -n ${CONDA_ENV_NAME} python scripts/judge_vllm_predictions.py --predictions-dir ${RUN_DIR}"
fi
exit "${RUN_STATUS}"
