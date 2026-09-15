#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dyf/code/distill/MAES}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:?Set PREDICTIONS_DIR to a completed baseline RUN_DIR.}"
TASKS="${TASKS:-mmvet,mmbench,videommmu}"
JUDGE_MAX_ATTEMPTS="${JUDGE_MAX_ATTEMPTS:-2}"
PORT="${PORT:-8010}"
JUDGE_LOG="${JUDGE_LOG:-${PREDICTIONS_DIR}/local_judge/server.log}"
JUDGE_MODEL="${JUDGE_MODEL:-/home/data/dyf/models/Qwen2.5-32B-Instruct}"
SERVER_PID=""

if [[ ! "${JUDGE_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: JUDGE_MAX_ATTEMPTS must be a positive integer." >&2
    exit 2
fi

available_tasks=()
IFS=',' read -r -a requested_tasks <<< "${TASKS}"
for requested in "${requested_tasks[@]}"; do
    case "${requested}" in
        mmvet) pattern='*_samples_mmvet.jsonl'; normalized=mmvet ;;
        mmbench) pattern='*_samples_mmbench_en_dev*.jsonl'; normalized=mmbench ;;
        videommmu|video_mmmu) pattern='*_samples_video_mmmu_*_local.jsonl'; normalized=videommmu ;;
        *) echo "warning: unknown Judge task ${requested}; skipping." >&2; continue ;;
    esac
    if find "${PREDICTIONS_DIR}" -type f -name "${pattern}" -not -path '*/local_judge/*' -print -quit 2>/dev/null | grep -q .; then
        available_tasks+=("${normalized}")
    else
        echo "warning: no prediction file for Judge task ${requested}; skipping." >&2
    fi
done
if (( ${#available_tasks[@]} == 0 )); then
    echo "No requested Judge task has predictions; nothing to do."
    exit 0
fi
IFS=,
TASKS="${available_tasks[*]}"
unset IFS

mkdir -p "$(dirname "${JUDGE_LOG}")"

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap stop_server EXIT INT TERM

cd "${REPO_ROOT}"
if curl --silent --fail --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "error: judge port ${PORT} is already serving a model; refusing to use an unverified server." >&2
    exit 2
fi
setsid env \
    JUDGE_MODEL="${JUDGE_MODEL}" \
    SERVED_MODEL_NAME=local-mm-judge \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}" \
    PORT="${PORT}" \
    bash scripts/serve_vllm_mm_judge.sh >> "${JUDGE_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "error: judge server exited before becoming ready; see ${JUDGE_LOG}" >&2
        exit 1
    fi
    if curl --silent --fail --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
if [[ "${ready}" != "1" ]]; then
    echo "error: judge server was not ready within 30 minutes; see ${JUDGE_LOG}" >&2
    exit 1
fi

judge_status=0
for ((attempt = 1; attempt <= JUDGE_MAX_ATTEMPTS; attempt++)); do
    echo "[judge attempt] ${attempt}/${JUDGE_MAX_ATTEMPTS}; tasks=${TASKS}"
    set +e
    conda run --no-capture-output -n vllm-maes \
        python scripts/judge_vllm_predictions.py \
            --predictions-dir "${PREDICTIONS_DIR}" \
            --tasks "${TASKS}" \
            --api-base "http://127.0.0.1:${PORT}/v1" \
            --workers "${JUDGE_WORKERS:-8}"
    judge_status=$?
    set -e
    (( judge_status == 0 )) && exit 0
    (( attempt < JUDGE_MAX_ATTEMPTS )) && echo "warning: Judge attempt failed; retrying persisted failures once." >&2
done
exit "${judge_status}"
