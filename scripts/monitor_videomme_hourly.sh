#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dyf/code/distill/MAES}"
REPORT="${REPORT:-${REPO_ROOT}/docs/自动化执行结果.md}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/vllm_baseline/qwen3-vl-30b-a3b/ep4-videomme-native-video}"
TASK_NAME="${TASK_NAME:-videomme_qwen3_vllm}"
PRIMARY_SESSION="${PRIMARY_SESSION:-maes-todo-auto-v4}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"

append_record() {
    printf -- '- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "${REPORT}"
}

latest_file() {
    find "$1" -type f -name "$2" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

latest_persisted() {
    local log_path="$1"
    [[ -n "${log_path}" && -f "${log_path}" ]] || return 0
    tr '\r' '\n' < "${log_path}" \
        | sed $'s/\033\[[0-9;]*m//g' \
        | sed -n 's/.*ResponseCache: persisted \([0-9][0-9]*\/[0-9][0-9]*\) new responses.*/\1/p' \
        | tail -n 1
}

if [[ ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "INTERVAL_SECONDS must be a positive integer." >&2
    exit 2
fi

while :; do
    sleep "${INTERVAL_SECONDS}"

    if [[ -f "${RUN_DIR}/status/${TASK_NAME}.complete" ]]; then
        append_record "Video-MME 小时监控结束：任务已完成。"
        exit 0
    fi
    if [[ -f "${RUN_DIR}/status/${TASK_NAME}.failed" ]]; then
        append_record "Video-MME 小时监控结束：修复后重试仍失败，主接力器将跳到下一项。"
        exit 0
    fi
    if ! tmux has-session -t "${PRIMARY_SESSION}" 2>/dev/null; then
        append_record "Video-MME 小时监控结束：主执行器 ${PRIMARY_SESSION} 已退出，任务没有 complete/failed marker。"
        exit 1
    fi

    log_path="$(latest_file "${RUN_DIR}/logs" "${TASK_NAME}*.log")"
    persisted="$(latest_persisted "${log_path}")"
    heartbeat="$(latest_file "${RUN_DIR}/watchdog/${TASK_NAME}" response_cache.json)"
    if [[ -n "${heartbeat}" && -f "${heartbeat}" ]]; then
        heartbeat_age=$(( $(date +%s) - $(stat -c %Y "${heartbeat}") ))
    else
        heartbeat_age=-1
    fi
    append_record "Video-MME 进度 ${persisted:-尚无持久化响应}；最新逐条心跳距今 ${heartbeat_age}s；主执行器 ${PRIMARY_SESSION} 存活。"
done
