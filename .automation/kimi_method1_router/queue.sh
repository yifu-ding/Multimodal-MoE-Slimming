#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dyf/code/distill/MAES
STATE="${ROOT}/.automation/kimi_method1_router"
REPORT="${STATE}/REPORT.md"
SUPERVISOR=/home/dyf/.codex/skills/background-todo-runner/scripts/tmux_supervisor.sh
WORKER_SESSION=kimi-method1-router-worker
SUPERVISOR_SESSION=kimi-method1-router-supervisor

record() {
    printf -- '- %s: %s\n' "$(date '+%F %H:%M %Z')" "$1" >> "${REPORT}"
}

printf '%s\n' queued > "${STATE}/current.txt"
record "QUEUED | waiting for the current kimi-direct campaign to validate 28/28 benchmarks and 2/2 Judge stages."
while true; do
    set +e
    bash "${ROOT}/.automation/kimi_direct/completion_check.sh"
    status=$?
    set -e
    if (( status == 0 )); then
        break
    fi
    if (( status == 2 )); then
        record "ATTENTION | current kimi-direct completion state is invalid; method1-only campaign was not started."
        exit 2
    fi
    sleep 60
done

printf '%s\n' waiting-for-resources > "${STATE}/current.txt"
record "READY | current kimi-direct campaign is complete; waiting for all four GPUs to become free."
until bash "${STATE}/resource_check.sh"; do
    sleep 60
done

if ! tmux has-session -t "${WORKER_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${WORKER_SESSION}" \
        "cd '${ROOT}' && exec bash '${STATE}/pipeline.sh' >> '${STATE}/pipeline.log' 2>&1"
fi
if ! tmux has-session -t "${SUPERVISOR_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SUPERVISOR_SESSION}" \
        "WORKSPACE='${ROOT}' TODO_FILE='${STATE}/TODO.md' REPORT_FILE='${REPORT}' PIPELINE_SESSION='${WORKER_SESSION}' PIPELINE_SCRIPT='${STATE}/pipeline.sh' COMPLETION_CHECK_SCRIPT='${STATE}/completion_check.sh' PROGRESS_SCRIPT='${STATE}/progress.sh' RESOURCE_CHECK_SCRIPT='${STATE}/resource_check.sh' STATE_DIR='${STATE}/supervisor' PIPELINE_LOG='${STATE}/pipeline.log' INTERVAL_SECONDS=5400 AUTO_RESTART=0 exec bash '${SUPERVISOR}' >> '${STATE}/supervisor.log' 2>&1"
fi
record "STARTED | worker=${WORKER_SESSION}, supervisor=${SUPERVISOR_SESSION}."
