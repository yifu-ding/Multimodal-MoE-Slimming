#!/usr/bin/env bash
set -u

WORKSPACE="${WORKSPACE:?Set WORKSPACE to the repository root.}"
TODO_FILE="${TODO_FILE:?Set TODO_FILE to the TODO Markdown file.}"
REPORT_FILE="${REPORT_FILE:?Set REPORT_FILE to the Markdown status report.}"
PIPELINE_SESSION="${PIPELINE_SESSION:?Set PIPELINE_SESSION to the tmux worker name.}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:?Set PIPELINE_SCRIPT to the resumable dispatcher.}"
COMPLETION_CHECK_SCRIPT="${COMPLETION_CHECK_SCRIPT:?Set COMPLETION_CHECK_SCRIPT.}"
PROGRESS_SCRIPT="${PROGRESS_SCRIPT:?Set PROGRESS_SCRIPT.}"
RESOURCE_CHECK_SCRIPT="${RESOURCE_CHECK_SCRIPT:?Set RESOURCE_CHECK_SCRIPT.}"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-5400}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-1800}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
# Reasoning recovery belongs to an explicitly authorized Debug executor.
AUTO_RESTART="${AUTO_RESTART:-0}"
STATE_DIR="${STATE_DIR:-${WORKSPACE}/.automation/supervisor}"
PIPELINE_LOG="${PIPELINE_LOG:-${WORKSPACE}/.automation/pipeline.log}"
STATE_FILE="${STATE_DIR}/state.tsv"
ALERT_FILE="${STATE_DIR}/alert-state"

mkdir -p "${STATE_DIR}" "$(dirname "${REPORT_FILE}")" "$(dirname "${PIPELINE_LOG}")"
touch "${REPORT_FILE}" "${PIPELINE_LOG}"

timestamp() {
    date '+%F %H:%M %Z'
}

append_report() {
    local content="$1"
    {
        flock 9
        printf '%s\n' "${content}" >> "${REPORT_FILE}"
    } 9>>"${STATE_DIR}/report.lock"
}

alert_once() {
    local key="$1"
    local message="$2"
    local previous=""
    [[ -r "${ALERT_FILE}" ]] && previous="$(<"${ALERT_FILE}")"
    [[ "${previous}" == "${key}" ]] && return 0
    printf '%s\n' "${key}" > "${ALERT_FILE}"
    append_report ""
    append_report "> [!WARNING]"
    append_report "> **ATTENTION ($(timestamp))**  "
    append_report "> ${message}"
}

complete_once() {
    local message="$1"
    local previous=""
    [[ -r "${ALERT_FILE}" ]] && previous="$(<"${ALERT_FILE}")"
    [[ "${previous}" == complete ]] && return 0
    printf '%s\n' complete > "${ALERT_FILE}"
    append_report ""
    append_report "> [!IMPORTANT]"
    append_report "> **DONE ($(timestamp))**  "
    append_report "> ${message}"
}

record_event() {
    append_report "- $(timestamp): $1"
}

progress_text() {
    local output
    if output="$(bash "${PROGRESS_SCRIPT}" 2>&1)"; then
        printf '%s' "$(tr '\n' ' ' <<< "${output}" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
    else
        printf 'progress check failed: %s' "$(tr '\n' ' ' <<< "${output}" | sed -E 's/[[:space:]]+/ /g')"
    fi
}

progress_hash() {
    printf '%s' "$1" | sha256sum | awk '{print $1}'
}

read_state() {
    local attempts=0 last_restart=0 old_hash=none
    if [[ -r "${STATE_FILE}" ]]; then
        IFS=$'\t' read -r attempts last_restart old_hash < "${STATE_FILE}" || true
    fi
    [[ "${attempts}" =~ ^[0-9]+$ ]] || attempts=0
    [[ "${last_restart}" =~ ^[0-9]+$ ]] || last_restart=0
    printf '%s\t%s\t%s' "${attempts}" "${last_restart}" "${old_hash:-none}"
}

write_state() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" > "${STATE_FILE}"
}

worker_is_active() {
    tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null
}

start_worker() {
    local command
    printf -v command 'cd %q && exec bash %q >> %q 2>&1' \
        "${WORKSPACE}" "${PIPELINE_SCRIPT}" "${PIPELINE_LOG}"
    tmux new-session -d -s "${PIPELINE_SESSION}" "${command}"
}

inspect_once() {
    local progress hash state attempts last_restart old_hash now since_restart check_status
    progress="$(progress_text)"
    hash="$(progress_hash "${progress}")"
    state="$(read_state)"
    IFS=$'\t' read -r attempts last_restart old_hash <<< "${state}"
    now="$(date +%s)"

    if [[ "${hash}" != "${old_hash}" ]]; then
        attempts=0
        write_state "${attempts}" "${last_restart}" "${hash}"
    fi

    if worker_is_active; then
        printf '%s\n' healthy > "${ALERT_FILE}"
        record_event "RUNNING | ${progress} | session=${PIPELINE_SESSION}."
        return 0
    fi

    bash "${COMPLETION_CHECK_SCRIPT}"
    check_status=$?
    if (( check_status == 0 )); then
        complete_once "Authorized TODO scope is complete; ${progress}. Supervisor remains active."
        return 0
    fi
    if (( check_status != 1 )); then
        alert_once invalid-state "Worker is absent and completion state is invalid; ${progress}. Inspect ${COMPLETION_CHECK_SCRIPT} and ${PIPELINE_LOG}."
        return 0
    fi

    if [[ "${AUTO_RESTART}" != "1" ]]; then
        alert_once diagnosis-required "Worker stopped with unfinished work; diagnose, fix and validate before resuming. Automatic shell restart is disabled; use the authorized Debug executor. ${progress}."
        return 0
    fi

    if (( attempts >= MAX_RESTARTS )); then
        alert_once retry-limit "Worker stopped with unfinished work and no progress after ${attempts} restarts; ${progress}. Debug ${PIPELINE_LOG} before resuming."
        return 0
    fi

    since_restart=$(( now - last_restart ))
    if (( last_restart > 0 && since_restart < COOLDOWN_SECONDS )); then
        alert_once cooldown "Worker stopped with unfinished work; restart cooldown is active. ${progress}."
        return 0
    fi
    if ! bash "${RESOURCE_CHECK_SCRIPT}"; then
        alert_once resources-busy "Worker stopped with unfinished work, but required resources are busy; ${progress}. Next check will retry."
        return 0
    fi

    attempts=$(( attempts + 1 ))
    write_state "${attempts}" "${now}" "${hash}"
    if start_worker; then
        printf '%s\n' recovering > "${ALERT_FILE}"
        record_event "RECOVERY | restarted session=${PIPELINE_SESSION} (${attempts}/${MAX_RESTARTS}) | ${progress}."
    else
        alert_once launch-failed "Failed to restart session=${PIPELINE_SESSION}; ${progress}. Inspect tmux and ${PIPELINE_LOG}."
    fi
}

for required in \
    "${TODO_FILE}" \
    "${PIPELINE_SCRIPT}" \
    "${COMPLETION_CHECK_SCRIPT}" \
    "${PROGRESS_SCRIPT}" \
    "${RESOURCE_CHECK_SCRIPT}"; do
    if [[ ! -r "${required}" ]]; then
        alert_once config-error "Required workflow file is not readable: ${required}."
        exit 2
    fi
done

inspect_once
while true; do
    sleep "${INTERVAL_SECONDS}"
    inspect_once
done
