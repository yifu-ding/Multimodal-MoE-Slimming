#!/usr/bin/env bash
set -euo pipefail

HEARTBEAT_DIR=""
STALL_TIMEOUT_SECONDS=1800
POLL_SECONDS=30

while (( $# > 0 )); do
    case "$1" in
        --heartbeat-dir)
            HEARTBEAT_DIR="$2"
            shift 2
            ;;
        --stall-timeout-seconds)
            STALL_TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "error: unknown watchdog argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${HEARTBEAT_DIR}" || $# -eq 0 ]]; then
    echo "usage: $0 --heartbeat-dir DIR [--stall-timeout-seconds N] [--poll-seconds N] -- COMMAND..." >&2
    exit 2
fi
for value_name in STALL_TIMEOUT_SECONDS POLL_SECONDS; do
    if [[ ! "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ${value_name} must be a positive integer; got ${!value_name}" >&2
        exit 2
    fi
done

mkdir -p "${HEARTBEAT_DIR}"
START_EPOCH="$(date +%s)"
export LMMS_WATCHDOG_DIR="${HEARTBEAT_DIR}"

setsid "$@" &
COMMAND_PID=$!

terminate_group() {
    local signal="$1"
    kill "-${signal}" -- "-${COMMAND_PID}" 2>/dev/null || \
        kill "-${signal}" "${COMMAND_PID}" 2>/dev/null || true
}

cleanup() {
    if kill -0 "${COMMAND_PID}" 2>/dev/null; then
        terminate_group TERM
    fi
}
trap cleanup INT TERM EXIT

while kill -0 "${COMMAND_PID}" 2>/dev/null; do
    for ((elapsed = 0; elapsed < POLL_SECONDS; elapsed++)); do
        sleep 1 &
        wait $! || true
        kill -0 "${COMMAND_PID}" 2>/dev/null || break
    done
    kill -0 "${COMMAND_PID}" 2>/dev/null || break

    NOW="$(date +%s)"
    LATEST_HEARTBEAT="$(
        find "${HEARTBEAT_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%T@\n' 2>/dev/null \
            | sort -nr \
            | head -n 1 \
            | cut -d. -f1
    )"
    REFERENCE_EPOCH="${LATEST_HEARTBEAT:-${START_EPOCH}}"
    if (( NOW - REFERENCE_EPOCH < STALL_TIMEOUT_SECONDS )); then
        continue
    fi

    TIMEOUT_RECORD="${HEARTBEAT_DIR}/watchdog_timeout.$(date +%Y%m%d-%H%M%S).json"
    printf '{"pid":%s,"timed_out_at":%s,"stall_seconds":%s,"heartbeat_dir":"%s"}\n' \
        "${COMMAND_PID}" "${NOW}" "${STALL_TIMEOUT_SECONDS}" "${HEARTBEAT_DIR}" \
        > "${TIMEOUT_RECORD}"
    echo "error: no evaluation heartbeat for ${STALL_TIMEOUT_SECONDS}s; terminating process group ${COMMAND_PID}." >&2
    find "${HEARTBEAT_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 3 >&2 || true
    terminate_group TERM
    sleep 10 &
    wait $! || true
    if kill -0 "${COMMAND_PID}" 2>/dev/null; then
        terminate_group KILL
    fi
    wait "${COMMAND_PID}" 2>/dev/null || true
    trap - INT TERM EXIT
    exit 120
done

set +e
wait "${COMMAND_PID}"
STATUS=$?
set -e
trap - INT TERM EXIT
exit "${STATUS}"
