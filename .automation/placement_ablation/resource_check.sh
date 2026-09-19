#!/usr/bin/env bash
set -u

if [[ "${ALLOW_CONCURRENT_OURS:-0}" != "1" ]]; then
    if tmux has-session -t maes-todo-ours 2>/dev/null; then
        exit 1
    fi
    if pgrep -u "$(id -u)" -f 'python -m lmms_eval' >/dev/null; then
        exit 1
    fi
fi
load_one="$(cut -d' ' -f1 /proc/loadavg)"
awk -v load_avg="${load_one}" -v limit="${PLACEMENT_MAX_LOAD:-8.0}" \
    'BEGIN { exit !(load_avg < limit) }'
