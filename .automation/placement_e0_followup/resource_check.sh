#!/usr/bin/env bash
set -u

expected="0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"
actual="$(lscpu -p=CPU,CORE,ONLINE | awk -F, '!/^#/ && $3 == "Y" && $1 % 2 == 0 {print $1}' | tr '\n' ' ' | sed 's/ $//')"
if [[ "${actual}" != "${expected}" ]]; then
    printf 'unexpected CPU topology: %s\n' "${actual}" >&2
    exit 2
fi
if pgrep -u "$(id -u)" -f 'scripts/run_placement_e0_followup.py worker' >/dev/null; then
    exit 1
fi
exit 0
