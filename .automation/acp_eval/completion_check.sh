#!/usr/bin/env bash
# 0 = all 21 (3 models x 7 tasks) authorized runs are complete.
# 1 = work remains.
# 2 = invalid/unexpected state (should not normally happen here).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

total_done=0
total_expected=$(( ${#TASK_LIST[@]} * ${#MODEL_NAMES[@]} ))
for name in "${MODEL_NAMES[@]}"; do
    IFS='|' read -r model_id plan run_dir <<< "$(model_spec "${name}")"
    if [[ ! -f "${plan}" ]]; then
        echo "invalid: missing EP4 plan for ${name}: ${plan}" >&2
        exit 2
    fi
    total_done=$((total_done + $(count_complete "${name}" "${run_dir}")))
done

if (( total_done == total_expected )); then
    exit 0
fi
exit 1
