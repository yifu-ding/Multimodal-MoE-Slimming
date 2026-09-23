#!/usr/bin/env bash
# Prints one concise line summarizing completed/total tasks across all 3 models.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

total=0
line=""
for name in "${MODEL_NAMES[@]}"; do
    IFS='|' read -r model_id plan run_dir <<< "$(model_spec "${name}")"
    done_count="$(count_complete "${run_dir}")"
    total=$((total + done_count))
    line+="${name}=${done_count}/${#TASK_LIST[@]} "
done

printf 'acp p30 padded: %s(total %d/%d)\n' "${line}" "${total}" "$(( ${#TASK_LIST[@]} * ${#MODEL_NAMES[@]} ))"
