#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for name in "${CONFIGS[@]}"; do
    IFS='|' read -r scores plan run_dir <<< "$(config_paths "${name}")"
    [[ -s "${scores}" && -s "${plan}" ]] || exit 1
    [[ -f "${run_dir}/status/coco2017_cap_val_local.complete" ]] || exit 1
    [[ -f "${run_dir}/status/mmvet.complete" ]] || exit 1
    summary="${run_dir}/local_judge/mmvet_summary.json"
    [[ -s "${summary}" ]] || exit 1
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["task"] == "mmvet" and d["num_scored"] > 0 and d["num_failed"] == 0' "${summary}" || exit 2
done
exit 0
