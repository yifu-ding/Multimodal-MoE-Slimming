#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

coco=0
mmvet_predict=0
mmvet_judge=0
plans=0
current=none
for name in "${CONFIGS[@]}"; do
    IFS='|' read -r scores plan run_dir <<< "$(config_paths "${name}")"
    if [[ -s "${plan}" ]]; then
        plans=$((plans + 1))
    elif [[ "${current}" == none ]]; then
        current="${name}/plan"
    fi
    if [[ -f "${run_dir}/status/coco2017_cap_val_local.complete" ]]; then
        coco=$((coco + 1))
    elif [[ "${current}" == none ]]; then
        current="${name}/coco"
    fi
    if [[ -f "${run_dir}/status/mmvet.complete" ]]; then
        mmvet_predict=$((mmvet_predict + 1))
    elif [[ "${current}" == none ]]; then
        current="${name}/mmvet-predict"
    fi
    if [[ -s "${run_dir}/local_judge/mmvet_summary.json" ]]; then
        mmvet_judge=$((mmvet_judge + 1))
    elif [[ "${current}" == none ]]; then
        current="${name}/mmvet-judge"
    fi
done
printf 'acp-align128 plans=%d/6 coco=%d/6 mmvet_predict=%d/6 mmvet_judge=%d/6 current=%s\n' \
    "${plans}" "${coco}" "${mmvet_predict}" "${mmvet_judge}" "${current}"
