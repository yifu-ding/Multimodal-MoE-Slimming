#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dyf/code/distill/MAES"
FLOW="${ROOT}/.automation/placement_e0_followup"
OUT="${ROOT}/results/placement_e0_followup"
PYTHON="/home/dyf/miniconda/envs/vllm-maes/bin/python"
REPORT="${FLOW}/STATUS.md"
CPU_SETS=("0,2,4,6" "8,10,12,14" "16,18,20,22" "24,26,28,30")

mkdir -p "${OUT}/cases" "${OUT}/logs"
cd "${ROOT}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

record() {
    {
        flock 9
        printf '\n- %s CST: %s\n' "$(date '+%F %H:%M')" "$1" >>"${REPORT}"
    } 9>>"${FLOW}/report.lock"
}

children=()
cleanup() {
    local pid
    for pid in "${children[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

record "RUNNING: generating the exact 31-case E0b manifest."
"${PYTHON}" scripts/run_placement_e0_followup.py manifest \
    --output "${OUT}/manifest.json" >>"${OUT}/logs/pipeline.log" 2>&1

record "RUNNING: four CPU-isolated E0b shards started with 300-second HiGHS budgets."
for shard in 0 1 2 3; do
    taskset -c "${CPU_SETS[${shard}]}" "${PYTHON}" scripts/run_placement_e0_followup.py worker \
        --manifest "${OUT}/manifest.json" \
        --output-dir "${OUT}/cases" \
        --shard-id "${shard}" --num-shards 4 \
        --total-time-limit 300 \
        >>"${OUT}/logs/shard-${shard}.log" 2>&1 &
    children+=("$!")
done

failed=0
for pid in "${children[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
children=()
if (( failed != 0 )); then
    record "ATTENTION: at least one E0b shard failed; completed cases remain reusable."
    exit 1
fi

"${PYTHON}" scripts/run_placement_e0_followup.py report \
    --manifest "${OUT}/manifest.json" \
    --output-dir "${OUT}/cases" \
    --output "${OUT}/summary.md" >>"${OUT}/logs/pipeline.log" 2>&1

if ! bash "${FLOW}/completion_check.sh"; then
    record "ATTENTION: E0b workers exited but artifact validation failed."
    exit 2
fi
record "DONE: all 31 E0b cases reached a validated terminal state."
