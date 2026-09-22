#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dyf/code/distill/MAES"
FLOW="${ROOT}/.automation/placement_grid"
OUT="${ROOT}/results/placement_grid"
PYTHON="/home/dyf/miniconda/envs/vllm-maes/bin/python"
REPORT="${FLOW}/STATUS.md"
CPU_SETS=("0,2,4,6" "8,10,12,14" "16,18,20,22" "24,26,28,30")

mkdir -p "${OUT}/cases" "${OUT}/logs" "${OUT}/mplconfig"
cd "${ROOT}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR="${OUT}/mplconfig"

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

if ! bash "${FLOW}/resource_check.sh"; then
    record "WAITING: required CPU sets are unavailable or another placement worker is active."
    exit 75
fi

record "RUNNING: generating the 144-cell grid manifest and matching reusable E0/E0b artifacts."
"${PYTHON}" scripts/run_placement_grid.py manifest \
    --output "${OUT}/manifest.json" >>"${OUT}/logs/pipeline.log" 2>&1

record "RUNNING: four balanced shards started on disjoint four-core affinity sets."
for shard in 0 1 2 3; do
    taskset -c "${CPU_SETS[${shard}]}" "${PYTHON}" scripts/run_placement_grid.py worker \
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
    record "ATTENTION: at least one grid shard failed; validated cells remain reusable."
    exit 1
fi

"${PYTHON}" scripts/run_placement_grid.py report \
    --manifest "${OUT}/manifest.json" \
    --output-dir "${OUT}/cases" \
    --output "${OUT}/summary.md" >>"${OUT}/logs/pipeline.log" 2>&1

if ! bash "${FLOW}/completion_check.sh"; then
    record "ATTENTION: grid workers exited but artifact validation failed."
    exit 2
fi
record "DONE: all 144 cells reached a validated terminal state; see results/placement_grid/summary.md."
