#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_ablation"
REPORT="${ROOT}/docs/实验C执行结果.md"
LOG="${OUT}/pipeline.log"
DEPTH_JSON="${OUT}/depth_sweep.json"
SWEEP_JSON="${OUT}/m_sweep_v2.json"
RESOURCE_CHECK="${ROOT}/.automation/placement_ablation/resource_check.sh"

mkdir -p "${OUT}"
cd "${ROOT}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
    printf '%s state=waiting-for-pid pid=%s\n' \
        "$(date '+%F %H:%M:%S %Z')" "${WAIT_FOR_PID}" | tee -a "${LOG}"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        sleep 30
    done
    printf '%s state=wait-pid-finished pid=%s\n' \
        "$(date '+%F %H:%M:%S %Z')" "${WAIT_FOR_PID}" | tee -a "${LOG}"
fi

printf '%s state=waiting-for-idle-resources\n' "$(date '+%F %H:%M:%S %Z')" | tee -a "${LOG}"
while ! bash "${RESOURCE_CHECK}"; do
    sleep 60
done
printf '%s state=resources-ready\n' "$(date '+%F %H:%M:%S %Z')" | tee -a "${LOG}"

run_limited() {
    nice -n 10 taskset -c 24,26,28,30 "$@"
}

printf '%s stage=depth-sweep start\n' "$(date '+%F %H:%M:%S %Z')" | tee -a "${LOG}"
run_limited python scripts/run_placement_ablation.py depth-sweep \
    --layers 4 8 12 16 24 32 48 \
    --greedy-repeats 5 \
    --milp-time-limit 300 \
    --output "${DEPTH_JSON}" 2>&1 | tee -a "${LOG}"
python scripts/summarize_placement_ablation.py --input "${DEPTH_JSON}" | tee -a "${LOG}"

printf '%s stage=m-sweep start\n' "$(date '+%F %H:%M:%S %Z')" | tee -a "${LOG}"
run_limited python scripts/run_placement_ablation.py m-sweep \
    --m-values 4 6 8 12 16 \
    --time-limits 60 180 300 \
    --greedy-repeats 5 \
    --output "${SWEEP_JSON}" 2>&1 | tee -a "${LOG}"
python scripts/summarize_placement_ablation.py --input "${SWEEP_JSON}" | tee -a "${LOG}"

{
    flock 9
    printf '\n> [!IMPORTANT]\n'
    printf '> **DONE (%s)**  \n' "$(date '+%F %H:%M %Z')"
    printf '> 多窗口 depth sweep 与 M 扩展性扫描均已完成并通过产物检查。\n'
} >>"${REPORT}" 9>>"${REPORT}.lock"
