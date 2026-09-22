#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dyf/code/distill/MAES"
FLOW="${ROOT}/.automation/placement_ablation_supplement"
OUT="${ROOT}/results/placement_ablation_supplement"
REPORT="${FLOW}/STATUS.md"
PYTHON="/home/dyf/miniconda/envs/vllm-maes/bin/python"
QWEN_SESSION="maes-qwen-p30-video"

mkdir -p "${OUT}"
cd "${ROOT}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

record() { printf '\n- %s CST: %s\n' "$(date '+%F %H:%M')" "$1" >>"${REPORT}"; }
run_limited() { nice -n 10 taskset -c 24,26,28,30 "${PYTHON}" "$@"; }

record "QUEUED: waiting for ${QWEN_SESSION} to finish successfully."
while tmux has-session -t "${QWEN_SESSION}" 2>/dev/null; do
    sleep 60
done
if ! bash "${ROOT}/.automation/qwen_p30_video/completion_check.sh"; then
    record "ATTENTION: Qwen video pipeline stopped without validated completion; Experiment C was not started."
    exit 2
fi
while ! bash "${FLOW}/resource_check.sh"; do
    sleep 60
done

record "RUNNING: five-case 60s versus 300s MILP pilot."
run_limited scripts/run_placement_ablation_supplement.py pilot \
    --plan runtime/ep4_plans/qwen3-vl-30b-a3b-p30-sparse-tier-v2.pt \
    --cases 0:12 12:12 24:12 0:16 16:16 \
    --time-limits 60 300 \
    --output "${OUT}/pilot.json" >>"${OUT}/pipeline.log" 2>&1

plans=(
    runtime/ep4_plans/qwen3-vl-30b-a3b-p30-sparse-tier-v2.pt
    runtime/ep4_plans/qwen3-vl-30b-a3b-p50-sparse-tier-v2.pt
    runtime/ep4_plans/kimi-p30-sparse-tier-v2.pt
    runtime/ep4_plans/kimi-p50-sparse-tier-v2.pt
    runtime/ep4_plans/internvl3_5-30b-a3b-p30-sparse-tier-v2.pt
    runtime/ep4_plans/internvl3_5-30b-a3b-p50-sparse-tier-v2.pt
)
record "RUNNING: stride-4 sliding depth windows for six plans."
run_limited scripts/run_placement_ablation_supplement.py depth \
    --plans "${plans[@]}" --layers 8 12 16 24 32 48 --stride 4 \
    --greedy-repeats 5 --milp-time-limit 60 --pilot "${OUT}/pilot.json" \
    --output "${OUT}/depth_sliding.json" >>"${OUT}/pipeline.log" 2>&1

p30_plans=(
    runtime/ep4_plans/qwen3-vl-30b-a3b-p30-sparse-tier-v2.pt
    runtime/ep4_plans/kimi-p30-sparse-tier-v2.pt
    runtime/ep4_plans/internvl3_5-30b-a3b-p30-sparse-tier-v2.pt
)
record "RUNNING: p=0.3 multi-model m sweep and greedy-neighborhood comparison."
run_limited scripts/run_placement_ablation_supplement.py m-sweep \
    --plans "${p30_plans[@]}" --m-values 4 5 6 7 8 9 10 11 12 16 \
    --full-bijection-m-values 5 6 7 --greedy-repeats 5 --milp-time-limit 300 \
    --output "${OUT}/m_sweep_multimodel.json" >>"${OUT}/pipeline.log" 2>&1

run_limited scripts/run_placement_ablation_supplement.py report \
    --pilot "${OUT}/pilot.json" --depth "${OUT}/depth_sliding.json" \
    --m-sweep "${OUT}/m_sweep_multimodel.json" --output "${OUT}/summary.md"

if ! bash "${FLOW}/completion_check.sh"; then
    record "ATTENTION: supplement finished but artifact validation failed."
    exit 1
fi
record "DONE: Experiment C supplement completed and validated; see results/placement_ablation_supplement/summary.md."
