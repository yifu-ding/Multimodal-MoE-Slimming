#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
CPU_AFFINITY="0-23,25,27,29,31"
LOG="${REPO_ROOT}/results/mmvet_recovery.log"
KIMI_RUN="${REPO_ROOT}/results/vllm_ours/kimi/ep4-p50-full"
INTERNVL_RUN="${REPO_ROOT}/results/vllm_ours/internvl3_5-30b-a3b/ep4-p50-full"
SMOKE_RUN="${REPO_ROOT}/artifacts/mmvet-recovery/internvl-p50-smoke"
PLAN="${REPO_ROOT}/runtime/ep4_plans/internvl3_5-30b-a3b-p50-sparse-tier-v2.pt"

exec > >(tee -a "${LOG}") 2>&1
cd "${REPO_ROOT}"

judge_complete() {
    local summary="$1"
    python - "${summary}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if (
    data.get("num_failed") == 0
    and data.get("num_source_samples", 0) > 0
    and data.get("num_scored") == data.get("num_source_samples")
) else 1)
PY
}

echo "[$(date '+%F %T')] Kimi MMVet Judge recovery"
if ! judge_complete "${KIMI_RUN}/local_judge/mmvet_summary.json"; then
    taskset -c "${CPU_AFFINITY}" env \
        PREDICTIONS_DIR="${KIMI_RUN}" TASKS=mmvet JUDGE_MAX_ATTEMPTS=2 \
        PORT=8010 CUDA_VISIBLE_DEVICES=0 JUDGE_WORKERS=1 \
        bash scripts/run_vllm_judge_stage.sh
fi
judge_complete "${KIMI_RUN}/local_judge/mmvet_summary.json"

echo "[$(date '+%F %T')] InternVL MMVet smoke"
if [[ ! -s "${SMOKE_RUN}/status/mmvet.complete" ]]; then
    mkdir -p "${SMOKE_RUN}"
    taskset -c "${CPU_AFFINITY}" env \
        DECORD_EOF_RETRY_MAX=20480 CUDA_VISIBLE_DEVICES=0,1,2,3 \
        GPU_MEMORY_UTILIZATION=0.85 FORCE=1 \
        EP4_PLAN="${PLAN}" MODEL=OpenGVLab/InternVL3_5-30B-A3B-HF \
        PRUNING_LABEL=ours_p50 TASKS=mmvet LIMIT=1 FAIL_FAST=1 \
        RUN_DIR="${SMOKE_RUN}" \
        bash scripts/run_vllm_ep4_pruned.sh
fi
test -s "${SMOKE_RUN}/status/mmvet.complete"
find "${SMOKE_RUN}/tasks/mmvet" -type f -name '*_samples_mmvet.jsonl' -size +0c -print -quit | grep -q .
grep -RqsE '\[MAES EP4\].*layer=' "${SMOKE_RUN}/logs"

echo "[$(date '+%F %T')] InternVL MMVet full recovery"
if [[ ! -s "${INTERNVL_RUN}/status/mmvet.complete" ]]; then
    taskset -c "${CPU_AFFINITY}" env \
        DECORD_EOF_RETRY_MAX=20480 CUDA_VISIBLE_DEVICES=0,1,2,3 \
        GPU_MEMORY_UTILIZATION=0.85 \
        EP4_PLAN="${PLAN}" MODEL=OpenGVLab/InternVL3_5-30B-A3B-HF \
        PRUNING_LABEL=ours_p50 TASKS=mmvet FAIL_FAST=1 \
        RUN_DIR="${INTERNVL_RUN}" \
        bash scripts/run_vllm_ep4_pruned.sh
fi
test -s "${INTERNVL_RUN}/status/mmvet.complete"
find "${INTERNVL_RUN}/tasks/mmvet" -type f -name '*_samples_mmvet.jsonl' -size +0c -print -quit | grep -q .

echo "[$(date '+%F %T')] InternVL MMVet Judge recovery"
if ! judge_complete "${INTERNVL_RUN}/local_judge/mmvet_summary.json" 2>/dev/null; then
    taskset -c "${CPU_AFFINITY}" env \
        PREDICTIONS_DIR="${INTERNVL_RUN}" TASKS=mmvet JUDGE_MAX_ATTEMPTS=2 \
        PORT=8010 CUDA_VISIBLE_DEVICES=0 JUDGE_WORKERS=1 \
        bash scripts/run_vllm_judge_stage.sh
fi
judge_complete "${INTERNVL_RUN}/local_judge/mmvet_summary.json"

python scripts/ours_campaign_state.py summary
echo "[$(date '+%F %T')] MMVet recovery complete"
