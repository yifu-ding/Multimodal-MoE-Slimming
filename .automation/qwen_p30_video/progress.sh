#!/usr/bin/env bash
set -euo pipefail
root=/home/dyf/code/distill/MAES/results/vllm_ours/qwen3-vl-30b-a3b/ep4-p30-full
for task in videomme_qwen3_vllm video_mmmu_local; do
    if [[ -s "$root/status/$task.complete" ]]; then
        printf '%s=complete ' "$task"
    else
        printf '%s=pending ' "$task"
    fi
done
if [[ -s "$root/local_judge/video_mmmu_summary.json" ]]; then
    python - "$root/local_judge/video_mmmu_summary.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(f"Judge={x.get('num_scored', 0)}/{x.get('num_source_samples', 900)} failed={x.get('num_failed', '?')}")
PY
else
    echo 'Judge=pending'
fi
