#!/usr/bin/env bash
set -euo pipefail
root=/home/dyf/code/distill/MAES/results/vllm_ours/qwen3-vl-30b-a3b/ep4-p30-full
current=''
for task in videomme_qwen3_vllm video_mmmu_local; do
    if [[ -s "$root/status/$task.complete" ]]; then
        printf '%s=complete ' "$task"
    else
        printf '%s=pending ' "$task"
        if [[ -z "$current" ]]; then
            current="$task"
        fi
    fi
done
if [[ -s "$root/local_judge/video_mmmu_summary.json" ]]; then
    python - "$root/local_judge/video_mmmu_summary.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(f"Judge={x.get('num_scored', 0)}/{x.get('num_source_samples', 900)} failed={x.get('num_failed', '?')}")
PY
else
    printf 'Judge=pending'
fi
if [[ -n "$current" ]]; then
    heartbeat="$(find "$root/watchdog/$current" -type f -name response_cache.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    if [[ -n "$heartbeat" && -s "$heartbeat" ]]; then
        python - "$heartbeat" "$current" <<'PY'
import datetime
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
data = json.loads(path.read_text())
completed = int(data.get("completed", 0))
total = int(data.get("total", 0))
percent = completed / max(total, 1)
updated = datetime.datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%F %T %Z")
print(f" | current={task} {completed}/{total} ({percent:.1%}) heartbeat={updated}")
PY
    else
        echo " | current=$current waiting-for-heartbeat"
    fi
else
    echo
fi
