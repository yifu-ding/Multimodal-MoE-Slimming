#!/usr/bin/env bash
set -euo pipefail
root=/home/dyf/code/distill/MAES/results/vllm_ours/qwen3-vl-30b-a3b/ep4-p30-full
python - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for task in ('videomme_qwen3_vllm', 'video_mmmu_local'):
    marker = root / 'status' / (task + '.complete')
    if not marker.is_file() or 'signature=' not in marker.read_text():
        sys.exit(1)
    results = list((root / 'tasks' / task).rglob('*_results.json'))
    if not results or not json.loads(max(results, key=lambda p: p.stat().st_mtime).read_text()).get('results'):
        sys.exit(1)
summary = root / 'local_judge' / 'video_mmmu_summary.json'
if not summary.is_file():
    sys.exit(1)
data = json.loads(summary.read_text())
if data.get('num_failed') != 0 or data.get('num_source_samples') != 900 or data.get('num_scored') != 900:
    sys.exit(1)
PY
