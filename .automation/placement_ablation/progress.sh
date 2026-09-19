#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
python - "${ROOT}/results/placement_ablation/depth_sweep.json" "${ROOT}/results/placement_ablation/m_sweep_v2.json" <<'PY'
import json
import sys
from pathlib import Path

counts = []
for path_value, expected in zip(sys.argv[1:], (174, 21)):
    path = Path(path_value)
    try:
        completed = len(json.loads(path.read_text(encoding="utf-8"))["records"])
    except Exception:
        completed = 0
    counts.append((completed, expected))
done = sum(value for value, _ in counts)
total = sum(value for _, value in counts)
percent = 100.0 * done / total
print(
    f"stage=placement-ablation completed={done}/{total} ({percent:.1f}%) "
    f"depth-sweep={counts[0][0]}/{counts[0][1]} "
    f"m-sweep={counts[1][0]}/{counts[1][1]}"
)
PY
