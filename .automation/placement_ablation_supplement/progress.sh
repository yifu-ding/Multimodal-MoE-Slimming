#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_ablation_supplement"
python - "${OUT}/pilot.json" "${OUT}/depth_sliding.json" "${OUT}/m_sweep_multimodel.json" <<'PY'
import json
import sys
from pathlib import Path

counts = []
for value, expected in zip(sys.argv[1:], (10, 396, 69)):
    try:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        completed = len(payload.get("records", []))
    except Exception:
        completed = 0
    counts.append((completed, expected))
done = sum(min(value, expected) for value, expected in counts)
total = sum(expected for _, expected in counts)
print(
    f"experiment-c-supplement={done}/{total} ({100.0 * done / total:.1f}%) "
    f"pilot={counts[0][0]}/{counts[0][1]} "
    f"depth={counts[1][0]}/{counts[1][1]} m-sweep={counts[2][0]}/{counts[2][1]}"
)
PY
