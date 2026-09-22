#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_ablation_supplement"
python - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
specs = (
    (root / "pilot.json", "supplement-pilot", 10),
    (root / "depth_sliding.json", "supplement-depth", 396),
    (root / "m_sweep_multimodel.json", "supplement-m-sweep", 69),
)
for path, experiment, expected in specs:
    if not path.is_file():
        raise SystemExit(1)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"invalid {path}: {error}", file=sys.stderr)
        raise SystemExit(2)
    if payload.get("experiment") != experiment or len(payload.get("records", [])) != expected:
        raise SystemExit(1)
    if experiment == "supplement-pilot" and not payload.get("validation", {}).get("passed"):
        print("pilot validation did not pass", file=sys.stderr)
        raise SystemExit(2)
if not (root / "summary.md").is_file():
    raise SystemExit(1)
PY
