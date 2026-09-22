#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_e0_followup"
PYTHON="/home/dyf/miniconda/envs/vllm-maes/bin/python"

"${PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(1)
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as error:
    print(f"invalid manifest: {error}", file=sys.stderr)
    raise SystemExit(2)
if manifest.get("group_counts") != {"retry-l4": 24, "m4-floor0": 5, "ep-scale": 2}:
    raise SystemExit(2)

affinities = ([0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22], [24, 26, 28, 30])
for index, case in enumerate(manifest.get("cases", [])):
    path = root / "cases" / f"{case['case_id']}.json"
    if not path.is_file():
        raise SystemExit(1)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"invalid {path}: {error}", file=sys.stderr)
        raise SystemExit(2)
    if result.get("experiment_sha256") != manifest.get("experiment_sha256"):
        raise SystemExit(2)
    if result.get("total_time_limit") != 300.0:
        raise SystemExit(2)
    if result.get("cpu_affinity") != affinities[index % 4]:
        raise SystemExit(2)
    ladder = result.get("ladder", {})
    if not ladder.get("attempts"):
        raise SystemExit(2)
    if case["group"] == "retry-l4":
        if ladder.get("retry_count", 0) <= 0:
            raise SystemExit(2)
        if not ladder.get("optimality_proven"):
            raise SystemExit(2)
        if ladder.get("spread") != case.get("expected_optimal_spread"):
            raise SystemExit(2)
    if case["group"] == "m4-floor0":
        if ladder.get("spread") != 0.0 or not ladder.get("optimality_proven"):
            raise SystemExit(2)

if len(manifest.get("cases", [])) != 31:
    raise SystemExit(2)
for name in ("summary.json", "summary.md"):
    if not (root / name).is_file():
        raise SystemExit(1)
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if summary.get("num_cases") != 31:
    raise SystemExit(2)
if summary.get("experiment_sha256") != manifest.get("experiment_sha256"):
    raise SystemExit(2)
PY
