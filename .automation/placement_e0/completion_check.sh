#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_e0"
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

cases = manifest.get("cases", [])
if len(cases) != 31:
    raise SystemExit(2)
if sum(case.get("kind") == "depth" for case in cases) != 29:
    raise SystemExit(2)
if sum(not bool(case["old_milp"]["solver_optimal"]) for case in cases if case.get("kind") == "depth") != 18:
    raise SystemExit(2)

affinities = ([0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22], [24, 26, 28, 30])
for index, case in enumerate(cases):
    path = root / "cases" / f"{case['case_id']}.json"
    if not path.is_file():
        raise SystemExit(1)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"invalid {path}: {error}", file=sys.stderr)
        raise SystemExit(2)
    if result.get("experiment") != "placement-e0-case":
        raise SystemExit(2)
    if result.get("experiment_sha256") != manifest.get("experiment_sha256"):
        raise SystemExit(2)
    if result.get("arm_a_limits") != [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]:
        raise SystemExit(2)
    if result.get("arm_b_time_limit") != 300.0:
        raise SystemExit(2)
    if result.get("cpu_affinity") != affinities[index % 4]:
        raise SystemExit(2)
    if not result.get("arm_a") or not isinstance(result.get("arm_b"), dict):
        raise SystemExit(2)

for name in ("summary.json", "summary.md", "relabelled_depth_cases.md"):
    if not (root / name).is_file():
        raise SystemExit(1)
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if summary.get("num_cases") != 31:
    raise SystemExit(2)
if summary.get("experiment_sha256") != manifest.get("experiment_sha256"):
    raise SystemExit(2)
PY
