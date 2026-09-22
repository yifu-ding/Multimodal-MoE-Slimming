#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_grid"
PYTHON="/home/dyf/miniconda/envs/vllm-maes/bin/python"

"${PYTHON}" - "${OUT}" <<'PY'
import csv
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

if manifest.get("num_cases") != 144:
    raise SystemExit(2)
if manifest.get("m_values") != [4, 6, 8, 12, 16, 24, 32, 48, 64]:
    raise SystemExit(2)
if manifest.get("layers") != [4, 8, 12, 16, 24, 32, 40, 48]:
    raise SystemExit(2)
if manifest.get("prune_ratios") != [0.3, 0.5]:
    raise SystemExit(2)

affinities = ([0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22], [24, 26, 28, 30])
seen = set()
for case in manifest.get("cases", []):
    key = (case["prune_ratio"], case["layers"], case["m"])
    if key in seen:
        raise SystemExit(2)
    seen.add(key)
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
    shard = int(case["shard_id"])
    if result.get("worker_cpu_affinity") != affinities[shard]:
        raise SystemExit(2)
    ladder = result.get("ladder", {})
    if ladder.get("stopped_reason") not in {"feasible", "budget_or_unproven"}:
        raise SystemExit(2)
    if ladder.get("found_feasible") and not ladder.get("attempts"):
        raise SystemExit(2)
    if result.get("source_kind") == "reused" and not ladder.get("optimality_proven"):
        raise SystemExit(2)

if len(seen) != 144:
    raise SystemExit(2)
for name in ("summary.json", "summary.md", "grid.csv", "heatmap-p30.png", "heatmap-p50.png"):
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(1)
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if summary.get("num_cases") != 144:
    raise SystemExit(2)
if summary.get("experiment_sha256") != manifest.get("experiment_sha256"):
    raise SystemExit(2)
with (root / "grid.csv").open(encoding="utf-8", newline="") as handle:
    if sum(1 for _ in csv.DictReader(handle)) != 144:
        raise SystemExit(2)
PY
