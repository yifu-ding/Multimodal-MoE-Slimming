#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_grid"
PYTHON="/home/dyf/miniconda/envs/vllm-maes/bin/python"

"${PYTHON}" - "${OUT}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted((root / "cases").glob("*.json")) if (root / "cases").is_dir() else []
rows = []
for path in files:
    try:
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
proven = sum(row.get("ladder", {}).get("optimality_proven") is True for row in rows)
reused = sum(row.get("source_kind") == "reused" for row in rows)
unproven = sum(
    row.get("ladder", {}).get("stopped_reason") == "budget_or_unproven" for row in rows
)
active = subprocess.run(
    ["tmux", "has-session", "-t", "=maes-placement-grid"],
    check=False,
    capture_output=True,
    text=True,
).returncode == 0
print(
    f"placement-grid={len(rows)}/144 ({100.0 * len(rows) / 144:.1f}%) "
    f"proven={proven} reused={reused} unproven={unproven} "
    f"pipeline={'running' if active else 'stopped'}"
)
PY
