#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
OUT="${ROOT}/results/placement_e0_followup"
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
retry = sum(row.get("ladder", {}).get("retry_count", 0) > 0 for row in rows)
proven = sum(row.get("ladder", {}).get("optimality_proven") is True for row in rows)
active = subprocess.run(
    ["tmux", "has-session", "-t", "=maes-placement-e0b"],
    check=False,
    capture_output=True,
    text=True,
).returncode == 0
print(
    f"placement-e0b={len(rows)}/31 ({100.0 * len(rows) / 31:.1f}%) "
    f"retry-branch={retry} proven={proven} pipeline={'running' if active else 'stopped'}"
)
PY
