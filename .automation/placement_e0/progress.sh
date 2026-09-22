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
files = sorted((root / "cases").glob("*.json")) if (root / "cases").is_dir() else []
valid = []
for path in files:
    try:
        valid.append(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
arm_a = sum(row.get("time_to_floor_upper_bound_seconds") is not None for row in valid)
arm_b_floor = sum(
    row.get("arm_b", {}).get("arithmetic_optimal") is True for row in valid
)
pipeline = "stopped"
try:
    import subprocess
    active = subprocess.run(
        ["tmux", "has-session", "-t", "=maes-placement-e0"],
        check=False,
        capture_output=True,
        text=True,
    )
    pipeline = "running" if active.returncode == 0 else "stopped"
except Exception:
    pass
print(
    f"placement-e0={len(valid)}/31 ({100.0 * len(valid) / 31:.1f}%) "
    f"remaining={31 - len(valid)} arm-a-floor={arm_a} arm-b-floor={arm_b_floor} "
    f"pipeline={pipeline}"
)
PY
