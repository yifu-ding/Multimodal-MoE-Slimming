#!/usr/bin/env bash
set -u

ROOT="/home/dyf/code/distill/MAES"
TABLE="${ROOT}/results/placement_ablation/table6.json"
SWEEP="${ROOT}/results/placement_ablation/m_sweep.json"

python - "${TABLE}" "${SWEEP}" <<'PY'
import json
import sys
from pathlib import Path

expected = ((Path(sys.argv[1]), "table6", 18), (Path(sys.argv[2]), "m-sweep", 21))
missing = False
for path, experiment, count in expected:
    if not path.exists():
        missing = True
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"invalid {path}: {error}", file=sys.stderr)
        raise SystemExit(2)
    if payload.get("experiment") != experiment or not isinstance(payload.get("records"), list):
        print(f"invalid experiment payload: {path}", file=sys.stderr)
        raise SystemExit(2)
    if len(payload["records"]) < count:
        missing = True
    for record in payload["records"]:
        if "method" not in record or "m" not in record:
            print(f"invalid record in {path}", file=sys.stderr)
            raise SystemExit(2)
    for suffix in (".csv", ".md"):
        if not path.with_suffix(suffix).is_file():
            missing = True
raise SystemExit(1 if missing else 0)
PY
