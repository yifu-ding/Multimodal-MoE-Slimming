#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import subprocess, sys
rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
sys.exit(0 if len(rows) >= 4 and all(int(x.strip()) < 2000 for x in rows[:4]) else 1)
PY
