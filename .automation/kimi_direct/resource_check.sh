#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import subprocess,sys; s=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True); a=[int(x.strip()) for x in s.splitlines()]; sys.exit(0 if len(a)>=4 and all(x<1024 for x in a[:4]) else 1)'
