#!/usr/bin/env bash
set -uo pipefail

mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
(( ${#rows[@]} == 4 )) || exit 1
for row in "${rows[@]}"; do
    used="${row#*,}"
    used="${used// /}"
    (( used <= 64 )) || exit 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)" ]] || exit 1
pgrep -f 'run_.*vllm.*\.sh|lmms_eval|vllm serve|VLLM::EngineCore' >/dev/null 2>&1 && exit 1
exit 0
