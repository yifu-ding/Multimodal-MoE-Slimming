#!/usr/bin/env bash
set -euo pipefail
mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
[[ "${#rows[@]}" -eq 4 ]] || exit 1
for expected in 0 1 2 3; do
    IFS=',' read -r index used <<< "${rows[expected]}"
    index="${index// /}"; used="${used// /}"
    [[ "${index}" == "${expected}" && "${used}" -le 64 ]] || exit 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
