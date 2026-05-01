#!/usr/bin/env bash
# Detect GPUs via nvidia-smi, print per-GPU VRAM usage, pick the GPU with
# the *smallest memory.used* (most free for new jobs), and set CUDA_VISIBLE_DEVICES.
#
# Usage
# -----
#   source scripts/select_least_used_gpu.sh
#   # or
#   . scripts/select_least_used_gpu.sh
#
# Direct execution also exports in that subshell and prints the line to copy:
#   bash scripts/select_least_used_gpu.sh
#
# If CUDA_VISIBLE_DEVICES is already non-empty in the environment, this script
# prints a warning and does nothing (does not override your setting).

set -euo pipefail

if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
  echo "select_least_used_gpu: warning: CUDA_VISIBLE_DEVICES is already set to '${CUDA_VISIBLE_DEVICES}'; leaving it unchanged (skip auto pick)." >&2
  return 0 2>/dev/null || exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "select_least_used_gpu: nvidia-smi not found" >&2
  return 2 2>/dev/null || exit 2
fi

mapfile -t _rows < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
if [[ ${#_rows[@]} -eq 0 ]]; then
  echo "select_least_used_gpu: no GPU rows from nvidia-smi" >&2
  return 1 2>/dev/null || exit 1
fi

_min_used=2147483647
_best_idx=""

for _line in "${_rows[@]}"; do
  # CSV: "0, 1234, 24576" — strip spaces
  _idx="${_line%%,*}"
  _rest="${_line#*,}"
  _used="${_rest%%,*}"
  _total="${_rest#*,}"
  _idx="${_idx// /}"
  _used="${_used// /}"
  _total="${_total// /}"
  echo "GPU ${_idx}: memory.used ${_used} MiB / ${_total} MiB"
  if [[ -z "${_best_idx}" ]] || (( _used < _min_used )); then
    _min_used="${_used}"
    _best_idx="${_idx}"
  fi
done

export CUDA_VISIBLE_DEVICES="${_best_idx}"
echo "Have set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (least memory.used: ${_min_used} MiB), you dont have to export it again"

# When sourced, export stays in current shell; when executed, only this subshell has it.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Note: run \`source ${BASH_SOURCE[0]}\` to export in your current shell."
fi
