#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${TARGET_ROOT:-/home/data2/dyf/LongVideoBench}"
HF_HOME="${HF_HOME:-/home/data2/dyf/hf_cache}"
LOG_DIR="${LOG_DIR:-${TARGET_ROOT}/logs}"

mkdir -p "${TARGET_ROOT}" "${LOG_DIR}"

echo "[LongVideoBench] target_root=${TARGET_ROOT}"
echo "[LongVideoBench] hf_home=${HF_HOME}"

hf download longvideobench/LongVideoBench \
  --repo-type dataset \
  --include 'videos.tar.part.*' \
  --local-dir "${TARGET_ROOT}"

echo "[LongVideoBench] download finished"

if [[ ! -f "${TARGET_ROOT}/videos.tar" ]]; then
  echo "[LongVideoBench] joining archive parts"
  cat "${TARGET_ROOT}"/videos.tar.part.* > "${TARGET_ROOT}/videos.tar"
fi

if [[ ! -d "${TARGET_ROOT}/videos" ]]; then
  echo "[LongVideoBench] extracting videos.tar"
  mkdir -p "${TARGET_ROOT}/videos"
  tar -xf "${TARGET_ROOT}/videos.tar" -C "${TARGET_ROOT}"
fi

echo "[LongVideoBench] ready: ${TARGET_ROOT}/videos"
