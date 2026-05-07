#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/home/data2/dyf/Video-MME}"
BASE_URL="https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main"

mkdir -p "${TARGET_DIR}"

files=(
  videos_chunked_01.zip
  videos_chunked_11.zip
  videos_chunked_12.zip
  videos_chunked_13.zip
  videos_chunked_14.zip
  videos_chunked_15.zip
  videos_chunked_16.zip
  videos_chunked_17.zip
  videos_chunked_18.zip
  videos_chunked_19.zip
  videos_chunked_20.zip
)

for name in "${files[@]}"; do
  path="${TARGET_DIR}/${name}"
  echo "[$(date -Iseconds)] downloading ${name} -> ${path}"
  curl -L -C - -o "${path}" "${BASE_URL}/${name}"
  echo "[$(date -Iseconds)] finished ${name}"
done
