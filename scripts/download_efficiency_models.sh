#!/usr/bin/env bash
set -euo pipefail
model_key="${1:?mistral or qwen235}"
case "$model_key" in
  mistral) repo=mistralai/Mistral-Small-4-119B-2603 ;;
  qwen235) repo=Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 ;;
  *) exit 2 ;;
esac
dest="/home/data3/dyf/models/${repo##*/}"
mkdir -p "$dest"
exec 9>"$dest/.download.lock"
flock -n 9 || exit 0
exec >>"$dest/download.log" 2>&1
date --iso-8601=seconds
export HF_XET_CACHE=/home/data3/dyf/hf-xet
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export DOWNLOAD_SOURCE=modelscope
echo "download_endpoint=${HF_ENDPOINT}"
echo "weight_download_source=${DOWNLOAD_SOURCE}"
/home/dyf/miniconda/envs/vllm-maes/bin/python \
  /home/dyf/code/distill/MAES/scripts/download_model_mirror.py "$repo" "$dest"
date --iso-8601=seconds
echo DOWNLOAD_COMPLETE
