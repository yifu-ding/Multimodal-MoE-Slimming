#!/usr/bin/env bash
set -euo pipefail

# Download the model series currently supported by this repository.
# Defaults:
#   - moonshotai/Kimi-VL-A3B-Instruct
#   - Qwen/Qwen3-VL-30B-A3B-Instruct
#   - OpenGVLab/InternVL-3.5-GPT-OSS-20B-A4B-Preview-HF
#
# Usage:
#   conda activate modes
#   hf auth login
#   bash scripts/download_models.sh
#
# Optional overrides:
#   MODEL_ROOT=/path/to/storage/models bash scripts/download_models.sh
#   EXTRA_MODELS="Qwen/Qwen3-VL-4B-Instruct" bash scripts/download_models.sh

if ! command -v hf >/dev/null 2>&1; then
    echo "error: 'hf' command not found. Activate the 'modes' environment first." >&2
    exit 1
fi

MODEL_ROOT="${MODEL_ROOT:-$(pwd)/storage/models}"
mkdir -p "${MODEL_ROOT}"

MODELS=(
    "moonshotai/Kimi-VL-A3B-Instruct"
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
    "OpenGVLab/InternVL-3.5-GPT-OSS-20B-A4B-Preview-HF"
)

if [[ -n "${EXTRA_MODELS:-}" ]]; then
    while IFS= read -r model_id; do
        [[ -n "${model_id}" ]] && MODELS+=("${model_id}")
    done < <(printf '%s\n' "${EXTRA_MODELS}")
fi

for model_id in "${MODELS[@]}"; do
    model_name="${model_id##*/}"
    target_dir="${MODEL_ROOT}/${model_name}"
    echo "==> Downloading ${model_id}"
    mkdir -p "${target_dir}"
    hf download "${model_id}" --local-dir "${target_dir}"
done

cat <<EOF

Downloads finished.
Models are under:
  ${MODEL_ROOT}

Note:
  This repository currently has runnable code for Kimi-VL and Qwen3-VL.
  The InternVL model above is included for download convenience, but this repo
  does not currently provide an InternVL evaluation/calibration entrypoint.
EOF
