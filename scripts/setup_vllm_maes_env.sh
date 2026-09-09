#!/usr/bin/env bash
set -euo pipefail

# Clone the existing MAES environment, then let pinned vLLM install its exact
# PyTorch/CUDA Python-package requirements. This script does not download any
# model or benchmark data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ENV="${SOURCE_ENV:-maes}"
TARGET_ENV="${TARGET_ENV:-vllm-maes}"
VLLM_VERSION="${VLLM_VERSION:-0.11.2}"

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is not available in PATH." >&2
    exit 2
fi
if conda env list | awk -v target="${TARGET_ENV}" '$1 == target { found=1 } END { exit(found ? 0 : 1) }'; then
    echo "error: target environment '${TARGET_ENV}' already exists." >&2
    echo "       Choose TARGET_ENV=... or remove it explicitly before retrying." >&2
    exit 2
fi
if ! conda env list | awk -v target="${SOURCE_ENV}" '$1 == target { found=1 } END { exit(found ? 0 : 1) }'; then
    echo "error: source environment '${SOURCE_ENV}' does not exist." >&2
    exit 2
fi

echo "Cloning conda environment: ${SOURCE_ENV} -> ${TARGET_ENV}"
conda create --yes --name "${TARGET_ENV}" --clone "${SOURCE_ENV}"

echo "Installing vllm==${VLLM_VERSION}; this will replace cloned torch packages with vLLM-compatible versions."
conda install --yes --name "${TARGET_ENV}" "openjdk=17"
conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install --upgrade pip setuptools wheel
conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install "vllm==${VLLM_VERSION}"

# Use the checked-out evaluator and task definitions rather than fetching a
# second lmms-eval copy from the network.
conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install --no-deps --editable "${REPO_ROOT}/lmms-eval"
conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install "qwen-vl-utils>=0.0.14" "decord>=0.6.0" requests openpyxl pycocoevalcap

echo "Verifying imports and CUDA visibility."
conda run --no-capture-output -n "${TARGET_ENV}" python - <<'PY'
import platform

import lmms_eval
import torch
import transformers
import vllm
from vllm.model_executor.models import ModelRegistry

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"vllm={vllm.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_gpu_count={torch.cuda.device_count()}")
architecture = "Qwen3VLMoeForConditionalGeneration"
print(f"qwen3_vl_moe_supported={architecture in ModelRegistry.get_supported_archs()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the new environment")
if torch.cuda.device_count() < 4:
    raise SystemExit("fewer than four GPUs are visible")
if architecture not in ModelRegistry.get_supported_archs():
    raise SystemExit(f"vLLM does not register {architecture}")
PY

echo "Environment '${TARGET_ENV}' is ready. No model or dataset was downloaded."
