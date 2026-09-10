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
RESUME_EXISTING="${RESUME_EXISTING:-0}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
OPENJDK_PACKAGE_URL="${OPENJDK_PACKAGE_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/openjdk-17.0.14-h39bb4c0_2.conda}"

pip_install() {
    conda run --no-capture-output -n "${TARGET_ENV}" \
        python -m pip install --index-url "${PYPI_INDEX_URL}" \
        --retries 10 --timeout 120 "$@"
}

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is not available in PATH." >&2
    exit 2
fi
if conda env list | awk -v target="${TARGET_ENV}" '$1 == target { found=1 } END { exit(found ? 0 : 1) }'; then
    if [[ "${RESUME_EXISTING}" != "1" ]]; then
        echo "error: target environment '${TARGET_ENV}' already exists." >&2
        echo "       Set RESUME_EXISTING=1 to continue an interrupted setup," >&2
        echo "       choose TARGET_ENV=..., or remove it explicitly before retrying." >&2
        exit 2
    fi
    echo "Resuming setup in existing environment '${TARGET_ENV}'."
else
    if ! conda env list | awk -v target="${SOURCE_ENV}" '$1 == target { found=1 } END { exit(found ? 0 : 1) }'; then
        echo "error: source environment '${SOURCE_ENV}' does not exist." >&2
        exit 2
    fi

    echo "Cloning conda environment: ${SOURCE_ENV} -> ${TARGET_ENV}"
    conda create --yes --name "${TARGET_ENV}" --clone "${SOURCE_ENV}"
fi

echo "Installing vllm==${VLLM_VERSION}; this will replace cloned torch packages with vLLM-compatible versions."
if ! conda run -n "${TARGET_ENV}" java -version >/dev/null 2>&1; then
    # This build is compatible with the lcms2/xorg versions cloned from maes.
    conda install --yes --name "${TARGET_ENV}" "${OPENJDK_PACKAGE_URL}"
fi
pip_install --upgrade pip setuptools wheel
pip_install "vllm==${VLLM_VERSION}"

# A compiled flash-attn inherited from the source environment may target the
# old Torch ABI. vLLM uses its bundled backend when that optional import fails.
if conda run -n "${TARGET_ENV}" python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("flash_attn") else 1)' \
    && ! conda run -n "${TARGET_ENV}" python -c 'import flash_attn' >/dev/null 2>&1; then
    echo "Removing flash-attn inherited with an incompatible Torch ABI."
    conda run --no-capture-output -n "${TARGET_ENV}" \
        python -m pip uninstall --yes flash-attn
fi

# Use the checked-out evaluator and task definitions rather than fetching a
# second lmms-eval copy from the network.
pip_install --no-deps --editable "${REPO_ROOT}/lmms-eval"
pip_install "qwen-vl-utils>=0.0.14" "decord>=0.6.0" requests openpyxl pycocoevalcap

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
