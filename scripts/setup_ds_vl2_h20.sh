#!/usr/bin/env bash
set -euo pipefail

# Clone an existing ds-vl2 environment and upgrade only the CUDA/PyTorch stack
# needed for H20 / sm_90, while keeping the original environment untouched.

SRC_ENV="${SRC_ENV:-ds-vl2}"
TARGET_ENV="${TARGET_ENV:-ds-vl2-h20}"
CONDA_ROOT="${CONDA_ROOT:-/home/data2/dyf/miniconda}"
TARGET_PYTHON="${CONDA_ROOT}/envs/${TARGET_ENV}/bin/python"

echo "Source env : ${SRC_ENV}"
echo "Target env : ${TARGET_ENV}"
echo "Conda root : ${CONDA_ROOT}"
echo ""

if [[ ! -d "${CONDA_ROOT}" ]]; then
    echo "Conda root not found: ${CONDA_ROOT}" >&2
    exit 1
fi

echo "[1/4] Cloning ${SRC_ENV} -> ${TARGET_ENV}"
CONDA_NO_PLUGINS=true conda create -n "${TARGET_ENV}" --clone "${SRC_ENV}" -y

echo "[2/4] Installing H20-compatible PyTorch CUDA 12.4 stack"
CONDA_NO_PLUGINS=true conda install --solver classic -n "${TARGET_ENV}" -y \
    pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 \
    -c pytorch -c nvidia

echo "[3/4] Removing stale CUDA 11 NCCL runtime if present"
"${TARGET_PYTHON}" -m pip uninstall -y nvidia-nccl-cu11 || true

echo "[4/4] Reinstalling CUDA 12 NCCL runtime"
"${TARGET_PYTHON}" -m pip install --force-reinstall --no-cache-dir nvidia-nccl-cu12==2.21.5

echo ""
echo "Validation"
"${TARGET_PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
"${TARGET_PYTHON}" -c "import deepseek_vl2, deepseek_vl2.models; print('deepseek ok')"

echo ""
echo "Done. Activate with:"
echo "  conda activate ${TARGET_ENV}"
