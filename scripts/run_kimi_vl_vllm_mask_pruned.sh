#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MASK_PLAN="${MASK_PLAN:?Set MASK_PLAN to a direct-pruning mask .pt file.}"
if [[ ! -f "${MASK_PLAN}" ]]; then
    echo "error: mask plan does not exist: ${MASK_PLAN}" >&2
    exit 2
fi
export MAES_MASK_PLAN="$(realpath "${MASK_PLAN}")"
export PYTHONPATH="${REPO_ROOT}/runtime/vllm_ep4:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
conda run --no-capture-output -n vllm-maes python -c \
    'import os; from src.vllm_mask_runtime import load_mask_plan; p=load_mask_plan(os.environ["MAES_MASK_PLAN"]); assert p["model"] == "moonshotai/Kimi-VL-A3B-Instruct"'
export MODEL=moonshotai/Kimi-VL-A3B-Instruct
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vllm_ours/kimi/direct-mask}"
export BASELINE_LABEL="${BASELINE_LABEL:-kimi_vl_direct_mask}"
export PARALLEL_MODE=ep4
export RUNNER_SCRIPT=scripts/run_kimi_vl_vllm_mask_pruned.sh
exec bash scripts/run_kimi_vl_vllm_baseline.sh "$@"
