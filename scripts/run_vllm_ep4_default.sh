#!/usr/bin/env bash
set -euo pipefail

# p=0 "Default" deployment baseline: native EP4 vLLM, no MAES pruning plan,
# no runtime patch installed. Mirrors run_vllm_ep4_pruned.sh's env scaffolding
# minus everything that depends on an EP4_PLAN, so it is a fair unpruned
# comparison point for the same MODEL/TASKS/batch-size efficiency sweep.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:?Set MODEL to the exact model ID.}"

case "${MODEL}" in
    moonshotai/Kimi-VL-A3B-Instruct)
        RUNNER="scripts/run_kimi_vl_vllm_baseline.sh"
        MODEL_TAG="kimi-vl-30b-a3b"
        ;;
    Qwen/Qwen3-VL-30B-A3B-Instruct)
        RUNNER="scripts/run_qwen3_vl_vllm_baseline.sh"
        MODEL_TAG="qwen3-vl-30b-a3b"
        ;;
    OpenGVLab/InternVL3_5-30B-A3B-HF)
        RUNNER="scripts/run_internvl35_vllm_baseline.sh"
        MODEL_TAG="internvl3_5-30b-a3b-hf"
        ;;
    *)
        echo "error: unsupported EP4 model: ${MODEL}" >&2
        exit 2
        ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vllm_ours/${MODEL_TAG}}"
export MAES_EXPERIMENT_FINGERPRINT="ep4_plan_sha256=none-default-p0"
unset MAES_EP4_PLAN MAES_MASK_PLAN MAES_EP4_STRATEGY || true
export MODEL OUTPUT_ROOT
export BASELINE_LABEL="${BASELINE_LABEL:-${MODEL_TAG}_default}"
export PARALLEL_MODE=ep4
export ENFORCE_EAGER=1
export RUNNER_SCRIPT="scripts/run_vllm_ep4_default.sh"

exec bash "${RUNNER}" "$@"
