#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EP4_PLAN="${EP4_PLAN:?Set EP4_PLAN to a compiled .pt plan.}"
MODEL="${MODEL:?Set MODEL to the exact model ID stored in the plan.}"
if [[ ! -f "${EP4_PLAN}" ]]; then
    echo "error: EP4_PLAN does not exist: ${EP4_PLAN}" >&2
    exit 2
fi

PLAN_INFO="$({
    EP4_PLAN="${EP4_PLAN}" EXPECTED_MODEL="${MODEL}" \
        conda run --no-capture-output -n vllm-maes python - <<'PY'
import os
from src.vllm_ep4_plan import load_ep4_plan

plan = load_ep4_plan(os.environ["EP4_PLAN"])
expected = os.environ["EXPECTED_MODEL"]
if plan["model"] != expected:
    raise SystemExit(f"plan model {plan['model']!r} does not match MODEL={expected!r}")
print(
    f"model={plan['model']} requested_prune={plan['prune_ratio']:.6f} "
    f"actual_prune={plan['actual_prune_ratio']:.6f}"
)
PY
} 2>&1)" || {
    echo "error: invalid EP4 plan: ${PLAN_INFO}" >&2
    exit 2
}
echo "[MAES EP4] ${PLAN_INFO}"
PLAN_SHA256="$(sha256sum "${EP4_PLAN}" | awk '{print $1}')"

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

PRUNING_LABEL="${PRUNING_LABEL:-ours}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vllm_ours/${MODEL_TAG}}"
export MAES_EP4_PLAN="$(realpath "${EP4_PLAN}")"
export MAES_EXPERIMENT_FINGERPRINT="ep4_plan_sha256=${PLAN_SHA256}"
export PYTHONPATH="${REPO_ROOT}/runtime/vllm_ep4${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL OUTPUT_ROOT
export BASELINE_LABEL="${BASELINE_LABEL:-${MODEL_TAG}_${PRUNING_LABEL}}"
export PARALLEL_MODE=ep4
export ENFORCE_EAGER=1
export RUNNER_SCRIPT="scripts/run_vllm_ep4_pruned.sh"

exec bash "${RUNNER}" "$@"
