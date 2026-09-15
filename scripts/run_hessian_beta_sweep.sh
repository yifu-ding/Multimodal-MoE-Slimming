#!/usr/bin/env bash
set -euo pipefail

# Standalone P10/P50/P90 beta sweep. All scientific settings default to the
# metadata stored in HESSIAN_PROBE and inconsistent overrides are rejected.

source scripts/select_least_used_gpu.sh

if [[ "${CONDA_DEFAULT_ENV:-}" == "ds-vl2-h20" ]]; then
    export LD_LIBRARY_PATH="/home/dyf/miniconda/envs/ds-vl2-h20/lib:${LD_LIBRARY_PATH:-}"
fi

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/maes-matplotlib-cache}"

HESSIAN_PROBE="${HESSIAN_PROBE:-}"
if [[ -z "${HESSIAN_PROBE}" ]]; then
    echo "HESSIAN_PROBE is required (path to hessian_probe_L*.pt)." >&2
    exit 1
fi
if [[ ! -f "${HESSIAN_PROBE}" ]]; then
    echo "Hessian probe does not exist: ${HESSIAN_PROBE}" >&2
    exit 1
fi

OUTPUT="${OUTPUT:-}"
MODEL_PATH="${MODEL_PATH:-}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
LAYER="${LAYER:-}"
LOSS_FN="${LOSS_FN:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SCORE_TOKENS_PER_SAMPLE="${SCORE_TOKENS_PER_SAMPLE:-}"
SCORE_TOKEN_BUDGET="${SCORE_TOKEN_BUDGET:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
BETAS="${BETAS:--0.5 0 0.25 0.5 0.75 1 1.25 1.5}"
FORCE="${FORCE:-0}"
read -r -a BETA_VALUES <<< "${BETAS}"

CMD=(
    python -m src.calibration.collect_hessian_beta_sweep
    --probe "${HESSIAN_PROBE}"
    --betas "${BETA_VALUES[@]}"
)

[[ -n "${OUTPUT}" ]] && CMD+=(--output "${OUTPUT}")
[[ -n "${MODEL_PATH}" ]] && CMD+=(--model_name_or_path "${MODEL_PATH}")
[[ -n "${SELECTION_MANIFEST}" ]] && CMD+=(--selection_manifest "${SELECTION_MANIFEST}")
[[ -n "${LAYER}" ]] && CMD+=(--layer "${LAYER}")
[[ -n "${LOSS_FN}" ]] && CMD+=(--loss_fn "${LOSS_FN}")
[[ -n "${BATCH_SIZE}" ]] && CMD+=(--batch_size "${BATCH_SIZE}")
[[ -n "${SCORE_TOKENS_PER_SAMPLE}" ]] && CMD+=(--score_tokens_per_sample "${SCORE_TOKENS_PER_SAMPLE}")
[[ -n "${SCORE_TOKEN_BUDGET}" ]] && CMD+=(--score_token_budget "${SCORE_TOKEN_BUDGET}")
[[ -n "${DEVICE_MAP}" ]] && CMD+=(--device_map "${DEVICE_MAP}")
[[ -n "${ATTN_IMPLEMENTATION}" ]] && CMD+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
[[ "${FORCE}" == "1" ]] && CMD+=(--force)
CMD+=("$@")

echo "Probe      : ${HESSIAN_PROBE}"
echo "Betas      : ${BETAS}"
echo "Overrides  : model=${MODEL_PATH:-<probe>} manifest=${SELECTION_MANIFEST:-<probe>} layer=${LAYER:-<probe>} loss=${LOSS_FN:-<probe>}"
"${CMD[@]}"
