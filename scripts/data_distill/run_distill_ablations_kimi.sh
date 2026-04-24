#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$(pwd)}"
BASE_SCRIPT="${BASE_SCRIPT:-${PREFIX}/scripts/data_distill/run_distill_compact_hidden.sh}"
MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
# HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH:-${PREFIX}/storage/data_distill_kimi/gqa-sample_at1.0-latest/teacher_hidden.pt}"
HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH:-storage/data_distill_kimi/mixed-num_1024-token_2048-sample_at1.0-0423143941/teacher_hidden.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-maes}"
WANDB_MODE="${WANDB_MODE:-online}"
ABLATION_SUITE="${ABLATION_SUITE:-all}"  # all | diversity | distribution
RUN_STAMP="${RUN_STAMP:-$(date +%m%d%H%M%S)}"
EXTRA_ARGS=("$@")

if [[ ! -f "${BASE_SCRIPT}" ]]; then
    echo "error: base script not found: ${BASE_SCRIPT}" >&2
    exit 1
fi

run_case() {
    local suite="$1"
    local case_name="$2"
    local diversity_ablation="$3"
    local distribution_ablation="$4"

    local wandb_run_name="kimi-distill-${suite}-${case_name}-${RUN_STAMP}"

    echo "==== suite=${suite} case=${case_name} div=${diversity_ablation} dist=${distribution_ablation} ===="
    MODEL_PATH="${MODEL_PATH}" \
    HIDDEN_PAYLOAD_PATH="${HIDDEN_PAYLOAD_PATH}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_RUN_NAME="${wandb_run_name}" \
    DIVERSITY_ABLATION="${diversity_ablation}" \
    DISTRIBUTION_ABLATION="${distribution_ablation}" \
    bash "${BASE_SCRIPT}" \
        "${EXTRA_ARGS[@]}"
}

if [[ "${ABLATION_SUITE}" == "all" || "${ABLATION_SUITE}" == "diversity" ]]; then
    run_case "diversity" "full" "full" "full"
    run_case "diversity" "no_div" "no_div" "full"
fi

if [[ "${ABLATION_SUITE}" == "all" || "${ABLATION_SUITE}" == "distribution" ]]; then
    run_case "distribution" "full" "full" "full"
    run_case "distribution" "moment_only" "full" "moment_only"
    run_case "distribution" "mmd_only" "full" "mmd_only"
fi
