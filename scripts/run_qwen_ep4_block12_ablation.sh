#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PLAN_ROOT="${PLAN_ROOT:-${REPO_ROOT}/artifacts/efficiency_figure/qwen3_gqa_ep4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PLAN_ROOT}/block12_ablation/prune_50}"
MEASURED_BATCHES="${MEASURED_BATCHES:-4}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"

declare -A plans=(
    [greedy]="${PLAN_ROOT}/weight_proxy_ep4_50.pt"
    [block12_cyclic]="${PLAN_ROOT}/weight_proxy_ep4_50_block12_cyclic.pt"
    [block12_optimized]="${PLAN_ROOT}/weight_proxy_ep4_50_block12_optimized.pt"
)

for variant in greedy block12_cyclic block12_optimized; do
    plan="${plans[${variant}]}"
    if [[ ! -f "${plan}" ]]; then
        echo "error: missing plan ${plan}" >&2
        exit 2
    fi
    for batch_size in 64 512; do
        EP4_PLAN="${plan}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}/${variant}" \
        START_BATCH_SIZE="${batch_size}" \
        MAX_BATCH_SIZE="${batch_size}" \
        MEASURED_BATCHES="${MEASURED_BATCHES}" \
        WARMUP_BATCHES="${WARMUP_BATCHES}" \
        GPU_MEMORY_UTILIZATION=0.90 \
        STRATEGIES=cross_layer \
            bash scripts/run_qwen_ep4_batch_sweep.sh
    done
done

echo "block12_ablation_results=${OUTPUT_ROOT}"
