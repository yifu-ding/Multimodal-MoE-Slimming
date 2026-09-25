#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
for model in kimi qwen30 internvl30; do
    python .automation/efficiency_p0/validate_result.py "artifacts/efficiency_p0/runs/${model}/default/bs_512" "${model}" 512 || exit 2
    for ratio in p30 p50; do for strategy in padded multi_kernel single_width cross_layer; do
        dir="artifacts/efficiency_batch512_three_models/runs/${ratio}/${model}/${strategy}/bs_512"
        [[ ! -f "${dir}/FAILED" ]] || exit 2
        python .automation/efficiency_batch512_three_models/validate_result.py "${dir}" "${model}" "${ratio}" "${strategy}" || exit 1
    done; done
done
