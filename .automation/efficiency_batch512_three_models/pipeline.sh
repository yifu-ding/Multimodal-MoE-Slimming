#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
A=.automation/efficiency_batch512_three_models
R=artifacts/efficiency_batch512_three_models/runs

plan_for() { case "$1:$2" in
    p30:kimi) echo artifacts/efficiency_batch64_three_models/plans/kimi-vl-p30-seed2603.pt;;
    p30:qwen30) echo artifacts/efficiency_batch64_three_models/plans/qwen3-vl-30b-p30-seed2603.pt;;
    p30:internvl30) echo artifacts/efficiency_batch64_three_models/plans/internvl3_5-p30-seed2603.pt;;
    p50:kimi) echo artifacts/efficiency_batch64_three_models/plans/kimi-vl-p50-seed2603.pt;;
    p50:qwen30) echo artifacts/efficiency_batch64_three_models/plans/qwen3-vl-30b-p50-seed2603.pt;;
    p50:internvl30) echo artifacts/efficiency_batch64_three_models/plans/internvl3_5-p50-seed2603.pt;;
esac; }
wait_gpu() { for _ in {1..60}; do bash "$A/resource_check.sh" && return; sleep 2; done; return 1; }
validate() { python "$A/validate_result.py" "$1" "$2" "$3" "$4"; }

run_case() {
    local ratio="$1" model="$2" strategy="$3" plan dir prune sha
    plan="$(plan_for "${ratio}" "${model}")"; dir="$R/${ratio}/${model}/${strategy}/bs_512"
    validate "${dir}" "${model}" "${ratio}" "${strategy}" && { echo "[skip] ${ratio} ${model} ${strategy}"; return; }
    wait_gpu; mkdir -p "${dir}"; rm -f "${dir}/COMPLETE" "${dir}/FAILED"
    [[ "${ratio}" == p30 ]] && prune=0.3 || prune=0.5; sha="$(sha256sum "${plan}" | awk '{print $1}')"
    cat > "${dir}/run_identity.json" <<EOF
{"model_key":"${model}","prune_ratio":${prune},"implementation":"${strategy}","plan":"${plan}","plan_sha256":"${sha}","batch_size":512,"prefill_tokens":512,"decode_prompt_tokens":32,"decode_tokens":128,"warmup_runs":1,"measured_runs":3,"prefix_cache":false,"kv_cache_memory_bytes_per_gpu":2147483648}
EOF
    if MODEL_KEY="${model}" EP4_PLAN="${plan}" STRATEGY="${strategy}" BATCH_SIZE=512 RUN_DIR="${dir}" \
        PREFILL_TOKENS=512 DECODE_PROMPT_TOKENS=32 DECODE_TOKENS=128 WARMUP_RUNS=1 MEASURED_RUNS=3 \
        MAX_MODEL_LEN=2048 MAX_NUM_BATCHED_TOKENS=262144 bash scripts/run_ep4_efficiency.sh \
        && validate "${dir}" "${model}" "${ratio}" "${strategy}"; then
        echo "[done] ${ratio} ${model} ${strategy}"
    else date --iso-8601=seconds > "${dir}/FAILED"; exit 1; fi
    python "$A/update_report.py"
}

for ratio in p30 p50; do for model in kimi qwen30 internvl30; do for strategy in padded multi_kernel single_width cross_layer; do
    run_case "${ratio}" "${model}" "${strategy}"
done; done; done
python "$A/update_report.py"
echo "Three-model p30/p50 batch=512 campaign complete."
