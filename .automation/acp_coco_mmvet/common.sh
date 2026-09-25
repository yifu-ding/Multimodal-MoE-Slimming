#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=/home/dyf/code/distill/MAES
AUTOMATION_DIR="${REPO_ROOT}/.automation/acp_coco_mmvet"
PLAN_DIR="${REPO_ROOT}/runtime/ep4_plans/acp_align128_rebuilt"
RESULT_ROOT="${REPO_ROOT}/results/vllm_acp_align128_completion"
REPORT="${REPO_ROOT}/docs/acp_coco_mmvet_results.md"
STATUS="${AUTOMATION_DIR}/STATUS.md"
CONFIGS=(kimi_p30 qwen30_p30 internvl30_p30 kimi_p50 qwen30_p50 internvl30_p50)

config_spec() {
    case "$1" in
        kimi_p30) echo 'moonshotai/Kimi-VL-A3B-Instruct|kimi-mixed-512|kimi-vl-a3b|0.3|p30' ;;
        qwen30_p30) echo 'Qwen/Qwen3-VL-30B-A3B-Instruct|qwen3-mixed-512|qwen3-vl-30b-a3b|0.3|p30' ;;
        internvl30_p30) echo 'OpenGVLab/InternVL3_5-30B-A3B-HF|internvl3_5-30b-a3b-mixed-512|internvl3_5-30b-a3b|0.3|p30' ;;
        kimi_p50) echo 'moonshotai/Kimi-VL-A3B-Instruct|kimi-mixed-512|kimi-vl-a3b|0.5|p50' ;;
        qwen30_p50) echo 'Qwen/Qwen3-VL-30B-A3B-Instruct|qwen3-mixed-512|qwen3-vl-30b-a3b|0.5|p50' ;;
        internvl30_p50) echo 'OpenGVLab/InternVL3_5-30B-A3B-HF|internvl3_5-30b-a3b-mixed-512|internvl3_5-30b-a3b|0.5|p50' ;;
        *) return 1 ;;
    esac
}

config_paths() {
    local name="$1" model scores_tag model_tag ratio ratio_tag
    IFS='|' read -r model scores_tag model_tag ratio ratio_tag <<< "$(config_spec "${name}")"
    printf '%s|%s|%s\n' \
        "${REPO_ROOT}/storage/scores/${scores_tag}/scores.pt" \
        "${PLAN_DIR}/${model_tag}-acp-align128-${ratio_tag}.pt" \
        "${RESULT_ROOT}/${model_tag}/ep4-${ratio_tag}-padded/run"
}
