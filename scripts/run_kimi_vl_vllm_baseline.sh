#!/usr/bin/env bash
set -euo pipefail

# Unpruned Kimi-VL-A3B-Instruct baseline through lmms-eval's vLLM backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL="${MODEL:-moonshotai/Kimi-VL-A3B-Instruct}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vllm_baseline/kimi-vl-30b-a3b}"
export BASELINE_LABEL="${BASELINE_LABEL:-kimi_vl_vllm}"
export ENABLE_QWEN3_NATIVE_VIDEO=0
export MAX_FRAME_NUM="${MAX_FRAME_NUM:-32}"
# Kimi-VL receives sampled video frames as separate image items. Keep the
# vLLM multimodal limit aligned with the runner's frame cap.
export LIMIT_MM_PER_PROMPT_JSON="${LIMIT_MM_PER_PROMPT_JSON:-{\"image\":${MAX_FRAME_NUM}}}"
export PYTHONPATH="${REPO_ROOT}/runtime/vllm_kimi_sdpa${PYTHONPATH:+:${PYTHONPATH}}"
export RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_kimi_vl_vllm_baseline.sh}"
export VIDEOMME_MAX_MODEL_LEN="${VIDEOMME_MAX_MODEL_LEN:-131072}"
export VIDEO_MMMU_MAX_MODEL_LEN="${VIDEO_MMMU_MAX_MODEL_LEN:-131072}"
export DEFAULT_TASKS="${DEFAULT_TASKS:-gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset,mvbench_available_3800}"

exec bash "${SCRIPT_DIR}/run_qwen3_vl_vllm_baseline.sh" "$@"
