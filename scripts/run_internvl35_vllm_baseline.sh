#!/usr/bin/env bash
set -euo pipefail

# Unpruned InternVL3.5-30B-A3B-HF baseline through lmms-eval's vLLM backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL="${MODEL:-OpenGVLab/InternVL3_5-30B-A3B-HF}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vllm_baseline/internvl3_5-30b-a3b-hf}"
export BASELINE_LABEL="${BASELINE_LABEL:-internvl35_vllm}"
export ENABLE_QWEN3_NATIVE_VIDEO=0
export RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_internvl35_vllm_baseline.sh}"
export VIDEOMME_MAX_MODEL_LEN="${VIDEOMME_MAX_MODEL_LEN:-40960}"
export VIDEO_MMMU_MAX_MODEL_LEN="${VIDEO_MMMU_MAX_MODEL_LEN:-40960}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_FRAME_NUM="${MAX_FRAME_NUM:-8}"
export VIDEO_NFRAMES="${VIDEO_NFRAMES:-8}"
export DEFAULT_TASKS="${DEFAULT_TASKS:-gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset,mvbench_available_3800}"

exec bash "${SCRIPT_DIR}/run_qwen3_vl_vllm_baseline.sh" "$@"
