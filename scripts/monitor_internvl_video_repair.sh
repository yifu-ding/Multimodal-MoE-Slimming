#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"
LOG="${REPO_ROOT}/results/internvl_video_repair.log"
RUN_DIR="${REPO_ROOT}/results/vllm_baseline/internvl3_5-30b-a3b-hf/ep4-full-all"
TASKS="videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset_local,mvbench_available_3800"

record() {
    printf -- '- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "${REPORT}"
}

run_logged() {
    local label="$1"
    shift
    record "开始 ${label}。"
    set +e
    "$@" 2>&1 | tee -a "${LOG}"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        record "${label} 结束时仍有失败项，exit_code=${status}；保留缓存并继续后续阶段。"
    else
        record "${label} 完成。"
    fi
    return 0
}

cd "${REPO_ROOT}"
record "InternVL 视频上下文修复已通过 VideoMMMU、EgoSchema、MVBench 单样本验证：显式 nframes=8 后输入落入模型原生 40960 上下文。"

run_logged "InternVL 五个视频任务修复后全量补跑" \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 PARALLEL_MODE=ep4 \
        RUN_DIR="${RUN_DIR}" TASKS="${TASKS}" FORCE=1 \
        TASK_MAX_ATTEMPTS=2 VIDEO_NFRAMES=8 MAX_FRAME_NUM=8 \
        MAX_MODEL_LEN=40960 VIDEOMME_MAX_MODEL_LEN=40960 \
        VIDEO_MMMU_MAX_MODEL_LEN=40960 \
        bash scripts/run_internvl35_vllm_baseline.sh

if find "${RUN_DIR}" -type f -name '*_samples_video_mmmu_*_local.jsonl' \
    -not -path '*/local_judge/*' -print -quit 2>/dev/null | grep -q .; then
    run_logged "InternVL VideoMMMU 本地 Judge" \
        env PREDICTIONS_DIR="${RUN_DIR}" TASKS=videommmu \
            PORT=8010 CUDA_VISIBLE_DEVICES=0 \
            bash scripts/run_vllm_judge_stage.sh
fi

run_logged "InternVL 修复后结果汇总" \
    python scripts/collect_vllm_results_md.py \
        --run-dir "${RUN_DIR}" \
        --model "OpenGVLab/InternVL3_5-30B-A3B-HF" \
        --pruning Unpruned

record "InternVL 视频补跑阶段结束；开始三模型 scores/plan 接力。"
