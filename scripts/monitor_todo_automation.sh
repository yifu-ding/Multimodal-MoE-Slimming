#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"
AUTOMATION_LOG="${REPO_ROOT}/results/todo_automation.log"
VIDEOMME_RUN="${REPO_ROOT}/results/vllm_baseline/qwen3-vl-30b-a3b/ep4-videomme-native-video"
VIDEOMME_TASK="videomme_qwen3_vllm"
QWEN_REMAINDER_RUN="${REPO_ROOT}/results/vllm_baseline/qwen3-vl-30b-a3b/ep4-full-no-mvbench"
KIMI_RUN="${REPO_ROOT}/results/vllm_baseline/kimi-vl-30b-a3b/ep4-full-all"

append_record() {
    printf -- '- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "${REPORT}"
}

latest_failure_context() {
    local run_dir="$1"
    local task_name="$2"
    local started_at="$3"
    local heartbeat
    heartbeat="$(find "${run_dir}/watchdog/${task_name}" -type f -name response_cache.json -printf '%T@ %p\n' 2>/dev/null | awk -v start="${started_at}" '$1 >= start {sub(/^[^ ]+ /, ""); print}' | tail -n 1 || true)"
    if [[ -n "${heartbeat}" && -f "${heartbeat}" ]]; then
        tr '\n' ' ' < "${heartbeat}"
    else
        printf '没有可用的逐样本心跳'
    fi
}

run_logged_stage() {
    local label="$1"
    local run_dir="$2"
    local task_name="$3"
    local started_at
    shift 3
    append_record "开始 ${label}。"
    started_at="$(date +%s)"
    set +e
    "$@" 2>&1 | tee -a "${AUTOMATION_LOG}"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        append_record "${label} 失败，exit_code=${status}；最后心跳：$(latest_failure_context "${run_dir}" "${task_name}" "${started_at}")；详见 ${AUTOMATION_LOG} 和对应 RUN_DIR/logs。已释放本阶段进程组并继续后续任务。"
        return "${status}"
    fi
    append_record "${label} 完成。"
}

cd "${REPO_ROOT}"
append_record "自动接力重新启动：逐条持久化视频响应；30 分钟无新响应时终止整个任务进程组、记录最后样本并继续。"

if run_logged_stage "Qwen3-VL Video-MME（断点缓存重跑）" "${VIDEOMME_RUN}" "${VIDEOMME_TASK}" \
    env DECORD_EOF_RETRY_MAX=20480 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PARALLEL_MODE=ep4 \
        TASKS=videomme \
        RUN_DIR="${VIDEOMME_RUN}" \
        STALL_TIMEOUT_SECONDS=1800 \
        WATCHDOG_POLL_SECONDS=30 \
        bash scripts/run_qwen3_vl_vllm_baseline.sh; then
    append_record "Qwen3-VL Video-MME 完成，准备运行 EgoSchema subset 和 MVBench available 3800。"
fi

if run_logged_stage "Qwen3-VL EgoSchema subset 与 MVBench available 3800" "${QWEN_REMAINDER_RUN}" "egoschema_subset,mvbench_available_3800" \
    env DECORD_EOF_RETRY_MAX=20480 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PARALLEL_MODE=ep4 \
        TASKS=egoschema_subset,mvbench_available_3800 \
        RUN_DIR="${QWEN_REMAINDER_RUN}" \
        bash scripts/run_qwen3_vl_vllm_baseline.sh; then
    :
fi

if run_logged_stage "Kimi-VL 全量 baseline 预测与本地评分" "${KIMI_RUN}" "all" \
    env DECORD_EOF_RETRY_MAX=20480 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PARALLEL_MODE=ep4 \
        RUN_DIR="${KIMI_RUN}" \
        TASKS=gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset,mvbench_available_3800 \
        bash scripts/run_kimi_vl_vllm_baseline.sh; then
    :
fi

append_record "Kimi-VL 四卡阶段结束；Judge 和后续 TODO 由持续执行任务接管。"
