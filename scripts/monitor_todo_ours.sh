#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"
AUTOMATION_LOG="${REPO_ROOT}/results/todo_ours.log"
ALL_TASKS="gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset,mvbench_available_3800"
OURS_GPU_MEMORY_UTILIZATION="${OURS_GPU_MEMORY_UTILIZATION:-0.85}"
RUN_P50="${RUN_P50:-0}"
OURS_MODELS="${OURS_MODELS:-all}"
ONLY_P50="${ONLY_P50:-0}"
OURS_CPU_AFFINITY="${OURS_CPU_AFFINITY:-0-23,25,27,29,31}"
export TASK_MAX_ATTEMPTS=1 JUDGE_MAX_ATTEMPTS=1

append_record() {
    printf -- '- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "${REPORT}"
}

plan_path_for() {
    local tag="$1"
    local ratio_tag="$2"
    local sparse_tier_v2="${REPO_ROOT}/runtime/ep4_plans/${tag}-${ratio_tag}-sparse-tier-v2.pt"
    if [[ -f "${sparse_tier_v2}" ]]; then
        printf '%s' "${sparse_tier_v2}"
        return
    fi
    printf '%s/storage/ep4_plans/%s-%s.pt' "${REPO_ROOT}" "${tag}" "${ratio_tag}"
}

run_stage() {
    local label="$1"
    shift
    append_record "开始 ${label}。"
    set +e
    taskset -c "${OURS_CPU_AFFINITY}" "$@" 2>&1 | tee -a "${AUTOMATION_LOG}"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        append_record "ATTENTION：${label} 失败，exit_code=${status}；详见 ${AUTOMATION_LOG}。未原样重试。"
        return "${status}"
    fi
    append_record "${label} 完成。"
}

run_one() {
    local model="$1"
    local tag="$2"
    local ratio="$3"
    local ratio_tag="$4"
    local plan smoke_run full_run failed=0
    plan="$(plan_path_for "${tag}" "${ratio_tag}")"
    smoke_run="${REPO_ROOT}/results/vllm_ours/${tag}/ep4-${ratio_tag}-smoke"
    full_run="${REPO_ROOT}/results/vllm_ours/${tag}/ep4-${ratio_tag}-full"

    if python scripts/ours_campaign_state.py check --model "${tag}" --ratio "${ratio_tag}" >/dev/null; then
        append_record "${tag} ${ratio_tag} benchmark/Judge 产物完整，跳过已完成配置。"
        return 0
    fi
    python scripts/ours_campaign_state.py stage --model "${tag}" --ratio "${ratio_tag}" --phase smoke --run-dir "${smoke_run}"

    if ! run_stage "${tag} ${ratio_tag} 一样本 EP4 smoke" \
        env DECORD_EOF_RETRY_MAX=20480 \
            CUDA_VISIBLE_DEVICES=0,1,2,3 \
            GPU_MEMORY_UTILIZATION="${OURS_GPU_MEMORY_UTILIZATION}" FORCE=1 \
            EP4_PLAN="${plan}" MODEL="${model}" \
            PRUNING_LABEL="ours_${ratio_tag}" \
            TASKS=gqa LIMIT=1 FAIL_FAST=1 \
            RUN_DIR="${smoke_run}" \
            bash scripts/run_vllm_ep4_pruned.sh; then
        append_record "ATTENTION：${tag} ${ratio_tag} smoke 失败；需 Debug 后恢复该配置。"
        return 1
    fi

    if ! grep -RqsE '\[MAES EP4\].*layer=' "${smoke_run}/logs"; then
        append_record "${tag} ${ratio_tag} smoke 未检测到 EP4 运行时激活记录；拒绝启动全量评测。"
        return 1
    fi

    python scripts/ours_campaign_state.py stage --model "${tag}" --ratio "${ratio_tag}" --phase benchmark --run-dir "${full_run}"
    if ! run_stage "${tag} ${ratio_tag} Ours 全量评测" \
        env DECORD_EOF_RETRY_MAX=20480 \
            CUDA_VISIBLE_DEVICES=0,1,2,3 \
            GPU_MEMORY_UTILIZATION="${OURS_GPU_MEMORY_UTILIZATION}" \
            EP4_PLAN="${plan}" MODEL="${model}" \
            PRUNING_LABEL="ours_${ratio_tag}" \
            TASKS="${ALL_TASKS}" FAIL_FAST=0 \
            RUN_DIR="${full_run}" \
            bash scripts/run_vllm_ep4_pruned.sh; then
        failed=1
        append_record "ATTENTION：${tag} ${ratio_tag} 有失败数据集；保留成功结果，继续 Judge、汇总；失败项需 Debug 后恢复。"
    fi

    python scripts/ours_campaign_state.py stage --model "${tag}" --ratio "${ratio_tag}" --phase Judge --run-dir "${full_run}"
    if ! run_stage "${tag} ${ratio_tag} Ours 本地 Judge" \
        env PREDICTIONS_DIR="${full_run}" PORT=8010 CUDA_VISIBLE_DEVICES=0 \
            bash scripts/run_vllm_judge_stage.sh; then
        append_record "${tag} ${ratio_tag} Judge 失败；保留无需 Judge 的结果并继续。"
        failed=1
    fi

    python scripts/ours_campaign_state.py stage --model "${tag}" --ratio "${ratio_tag}" --phase 汇总 --run-dir "${full_run}"
    if ! run_stage "${tag} ${ratio_tag} Ours 结果汇总" \
        python scripts/collect_vllm_results_md.py \
            --run-dir "${full_run}" \
            --model "${model}" \
            --pruning "Ours ${ratio}" \
            --report "${REPORT}"; then
        append_record "${tag} ${ratio_tag} 结果汇总失败；已记录并继续下一配置。"
        failed=1
    fi
    python scripts/ours_campaign_state.py check --model "${tag}" --ratio "${ratio_tag}" || failed=1
    return "${failed}"
}

main() {
cd "${REPO_ROOT}"
append_record "Ours 接力器已启动，GPU_MEMORY_UTILIZATION=${OURS_GPU_MEMORY_UTILIZATION}，CPU_AFFINITY=${OURS_CPU_AFFINITY}；等待三模型 30%/50% EP4 plans。"
while tmux has-session -t maes-todo-scores 2>/dev/null; do
    sleep 60
done

run_specifications() {
    local specification model tag ratio ratio_tag failed=0
    for specification in "$@"; do
        IFS='|' read -r model tag ratio ratio_tag <<< "${specification}"
        if [[ "${OURS_MODELS}" != "all" && ",${OURS_MODELS}," != *",${tag},"* ]]; then
            continue
        fi
        if [[ ! -f "$(plan_path_for "${tag}" "${ratio_tag}")" ]]; then
            append_record "${tag} ${ratio_tag} 缺少通过校验的 EP4 plan；跳过该配置并继续。"
            failed=1
            continue
        fi
        run_one "${model}" "${tag}" "${ratio}" "${ratio_tag}" || failed=1
    done
    return "${failed}"
}

# Finish all three p30 configurations before any p50 evaluation starts.
if [[ "${ONLY_P50}" != 1 ]]; then
run_specifications \
    'Qwen/Qwen3-VL-30B-A3B-Instruct|qwen3-vl-30b-a3b|0.3|p30' \
    'moonshotai/Kimi-VL-A3B-Instruct|kimi|0.3|p30' \
    'OpenGVLab/InternVL3_5-30B-A3B-HF|internvl3_5-30b-a3b|0.3|p30'
fi

if [[ "${RUN_P50}" != "1" ]]; then
    append_record "本次 p30 Ours 调用完成；p50 由已授权的独立队列按顺序接续。"
    exit 0
fi

run_specifications \
    'Qwen/Qwen3-VL-30B-A3B-Instruct|qwen3-vl-30b-a3b|0.5|p50' \
    'moonshotai/Kimi-VL-A3B-Instruct|kimi|0.5|p50' \
    'OpenGVLab/InternVL3_5-30B-A3B-HF|internvl3_5-30b-a3b|0.5|p50'
append_record "三个模型 50% Ours benchmark/Judge 产物检查通过。"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
