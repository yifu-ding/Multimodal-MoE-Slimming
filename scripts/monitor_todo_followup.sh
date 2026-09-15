#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"
AUTOMATION_LOG="${REPO_ROOT}/results/todo_followup.log"
KIMI_RUN="${REPO_ROOT}/results/vllm_baseline/kimi-vl-30b-a3b/ep4-full-all"
INTERNVL_RUN="${REPO_ROOT}/results/vllm_baseline/internvl3_5-30b-a3b-hf/ep4-full-all"
ALL_TASKS="gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset,mvbench_available_3800"
KIMI_VIDEO_TASKS="videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset,mvbench_available_3800"
PRIMARY_SESSION="maes-todo-auto-v4"

append_record() {
    printf -- '- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "${REPORT}"
}

wait_for_run_end() {
    local run_dir="$1"
    while :; do
        if [[ -f "${run_dir}/run_summary.txt" ]]; then
            return 0
        fi
        if ! tmux has-session -t "${PRIMARY_SESSION}" 2>/dev/null; then
            append_record "等待 ${run_dir}/run_summary.txt 时发现主接力器 ${PRIMARY_SESSION} 已退出；记录缺失并继续。"
            return 1
        fi
        sleep 60
    done
}

available_judge_tasks() {
    local run_dir="$1"
    local available=()
    find "${run_dir}" -type f -name '*_samples_mmvet.jsonl' -not -path '*/local_judge/*' -print -quit 2>/dev/null | grep -q . && available+=(mmvet)
    find "${run_dir}" -type f -name '*_samples_mmbench_en_dev*.jsonl' -not -path '*/local_judge/*' -print -quit 2>/dev/null | grep -q . && available+=(mmbench)
    find "${run_dir}" -type f -name '*_samples_video_mmmu_*_local.jsonl' -not -path '*/local_judge/*' -print -quit 2>/dev/null | grep -q . && available+=(videommmu)
    local IFS=,
    printf '%s' "${available[*]}"
}

run_available_judge() {
    local label="$1"
    local run_dir="$2"
    local tasks
    tasks="$(available_judge_tasks "${run_dir}")"
    if [[ -z "${tasks}" ]]; then
        append_record "${label} 没有可用预测文件；记录缺失并跳过 Judge。"
        return 0
    fi
    append_record "${label} 仅评测已有预测：${tasks}；缺失数据集单独跳过。"
    if ! run_stage "${label}" \
        env PREDICTIONS_DIR="${run_dir}" TASKS="${tasks}" PORT=8010 CUDA_VISIBLE_DEVICES=0 \
            bash scripts/run_vllm_judge_stage.sh; then
        append_record "${label} 失败；保留已逐条落盘的 Judge 结果并继续下一项。"
    fi
}

run_stage() {
    local label="$1"
    shift
    append_record "开始 ${label}。"
    set +e
    "$@" 2>&1 | tee -a "${AUTOMATION_LOG}"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        append_record "${label} 在允许的一次重试后仍失败，exit_code=${status}；详见 ${AUTOMATION_LOG}。跳过本项并继续。"
        return "${status}"
    fi
    append_record "${label} 完成。"
}

cd "${REPO_ROOT}"
append_record "Kimi/InternVL 后续接力器已启动；Kimi 五个视频任务的修复 smoke 已全部通过，开始补跑全量。"

if run_stage "Kimi-VL 修复后的五个视频 baseline" \
    env DECORD_EOF_RETRY_MAX=20480 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PARALLEL_MODE=ep4 \
        RUN_DIR="${KIMI_RUN}" \
        TASKS="${KIMI_VIDEO_TASKS}" \
        FORCE=1 \
        bash scripts/run_kimi_vl_vllm_baseline.sh; then :; fi

run_available_judge "Kimi-VL 本地 Judge（MMVet、MMBench、VideoMMMU）" "${KIMI_RUN}"
if ! run_stage "Kimi-VL 结果汇总" \
    python scripts/collect_vllm_results_md.py \
        --run-dir "${KIMI_RUN}" \
        --model "moonshotai/Kimi-VL-A3B-Instruct" \
        --pruning "Unpruned" \
        --report "${REPORT}"; then :; fi

append_record "Qwen3-VL Video-MME、EgoSchema 与 MVBench 已由主接力器按每项最多两次策略处理；后续接力器不再重复补跑，以免产生第三次尝试。"

if run_stage "InternVL3.5-30B-A3B-HF 全量 baseline" \
    env DECORD_EOF_RETRY_MAX=20480 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PARALLEL_MODE=ep4 \
        RUN_DIR="${INTERNVL_RUN}" \
        TASKS="${ALL_TASKS}" \
        bash scripts/run_internvl35_vllm_baseline.sh; then :; fi

run_available_judge "InternVL3.5 本地 Judge（MMVet、MMBench、VideoMMMU）" "${INTERNVL_RUN}"
if ! run_stage "InternVL3.5 结果汇总" \
    python scripts/collect_vllm_results_md.py \
        --run-dir "${INTERNVL_RUN}" \
        --model "OpenGVLab/InternVL3_5-30B-A3B-HF" \
        --pruning "Unpruned" \
        --report "${REPORT}"; then :; fi

append_record "Kimi、Qwen 补跑与 InternVL 阶段均已结束；无论局部任务是否失败，继续进入三模型 scores 阶段。"
