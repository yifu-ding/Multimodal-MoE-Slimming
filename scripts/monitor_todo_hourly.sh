#!/usr/bin/env bash
set -u

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"
LOG="${REPO_ROOT}/results/todo_hourly_monitor.log"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
HEARTBEAT_WARNING_SECONDS="${HEARTBEAT_WARNING_SECONDS:-1800}"
ALERT_STATE_FILE="${REPO_ROOT}/results/todo_monitor_alert_state"
RECOVERY_STATE_FILE="${REPO_ROOT}/results/todo_monitor_recovery_state"
PROGRESS_STATE_FILE="${REPO_ROOT}/results/todo_monitor_progress_state"
RECOVERY_LOG="${REPO_ROOT}/results/todo_recovery_wrapper.log"
# Recovery requires a diagnosed cause, a scoped fix and validation. The shell
# monitor cannot perform that reasoning; never blindly relaunch the campaign.
AUTO_RECOVER=0
RESTART_COOLDOWN_SECONDS="${RESTART_COOLDOWN_SECONDS:-1800}"
MAX_CONSECUTIVE_RESTARTS="${MAX_CONSECUTIVE_RESTARTS:-3}"
RECOVERY_SESSION="maes-todo-ours"
KIMI_OURS_RUN="${REPO_ROOT}/results/vllm_ours/kimi/ep4-p30-full"
INTERNVL_OURS_RUN="${REPO_ROOT}/results/vllm_ours/internvl3_5-30b-a3b/ep4-p30-full"
INTERNVL_SCORES="${REPO_ROOT}/storage/scores/internvl3_5-30b-a3b-mixed-512/scores.pt"
INTERNVL_P30_PLAN="${REPO_ROOT}/runtime/ep4_plans/internvl3_5-30b-a3b-p30-sparse-tier-v2.pt"
INTERNVL_P50_PLAN="${REPO_ROOT}/runtime/ep4_plans/internvl3_5-30b-a3b-p50-sparse-tier-v2.pt"
PIPELINE_SESSIONS=(
    maes-todo-followup
    maes-todo-scores
    maes-todo-ours
    maes-todo-results
)

EXPECTED_TASKS=(
    gqa
    coco2017_cap_val_local
    textvqa_val
    chartqa
    mmstar
    mmbench_en_dev_static_local
    mmvet
    mme
    realworldqa
    videomme
    longvideobench_val_v
    video_mmmu_local
    egoschema_subset_local
    mvbench_available_3800
)

pipeline_is_active() {
    local session
    for session in "${PIPELINE_SESSIONS[@]}"; do
        if tmux has-session -t "${session}" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

active_stage() {
    if tmux has-session -t maes-todo-followup 2>/dev/null; then
        printf '%s' 'Baseline/Judge'
    elif tmux has-session -t maes-todo-scores 2>/dev/null; then
        printf '%s' 'Scores/EP4 Plans'
    elif tmux has-session -t maes-todo-ours 2>/dev/null; then
        printf '%s' 'Ours 评测'
    elif tmux has-session -t maes-todo-results 2>/dev/null; then
        printf '%s' '结果汇总'
    else
        printf '%s' '全部完成'
    fi
}

model_label() {
    local path="$1"
    case "${path}" in
        *'/kimi-vl-'*|*'/vllm_ours/kimi/'*) printf '%s' 'Kimi-VL' ;;
        *'/internvl3_5-'*) printf '%s' 'InternVL3.5' ;;
        *'/qwen3-vl-'*) printf '%s' 'Qwen3-VL' ;;
        *) printf '%s' '未知模型' ;;
    esac
}

task_label() {
    local task="$1"
    case "${task}" in
        videomme*) printf '%s' 'Video-MME' ;;
        longvideobench*) printf '%s' 'LongVideoBench' ;;
        video_mmmu*) printf '%s' 'VideoMMMU' ;;
        egoschema*) printf '%s' 'EgoSchema subset' ;;
        mvbench*) printf '%s' 'MVBench available 3800' ;;
        coco2017*) printf '%s' 'COCO Caption' ;;
        textvqa*) printf '%s' 'TextVQA' ;;
        chartqa*) printf '%s' 'ChartQA' ;;
        mmbench*) printf '%s' 'MMBench' ;;
        mmvet*) printf '%s' 'MMVet' ;;
        mmstar*) printf '%s' 'MMStar' ;;
        realworldqa*) printf '%s' 'RealWorldQA' ;;
        gqa*) printf '%s' 'GQA' ;;
        mme*) printf '%s' 'MME' ;;
        *) printf '%s' "${task}" ;;
    esac
}

format_duration() {
    local seconds="$1" hours minutes
    (( seconds < 0 )) && seconds=0
    hours=$(( seconds / 3600 ))
    minutes=$(( (seconds % 3600) / 60 ))
    if (( hours > 0 )); then
        printf '%dh%02dm' "${hours}" "${minutes}"
    elif (( minutes > 0 )); then
        printf '%dm' "${minutes}"
    else
        printf '%ds' "${seconds}"
    fi
}

session_list() {
    local session active=()
    for session in "${PIPELINE_SESSIONS[@]}"; do
        tmux has-session -t "${session}" 2>/dev/null && active+=("${session}")
    done
    local IFS=,
    printf '%s' "${active[*]:-none}"
}

latest_heartbeat() {
    find \
        "${REPO_ROOT}/results/vllm_baseline" \
        "${REPO_ROOT}/results/vllm_ours" \
        -type f -path '*/watchdog/*/response_cache.json' \
        -printf '%T@|%p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1
}

latest_pipeline_activity_epoch() {
    local file epoch latest=0
    for file in \
        "${REPO_ROOT}/results/todo_automation.log" \
        "${REPO_ROOT}/results/todo_followup.log" \
        "${REPO_ROOT}/results/todo_scores.log" \
        "${REPO_ROOT}/results/todo_ours.log" \
        "${REPO_ROOT}/results/todo_results.log"; do
        [[ -f "${file}" ]] || continue
        epoch="$(stat -c %Y "${file}" 2>/dev/null || printf '0')"
        (( epoch > latest )) && latest="${epoch}"
    done
    printf '%s' "${latest}"
}

set_alert_state() {
    printf '%s\n' "$1" > "${ALERT_STATE_FILE}"
}

emit_warning_once() {
    local key="$1"
    local message="$2"
    local now previous=""
    now="$(date '+%F %H:%M')"
    [[ -f "${ALERT_STATE_FILE}" ]] && previous="$(<"${ALERT_STATE_FILE}")"
    [[ "${previous}" == "${key}" ]] && return 0

    set_alert_state "${key}"
    printf '[%s CST] WARNING: %s\n' "${now}" "${message}" >> "${LOG}"
    printf '\n> [!WARNING]\n> **ATTENTION：自动化执行流水线需要关注（%s CST）**  \n> %s\n\n' \
        "${now}" "${message}" >> "${REPORT}"
}

emit_recovery() {
    local message="$1"
    local now
    now="$(date '+%F %H:%M')"
    printf '[%s CST] RECOVERY: %s\n' "${now}" "${message}" >> "${LOG}"
    printf '\n> [!IMPORTANT]\n> **自动恢复（%s CST）**  \n> %s\n\n' \
        "${now}" "${message}" >> "${REPORT}"
}

task_is_complete() {
    local run_dir="$1"
    local task="$2"
    local marker="${run_dir}/status/${task}.complete"
    [[ -s "${marker}" ]] && return 0

    # Video-MME uses this historical safe-task name in the shared runner.
    if [[ "${task}" == "videomme" && -s "${run_dir}/status/videomme_qwen3_vllm.complete" ]]; then
        return 0
    fi
    return 1
}

completed_task_count() {
    local run_dir="$1"
    local task completed=0
    for task in "${EXPECTED_TASKS[@]}"; do
        task_is_complete "${run_dir}" "${task}" && completed=$(( completed + 1 ))
    done
    printf '%s' "${completed}"
}

judge_is_complete() {
    local run_dir="$1"
    [[ -s "${run_dir}/local_judge/mmvet_summary.json" ]] &&
        [[ -s "${run_dir}/local_judge/mmbench_summary.json" ]] &&
        [[ -s "${run_dir}/local_judge/video_mmmu_summary.json" ]]
}

artifact_state() {
    local kimi_count internvl_count kimi_judge internvl_judge scores plans
    kimi_count="$(completed_task_count "${KIMI_OURS_RUN}")"
    internvl_count="$(completed_task_count "${INTERNVL_OURS_RUN}")"
    kimi_judge=no
    internvl_judge=no
    scores=missing
    plans=missing
    judge_is_complete "${KIMI_OURS_RUN}" && kimi_judge=yes
    judge_is_complete "${INTERNVL_OURS_RUN}" && internvl_judge=yes
    [[ -s "${INTERNVL_SCORES}" ]] && scores=ready
    if [[ -s "${INTERNVL_P30_PLAN}" && -s "${INTERNVL_P50_PLAN}" ]]; then
        plans=ready
    fi
    printf 'Kimi p30=%s/%s, Judge=%s; InternVL Scores=%s, Plans=%s, p30=%s/%s, Judge=%s' \
        "${kimi_count}" "${#EXPECTED_TASKS[@]}" "${kimi_judge}" \
        "${scores}" "${plans}" "${internvl_count}" "${#EXPECTED_TASKS[@]}" "${internvl_judge}"
    printf '; '
    python "${REPO_ROOT}/scripts/ours_campaign_state.py" summary
}

campaign_has_remaining_work() {
    local kimi_count internvl_count
    kimi_count="$(completed_task_count "${KIMI_OURS_RUN}")"
    internvl_count="$(completed_task_count "${INTERNVL_OURS_RUN}")"
    (( kimi_count < ${#EXPECTED_TASKS[@]} )) && return 0
    judge_is_complete "${KIMI_OURS_RUN}" || return 0
    [[ -s "${INTERNVL_SCORES}" ]] || return 0
    [[ -s "${INTERNVL_P30_PLAN}" ]] || return 0
    [[ -s "${INTERNVL_P50_PLAN}" ]] || return 0
    (( internvl_count < ${#EXPECTED_TASKS[@]} )) && return 0
    judge_is_complete "${INTERNVL_OURS_RUN}" || return 0
    python "${REPO_ROOT}/scripts/ours_campaign_state.py" check >/dev/null || return 0
    return 1
}

gpus_are_idle() {
    local compute_apps
    if ! compute_apps="$(nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null)"; then
        return 1
    fi
    [[ -z "$(tr -d '[:space:]' <<< "${compute_apps}")" ]]
}

read_recovery_state() {
    local attempts=0 last_restart=0
    if [[ -r "${RECOVERY_STATE_FILE}" ]]; then
        read -r attempts last_restart < "${RECOVERY_STATE_FILE}" || true
    fi
    [[ "${attempts}" =~ ^[0-9]+$ ]] || attempts=0
    [[ "${last_restart}" =~ ^[0-9]+$ ]] || last_restart=0
    printf '%s %s' "${attempts}" "${last_restart}"
}

write_recovery_state() {
    printf '%s %s\n' "$1" "$2" > "${RECOVERY_STATE_FILE}"
}

mark_pipeline_healthy() {
    local recovery_state attempts last_restart progress previous=0 path run
    progress=$(( $(completed_task_count "${KIMI_OURS_RUN}") + $(completed_task_count "${INTERNVL_OURS_RUN}") ))
    for path in "${INTERNVL_SCORES}" "${INTERNVL_P30_PLAN}" "${INTERNVL_P50_PLAN}"; do
        [[ -s "${path}" ]] && progress=$(( progress + 1 ))
    done
    for run in "${KIMI_OURS_RUN}" "${INTERNVL_OURS_RUN}"; do
        judge_is_complete "${run}" && progress=$(( progress + 1 ))
    done
    if [[ ! -r "${PROGRESS_STATE_FILE}" ]]; then
        printf '%s\n' "${progress}" > "${PROGRESS_STATE_FILE}"
        return 0
    fi
    read -r previous < "${PROGRESS_STATE_FILE}" || true
    [[ "${previous}" =~ ^[0-9]+$ ]] || previous=0
    (( progress > previous )) || return 0
    printf '%s\n' "${progress}" > "${PROGRESS_STATE_FILE}"
    recovery_state="$(read_recovery_state)"
    read -r attempts last_restart <<< "${recovery_state}"
    if (( attempts > 0 )); then
        emit_recovery "检测到新增完成产物（${previous} -> ${progress}），连续无进展恢复计数已清零；$(artifact_state)。"
        write_recovery_state 0 0
    fi
}

restart_current_campaign() {
    local now_epoch recovery_state attempts last_restart since_restart state
    now_epoch="$(date +%s)"
    state="$(artifact_state)"
    recovery_state="$(read_recovery_state)"
    read -r attempts last_restart <<< "${recovery_state}"
    since_restart=$(( now_epoch - last_restart ))

    if (( attempts >= MAX_CONSECUTIVE_RESTARTS )); then
        emit_warning_once recovery-limit \
            "自动恢复已连续尝试 ${attempts} 次且无新增完成产物，已停止重复拉起但巡检仍保持运行；当前产物：${state}。请检查 results/todo_recovery_wrapper.log、results/todo_ours.log、results/todo_scores.log 和对应任务 status/*.failed，定位并验证修复后再恢复。"
        return 1
    fi
    if (( last_restart > 0 && since_restart < RESTART_COOLDOWN_SECONDS )); then
        emit_warning_once recovery-cooldown \
            "流水线停止且仍有未完成项，但处于自动恢复冷却期（剩余约 $(format_duration "$(( RESTART_COOLDOWN_SECONDS - since_restart ))")）；当前产物：${state}。"
        return 1
    fi
    if ! gpus_are_idle; then
        emit_warning_once recovery-gpu-busy \
            "流水线停止且仍有未完成项，但检测到 GPU 上还有计算进程，暂不抢占；下次巡检重试。当前产物：${state}。"
        return 1
    fi

    attempts=$(( attempts + 1 ))
    write_recovery_state "${attempts}" "${now_epoch}"
    if tmux new-session -d -s "${RECOVERY_SESSION}" \
        "cd '${REPO_ROOT}' && exec bash scripts/monitor_kimi_internvl_p30.sh >> '${RECOVERY_LOG}' 2>&1"; then
        set_alert_state recovering
        emit_recovery "发现当前链路仍有未完成项且 GPU 空闲，已自动拉起 ${RECOVERY_SESSION}（连续尝试 ${attempts}/${MAX_CONSECUTIVE_RESTARTS}）；${state}。已完成 benchmark 由 complete marker 跳过，不会重跑；p50 评测仍暂缓。"
        return 0
    fi

    emit_warning_once recovery-launch-failed \
        "尝试拉起 ${RECOVERY_SESSION} 失败；当前产物：${state}。巡检不会退出，将在冷却后重试。"
    return 1
}

handle_pipeline_stopped() {
    local state previous="" debug_status
    debug_status="无人值守 Debug 监督会话未运行，等待诊断；巡检继续。"
    if tmux has-session -t maes-todo-debug 2>/dev/null; then
        debug_status="无人值守 Debug 已接入，状态及证据见 artifacts/unattended-debug/；如达到限制则 ATTENTION，巡检继续。"
    fi
    state="$(artifact_state)"
    [[ -r "${ALERT_STATE_FILE}" ]] && previous="$(<"${ALERT_STATE_FILE}")"
    if ! campaign_has_remaining_work; then
        if [[ "${previous}" != campaign-complete ]]; then
            printf '\n> [!IMPORTANT]\n> DONE：InternVL p30 及三个模型 p50 benchmark/Judge 产物检查通过（%s CST）；巡检仍保持运行。%s。\n' "$(date '+%F %H:%M')" "${state}" >> "${REPORT}"
            set_alert_state campaign-complete
        fi
        return 0
    fi

    if [[ "${previous}" == "healthy" || "${previous}" == "recovering" || -z "${previous}" ]]; then
        emit_warning_once pipeline-stopped \
            "自动化执行流水线已停止，但检查真实产物后仍有未完成项：${state}。禁止原命令直接重跑：先检查日志、记录原因、修复、验证，再恢复未完成项；连续 3 轮 Debug 后恢复仍无进展则停止尝试。${debug_status}"
    fi
    if [[ "${AUTO_RECOVER}" == "1" ]]; then
        restart_current_campaign || true
    fi
}

recent_failures() {
    find \
        "${REPO_ROOT}/results/vllm_baseline" \
        "${REPO_ROOT}/results/vllm_ours" \
        -type f -path '*/status/*.failed' -mmin -65 \
        -printf '%p\n' 2>/dev/null \
        | paste -sd ',' -
}

record_snapshot() {
    local now now_epoch stage sessions heartbeat heartbeat_epoch heartbeat_path
    local heartbeat_age payload completed total phase progress failures state detail
    local task task_name model task_start_file task_start_epoch elapsed percentage speed eta
    local gpu gpu_count gpu_used gpu_total gpu_util gpu_summary failure_count
    local activity_epoch activity_age warning_key="" warning_message=""
    now="$(date '+%F %H:%M')"
    now_epoch="$(date +%s)"
    stage="$(active_stage)"
    sessions="$(session_list)"
    heartbeat=""
    if [[ "${stage}" != 'Scores/EP4 Plans' ]]; then
        heartbeat="$(latest_heartbeat)"
    fi
    state="${stage} 阶段运行中"
    detail="sessions=${sessions}"

    if [[ "${stage}" == 'Scores/EP4 Plans' ]]; then
        state="$(python "${REPO_ROOT}/scripts/internvl_scores_progress.py" 2>&1)"
        detail+="; stage=${stage}; source=storage/scores/internvl3_5-30b-a3b-mixed-512/logs/shard*.log"
        if [[ "${state}" == *WARNING* ]]; then
            warning_key="scores-stalled"
            warning_message="${state}；请检查 InternVL Scores shard 日志，禁止原样重试。"
        fi
    elif [[ -f "${REPO_ROOT}/artifacts/ours-active-stage.json" ]] && tmux has-session -t maes-todo-ours 2>/dev/null; then
        state="$(python "${REPO_ROOT}/scripts/ours_active_progress.py" 2>&1)"
        detail+="; stage=${stage}; source=artifacts/ours-active-stage.json"
        if [[ "${state}" == *WARNING* ]]; then
            warning_key="ours-stalled"
            warning_message="${state}；请检查当前模型/剪枝率日志。"
        fi
    elif [[ -n "${heartbeat}" ]]; then
        heartbeat_epoch="${heartbeat%%|*}"
        heartbeat_epoch="${heartbeat_epoch%%.*}"
        heartbeat_path="${heartbeat#*|}"
        heartbeat_age=$(( now_epoch - heartbeat_epoch ))
        payload="$(tr -d '\n' < "${heartbeat_path}" 2>/dev/null)"
        completed="$(sed -n 's/.*"completed":[[:space:]]*\([0-9][0-9]*\).*/\1/p' <<< "${payload}")"
        total="$(sed -n 's/.*"total":[[:space:]]*\([0-9][0-9]*\).*/\1/p' <<< "${payload}")"
        phase="$(sed -n 's/.*"phase":[[:space:]]*"\([^"]*\)".*/\1/p' <<< "${payload}")"
        progress="${completed:-?}/${total:-?}"
        task="$(basename "$(dirname "$(dirname "${heartbeat_path}")")")"
        task_name="$(task_label "${task}")"
        model="$(model_label "${heartbeat_path}")"
        task_start_file="$(dirname "${heartbeat_path}")/rank_0.json"
        task_start_epoch="$(stat -c %Y "${task_start_file}" 2>/dev/null || printf '%s' "${heartbeat_epoch}")"
        elapsed=$(( now_epoch - task_start_epoch ))
        (( elapsed < 1 )) && elapsed=1

        percentage="$(awk -v done="${completed:-0}" -v all="${total:-0}" \
            'BEGIN { if (all > 0) printf "%.1f", done * 100 / all; else printf "0.0" }')"
        speed="$(awk -v done="${completed:-0}" -v seconds="${elapsed}" \
            'BEGIN { if (seconds > 0) printf "%.2f", done / seconds; else printf "0.00" }')"
        eta="$(awk -v done="${completed:-0}" -v all="${total:-0}" -v seconds="${elapsed}" \
            'BEGIN { if (done > 0 && all >= done) printf "%.0f", (all - done) * seconds / done; else printf "0" }')"

        state="${model} ${task_name} 进度 ${progress} (${percentage}%) | ${speed} it/s | elapsed $(format_duration "${elapsed}") | ETA $(format_duration "${eta}")"
        activity_epoch="$(latest_pipeline_activity_epoch)"
        if (( activity_epoch > 0 )); then
            activity_age=$(( now_epoch - activity_epoch ))
        else
            activity_age=-1
        fi

        if (( heartbeat_age > HEARTBEAT_WARNING_SECONDS && activity_age > HEARTBEAT_WARNING_SECONDS )); then
            state+=" | WARNING：心跳和流水线日志均已停止 $(format_duration "${heartbeat_age}")"
            warning_key="stalled:${sessions}:${heartbeat_path}"
            warning_message="自动化流水线可能卡住：${model} ${task_name} 的任务心跳已停 $(format_duration "${heartbeat_age}")，流水线日志也已超过 $(format_duration "${activity_age}") 未更新；活动会话：${sessions}。请检查 ${LOG} 和对应任务日志。"
        elif (( heartbeat_age > HEARTBEAT_WARNING_SECONDS )); then
            state+=" | 任务心跳 $(format_duration "${heartbeat_age}") 前，流水线日志仍在更新"
        else
            state+=" | 心跳 ${heartbeat_age}s 前"
        fi
        detail+="; stage=${stage}; phase=${phase:-unknown}; heartbeat_path=${heartbeat_path}"
    else
        state+=" | 暂无任务心跳"
    fi

    gpu="$(nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null)"
    if [[ -n "${gpu}" ]]; then
        read -r gpu_count gpu_used gpu_total gpu_util < <(
            awk -F, '
                { count++; used += $2; total += $3; util += $4 }
                END { printf "%d %.1f %.1f %.0f\n", count, used / count / 1024, total / count / 1024, util / count }
            ' <<< "${gpu}"
        )
        gpu_summary="GPU ${gpu_count} 卡，显存 ${gpu_used}/${gpu_total} GiB/卡，瞬时利用率 ${gpu_util}%"
    else
        gpu_summary="GPU 状态不可用"
    fi
    state+=" | ${gpu_summary}"

    failures="$(recent_failures)"
    if [[ -n "${failures}" ]]; then
        failure_count="$(tr ',' '\n' <<< "${failures}" | sed '/^$/d' | wc -l)"
        state+=" | 最近一小时失败 ${failure_count} 项"
        detail+="; recent_failures=${failures}"
    fi

    printf '[%s CST] %s\n  详情：%s\n' "${now}" "${state}" "${detail}" >> "${LOG}"
    printf -- '- %s CST：%s。\n' "${now}" "${state}" >> "${REPORT}"

    if [[ -n "${warning_message}" ]]; then
        emit_warning_once "${warning_key}" "${warning_message}"
    else
        set_alert_state healthy
    fi
}

main() {
mkdir -p "$(dirname "${LOG}")"
if [[ "${1:-}" == "--status" ]]; then
    artifact_state
    printf '\n'
    if campaign_has_remaining_work; then
        printf 'remaining_work=yes\n'
    else
        printf 'remaining_work=no\n'
    fi
    exit 0
fi

mark_pipeline_healthy
if pipeline_is_active; then
    record_snapshot
else
    handle_pipeline_stopped
fi
[[ "${1:-}" == "--once" ]] && return 0
while true; do
    sleep "${INTERVAL_SECONDS}"
    mark_pipeline_healthy
    if pipeline_is_active; then
        record_snapshot
    else
        handle_pipeline_stopped
    fi
done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
