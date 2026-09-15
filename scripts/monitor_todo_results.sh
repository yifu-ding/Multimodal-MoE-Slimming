#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"
COLLECTOR="${REPO_ROOT}/scripts/collect_vllm_results_md.py"
QWEN_VIDEO="${REPO_ROOT}/results/vllm_baseline/qwen3-vl-30b-a3b/ep4-videomme-native-video"
QWEN_FULL="${REPO_ROOT}/results/vllm_baseline/qwen3-vl-30b-a3b/ep4-full-no-mvbench"
KIMI_FULL="${REPO_ROOT}/results/vllm_baseline/kimi-vl-30b-a3b/ep4-full-all"
INTERNVL_FULL="${REPO_ROOT}/results/vllm_baseline/internvl3_5-30b-a3b-hf/ep4-full-all"

latest_invocation_epoch() {
    local run_dir="$1"
    local latest=""
    latest="$(
        find "${run_dir}" -maxdepth 1 -type f -name 'run_config*.txt' -printf '%T@\n' 2>/dev/null \
            | sort -nr \
            | head -n 1 \
            | cut -d. -f1
    )" || true
    printf '%s\n' "${latest:-0}"
}

marker_is_current() {
    local marker="$1"
    local invocation_epoch="$2"
    local marker_epoch
    [[ -f "${marker}" ]] || return 1
    marker_epoch="$(stat -c %Y "${marker}")"
    (( marker_epoch >= invocation_epoch ))
}

wait_for_tasks() {
    local run_dir="$1"
    local owner_session="$2"
    local invocation_epoch="$3"
    shift 3
    while :; do
        local missing=0
        local failed=0
        local task
        for task in "$@"; do
            if [[ -f "${run_dir}/status/${task}.complete" ]]; then
                continue
            fi
            if marker_is_current "${run_dir}/status/${task}.failed" "${invocation_epoch}"; then
                failed=1
                continue
            fi
            missing=1
        done
        if (( missing == 0 )); then
            (( failed == 0 ))
            return
        fi
        if ! tmux has-session -t "${owner_session}" 2>/dev/null; then
            printf -- '- %s CST：等待 %s marker 时执行器 %s 已退出；记录现有结果并跳过缺失项。\n' \
                "$(date '+%F %H:%M')" "${run_dir}" "${owner_session}" >> "${REPORT}"
            return 2
        fi
        sleep 60
    done
}

append_results() {
    local run_dir="$1"
    if ! python "${COLLECTOR}" --run-dir "${run_dir}" --report "${REPORT}"; then
        printf -- '- %s CST：结果汇总失败：%s。\n' "$(date '+%F %H:%M')" "${run_dir}" >> "${REPORT}"
        return 1
    fi
}

append_failures() {
    local run_dir="$1"
    local invocation_epoch="$2"
    local marker task exit_code log_file reason
    shopt -s nullglob
    for marker in "${run_dir}"/status/*.failed; do
        marker_is_current "${marker}" "${invocation_epoch}" || continue
        task="$(awk -F= '$1 == "task" { print $2; exit }' "${marker}")"
        exit_code="$(awk -F= '$1 == "exit_code" { print $2; exit }' "${marker}")"
        log_file="$(awk -F= '$1 == "log" { sub(/^log=/, ""); print; exit }' "${marker}")"
        reason=""
        if [[ -n "${log_file}" && -f "${log_file}" ]]; then
            reason="$(tr '\r' '\n' < "${log_file}" | sed $'s/\033\\[[0-9;]*m//g' | grep -E 'Error during evaluation:|AssertionError:|RuntimeError:|ValueError:|CUDA out of memory|Killed|error:' | tail -n 1 || true)"
        fi
        [[ -n "${reason}" ]] || reason="详见 ${log_file:-${marker}}"
        printf -- '- %s CST：%s 中数据集 %s 在一次重试后仍失败（exit_code=%s）：%s；已跳过。\n' \
            "$(date '+%F %H:%M')" "${run_dir}" "${task:-unknown}" "${exit_code:-unknown}" "${reason}" >> "${REPORT}"
    done
    shopt -u nullglob
}

QWEN_VIDEO_EPOCH="$(latest_invocation_epoch "${QWEN_VIDEO}")"
QWEN_FULL_EPOCH="$(latest_invocation_epoch "${QWEN_FULL}")"

# Video-MME has already exhausted its allowed retry and is documented in the
# report. Do not append the same terminal failure again when this monitor is
# restarted to pick up the remaining Qwen tasks.
if ! grep -q 'Video-MME 最终结论' "${REPORT}"; then
    if wait_for_tasks "${QWEN_VIDEO}" maes-todo-auto-v4 "${QWEN_VIDEO_EPOCH:-0}" videomme_qwen3_vllm; then
        append_results "${QWEN_VIDEO}"
    fi
    append_failures "${QWEN_VIDEO}" "${QWEN_VIDEO_EPOCH:-0}"
fi

if wait_for_tasks "${QWEN_FULL}" maes-todo-auto-v4 "${QWEN_FULL_EPOCH:-0}" egoschema_subset_local mvbench_available_3800; then :; fi
# A failed task must not hide metrics produced by its successful siblings.
append_results "${QWEN_FULL}" || true
append_failures "${QWEN_FULL}" "${QWEN_FULL_EPOCH:-0}"

while :; do
    if [[ -f "${KIMI_FULL}/run_summary.txt" ]] && ! tmux has-session -t maes-todo-followup 2>/dev/null; then
        append_results "${KIMI_FULL}" || true
        append_failures "${KIMI_FULL}" "$(latest_invocation_epoch "${KIMI_FULL}")"
        break
    fi
    if ! tmux has-session -t maes-todo-followup 2>/dev/null; then
        printf -- '- %s CST：Kimi run summary 缺失且 followup 已退出；跳过结果汇总。\n' "$(date '+%F %H:%M')" >> "${REPORT}"
        append_failures "${KIMI_FULL}" "$(latest_invocation_epoch "${KIMI_FULL}")"
        break
    fi
    sleep 60
done

while :; do
    if [[ -f "${INTERNVL_FULL}/run_summary.txt" ]] && ! tmux has-session -t maes-todo-followup 2>/dev/null; then
        append_results "${INTERNVL_FULL}" || true
        append_failures "${INTERNVL_FULL}" "$(latest_invocation_epoch "${INTERNVL_FULL}")"
        break
    fi
    if ! tmux has-session -t maes-todo-followup 2>/dev/null; then
        printf -- '- %s CST：InternVL run summary 缺失且 followup 已退出；跳过结果汇总。\n' "$(date '+%F %H:%M')" >> "${REPORT}"
        append_failures "${INTERNVL_FULL}" "$(latest_invocation_epoch "${INTERNVL_FULL}")"
        break
    fi
    sleep 60
done
