#!/usr/bin/env bash
set -euo pipefail
cd /home/dyf/code/distill/MAES
REPORT="$PWD/docs/自动化执行结果.md"
record() { printf '\n- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "$REPORT"; }
trap 'record "ATTENTION：三模型 p50 队列停止，exit=$?；检查 artifacts/ours-p50-queue.log，先 Debug 再恢复，禁止原样重试。"' ERR
ensure_plan() {
    /home/dyf/miniconda/envs/maes/bin/python scripts/ensure_ours_ep4_plan.py "$@"
}
main() {
record "已排队：当前 p30 benchmark/Judge 调用结束后，按 Qwen3-VL → Kimi → InternVL 顺序执行 p50；失败项保留待 Debug，不阻塞独立模型。"
while tmux has-session -t maes-todo-scores 2>/dev/null || tmux has-session -t maes-todo-ours 2>/dev/null; do
    sleep 60
done
if ! python scripts/ours_campaign_state.py check --model internvl3_5-30b-a3b --ratio p30; then
    record "ATTENTION：InternVL p30 仍有缺项，交给 Debug 补修；按最新授权继续独立的三模型 p50。"
fi
# Wait for unrelated GPU users without taking their resources.
while true; do
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "${apps//[[:space:]]/}" ]] && break
    sleep 60
done
tmux rename-session maes-todo-ours
for spec in \
    'qwen3-vl-30b-a3b|Qwen/Qwen3-VL-30B-A3B-Instruct|qwen3|0-47' \
    'kimi|moonshotai/Kimi-VL-A3B-Instruct|kimi|1-26' \
    'internvl3_5-30b-a3b|OpenGVLab/InternVL3_5-30B-A3B-HF|internvl3_5-30b-a3b|0-47'; do
    IFS='|' read -r tag model score_tag layers <<< "$spec"
    if python scripts/ours_campaign_state.py check --model "$tag" --ratio p50; then
        record "$tag p50 已验证完成，跳过。"
        continue
    fi
    python scripts/ours_campaign_state.py stage --model "$tag" --ratio p50 --phase EP4-plan
    if ! ensure_plan \
        --tag "$tag" --model "$model" --scores "storage/scores/$score_tag-mixed-512/scores.pt" --layers "$layers"; then
        record "ATTENTION：$tag p50 plan 失败，保留证据待 Debug，继续下一模型；禁止放宽 plan 校验。"
        continue
    fi
    record "$tag p50 plan 校验通过，开始全部 14 benchmark、Judge 和汇总。"
    if ! RUN_P50=1 ONLY_P50=1 OURS_MODELS="$tag" TASK_MAX_ATTEMPTS=1 JUDGE_MAX_ATTEMPTS=1 \
        bash scripts/monitor_todo_ours.sh; then
        record "ATTENTION：$tag p50 有失败项，保留成功结果，继续下一模型；失败项由 Debug 补修。"
    fi
done
if python scripts/ours_campaign_state.py check; then
    record "DONE：InternVL p30 及三模型 p50 benchmark/Judge 均通过完成检查。"
else
    record "ATTENTION：所有独立配置已逐项执行，仍有缺项，后台 Debug 将按配置补修；不宣称全部完成。"
    return 1
fi
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
