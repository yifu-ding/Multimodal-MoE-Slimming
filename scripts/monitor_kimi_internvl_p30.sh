#!/usr/bin/env bash
set -u

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"

append_record() {
    printf -- '- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "${REPORT}"
}

cd "${REPO_ROOT}"
append_record "Kimi/InternVL p30 串行接力启动：Kimi 全量 -> InternVL Scores/新版 EP4 Plans -> InternVL 全量；p50 继续暂缓。"

env OURS_MODELS=kimi OURS_GPU_MEMORY_UTILIZATION=0.85 RUN_P50=0 \
    bash scripts/monitor_todo_ours.sh
kimi_status=$?
append_record "Kimi p30 Ours 阶段退出，exit_code=${kimi_status}；继续 InternVL Scores。"

tmux rename-session maes-todo-scores 2>/dev/null || true
env SCORE_MODELS=internvl3_5-30b-a3b SCORE_BATCH_SIZE=8 \
    SCORE_SECOND_ORDER_CHUNK_SIZE=2 \
    bash scripts/monitor_todo_scores.sh
scores_status=$?
append_record "InternVL Scores/EP4 Plans 阶段退出，exit_code=${scores_status}；继续 InternVL p30 评测。"

tmux rename-session maes-todo-ours 2>/dev/null || true
env OURS_MODELS=internvl3_5-30b-a3b OURS_GPU_MEMORY_UTILIZATION=0.85 RUN_P50=0 \
    bash scripts/monitor_todo_ours.sh
internvl_status=$?
append_record "InternVL p30 Ours 阶段退出，exit_code=${internvl_status}；本轮串行接力结束。"

if (( kimi_status != 0 || scores_status != 0 || internvl_status != 0 )); then
    exit 1
fi
