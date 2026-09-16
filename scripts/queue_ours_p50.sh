#!/usr/bin/env bash
set -euo pipefail
cd /home/dyf/code/distill/MAES
REPORT="$PWD/docs/自动化执行结果.md"
record() { printf '\n- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "$REPORT"; }
trap 'record "ATTENTION：三模型 p50 队列停止，exit=$?；检查 artifacts/ours-p50-queue.log，先 Debug 再恢复，禁止原样重试。"' ERR
record "已排队：InternVL p30 完成后，按 Qwen3-VL → Kimi → InternVL 顺序执行 p50；缺失 EP4 plan 时从已验证 Scores 生成，完整配置跳过。"
while tmux has-session -t maes-todo-scores 2>/dev/null || tmux has-session -t maes-todo-ours 2>/dev/null; do
    sleep 60
done
python scripts/ours_campaign_state.py check --model internvl3_5-30b-a3b --ratio p30
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
    python scripts/ours_campaign_state.py stage --model "$tag" --ratio p50 --phase EP4-plan
    /home/dyf/miniconda/envs/maes/bin/python scripts/ensure_ours_ep4_plan.py \
        --tag "$tag" --model "$model" --scores "storage/scores/$score_tag-mixed-512/scores.pt" --layers "$layers"
done
record "三模型 p50 EP4 Plans 校验通过，开始全部 14 benchmark、Judge 和结果汇总。"
RUN_P50=1 ONLY_P50=1 OURS_MODELS=all TASK_MAX_ATTEMPTS=1 JUDGE_MAX_ATTEMPTS=1 \
    bash scripts/monitor_todo_ours.sh
python scripts/ours_campaign_state.py check
record "DONE：InternVL p30 及三模型 p50 benchmark/Judge 均通过完成检查。"
