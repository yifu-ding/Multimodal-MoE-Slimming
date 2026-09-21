#!/usr/bin/env bash
set -euo pipefail
cd /home/dyf/code/distill/MAES
flow=.automation/qwen_p30_video
run_dir="$PWD/results/vllm_ours/qwen3-vl-30b-a3b/ep4-p30-full"
plan="$PWD/runtime/ep4_plans/qwen3-vl-30b-a3b-p30-sparse-tier-v2.pt"
report="$PWD/$flow/STATUS.md"
record() { printf '\n- %s CST: %s\n' "$(date '+%F %H:%M')" "$1" >> "$report"; }
task_complete() {
    [[ -s "$run_dir/status/$1.complete" ]] &&
    TASK="$1" ROOT="$run_dir" python - <<'PY'
import json, os, sys
from pathlib import Path
p=Path(os.environ['ROOT'])/'tasks'/os.environ['TASK']
files=list(p.rglob('*_results.json'))
sys.exit(0 if files and json.loads(max(files,key=lambda x:x.stat().st_mtime).read_text()).get('results') else 1)
PY
}
if bash "$flow/completion_check.sh"; then
    record 'DONE: 两项 benchmark 和 VideoMMMU Judge 已校验。'
    exit 0
fi
if [[ ! -f "$plan" ]]; then record "ATTENTION: EP4 plan 缺失：$plan"; exit 2; fi
for task in videomme_qwen3_vllm video_mmmu_local; do
    if task_complete "$task"; then record "$task 已完成，跳过。"; continue; fi
    if ! bash "$flow/resource_check.sh"; then record "WARNING: GPU 被占用，暂不运行 $task。"; exit 3; fi
    record "开始 $task；保留现有断点缓存。"
    if ! env DECORD_EOF_RETRY_MAX=20480 CUDA_VISIBLE_DEVICES=0,1,2,3 \
        GPU_MEMORY_UTILIZATION=0.70 EP4_PLAN="$plan" \
        MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct PRUNING_LABEL=ours_p30 \
        TASKS="$task" TASK_MAX_ATTEMPTS=1 FAIL_FAST=1 RUN_DIR="$run_dir" \
        bash scripts/run_vllm_ep4_pruned.sh; then
        record "ATTENTION: $task 失败，检查本轮 logs 后修复；保留缓存。"
        exit 1
    fi
    if ! task_complete "$task"; then record "ATTENTION: $task 结果校验失败。"; exit 1; fi
    record "$task 已完成并通过结果校验。"
done
if ! bash "$flow/completion_check.sh"; then
    record '开始 VideoMMMU Judge。'
    if ! env PREDICTIONS_DIR="$run_dir" TASKS=videommmu JUDGE_MAX_ATTEMPTS=1 PORT=8010 CUDA_VISIBLE_DEVICES=0 \
        bash scripts/run_vllm_judge_stage.sh; then
        record 'ATTENTION: VideoMMMU Judge 失败，保留预测与已有评分。'
        exit 1
    fi
fi
if bash "$flow/completion_check.sh"; then
    record 'DONE: VideoMME、VideoMMMU 与 VideoMMMU Judge 均通过校验。'
else
    record 'ATTENTION: 运行结束但最终结果校验未通过。'
    exit 1
fi
