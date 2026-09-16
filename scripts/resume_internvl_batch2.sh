#!/usr/bin/env bash
set -euo pipefail
cd /home/dyf/code/distill/MAES
DEBUG_DIR="${DEBUG_DIR:?Set a fresh debug directory}"
REPORT="$PWD/docs/自动化执行结果.md"
export HF_HOME=/home/data/dyf/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/maes-mpl
export SECOND_ORDER_IMPL=vectorized SECOND_ORDER_CHUNK_SIZE=2
export PATH="/home/dyf/miniconda/envs/maes/bin:$PATH"
mkdir -p "$DEBUG_DIR"
record() { printf '\n- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "$REPORT"; }
failed() {
    record "ATTENTION：InternVL batch=2 验证/恢复失败（exit=$?）；证据 $DEBUG_DIR。禁止原样重试，需继续 Debug。"
}
trap failed ERR
apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${apps//[[:space:]]/}" ]]
if [[ "${REUSE_SMOKE:-0}" != 1 ]]; then
record "开始 InternVL batch=2 Debug：原 manifest 第 105–112 条样本，四卡分别验证 L0–L3，保留原 token 配额、Flash Attention 和二阶 chunk=2。日志：$DEBUG_DIR。"
pids=()
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$rank" python scripts/debug_internvl_batch2.py "$rank" "$DEBUG_DIR/shard$rank" > "$DEBUG_DIR/shard$rank.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
(( status == 0 ))
fi
for rank in 0 1 2 3; do
    python scripts/validate_internvl_batch2_smoke.py "$DEBUG_DIR/shard$rank" "$rank"
done
python - "$DEBUG_DIR" "$REPORT" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
root = Path(sys.argv[1])
summaries = [json.loads((root / f"shard{i}/memory.json").read_text()) for i in range(4)]
with open(sys.argv[2], "a") as f:
    f.write(f"\n- {datetime.now():%F %H:%M} CST：InternVL batch=2 Debug 验证通过：L0–L3 各 8 条原失败样本、4 次反向传播成功，scores 完整性检查通过，decoder Flash Attention 2 已确认。\n")
    for s in summaries:
        f.write(f"  - L{s['layer']}：peak allocated={s['peak_allocated_gib']:.2f} GiB，peak reserved={s['peak_reserved_gib']:.2f} GiB，总显存={s['total_gib']:.2f} GiB。\n")
PY
# Retain failed-run evidence before the existing dispatcher reopens shard logs.
if [[ "${ARCHIVE_INVALID_SCORES:-0}" == 1 ]]; then
    old_dir="$PWD/storage/scores/internvl3_5-30b-a3b-mixed-512"
    archive_dir="${old_dir}.invalid-modality-$(basename "$DEBUG_DIR")"
    [[ ! -e "$archive_dir" ]]
    [[ ! -e runtime/ep4_plans/internvl3_5-30b-a3b-p30-sparse-tier-v2.pt ]]
    [[ ! -e runtime/ep4_plans/internvl3_5-30b-a3b-p50-sparse-tier-v2.pt ]]
    mv "$old_dir" "$archive_dir"
    mkdir -p "$old_dir/logs"
    record "旧 InternVL Scores（模态计数无效）已完整保留至 $archive_dir；新跑使用原目录，避免复用旧 shard。"
fi
mkdir -p "$DEBUG_DIR/previous-full-logs"
cp -a storage/scores/internvl3_5-30b-a3b-mixed-512/logs/. "$DEBUG_DIR/previous-full-logs/"
record "本轮 Debug 验证通过，恢复 InternVL 全量 Scores：batch=2、原 512 样本/262144 score tokens、48 层；之后生成 EP4 Plans 并接力 p30。"
SCORE_MODELS=internvl3_5-30b-a3b SCORE_BATCH_SIZE=2 SCORE_SECOND_ORDER_CHUNK_SIZE=2 \
    bash scripts/monitor_todo_scores.sh
python scripts/validate_scores_artifact.py \
    --scores storage/scores/internvl3_5-30b-a3b-mixed-512/scores.pt --expected-layers 0-47
test -s runtime/ep4_plans/internvl3_5-30b-a3b-p30-sparse-tier-v2.pt
tmux rename-session maes-todo-ours
record "InternVL 全量 Scores 校验通过，接力 p30 Ours 评测。"
OURS_MODELS=internvl3_5-30b-a3b OURS_GPU_MEMORY_UTILIZATION=0.85 RUN_P50=0 TASK_MAX_ATTEMPTS=1 \
    bash scripts/monitor_todo_ours.sh
record "InternVL p30 接力器退出；最终完成状态由原监督器依据产物检查。"
