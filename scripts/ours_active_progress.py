"""Read heartbeat only within the explicitly selected run and stage."""
import json
import time
from pathlib import Path
from ours_campaign_state import ROOT, STATE, run_status

state = json.loads(STATE.read_text())
names = {"kimi": "Kimi-VL", "qwen3-vl-30b-a3b": "Qwen3-VL", "internvl3_5-30b-a3b": "InternVL"}
model, ratio, phase = state["model"], state["ratio"], state["phase"]
elapsed = max(0, int(time.time() - state["started_at"]))
label = f"{names.get(model, model)} Ours {ratio[1:]}% {phase}"
root = Path(state["run_dir"]) if state.get("run_dir") else None
status = run_status(model, ratio)
label += f" | benchmark 已完成 {status['benchmarks']}/14，Judge {status['judge']}/3"
if phase in {"benchmark", "smoke"} and root:
    files = [p for p in root.glob("watchdog/*/*/response_cache.json") if p.stat().st_mtime >= state["started_at"]]
    if files:
        path = max(files, key=lambda p: p.stat().st_mtime)
        data = json.loads(path.read_text())
        completed, total = data.get("completed", 0), data.get("total", 0)
        age = int(time.time() - path.stat().st_mtime)
        label += f" | {path.parent.parent.name} {completed}/{total} ({completed/max(total,1):.1%}) | 心跳 {age}s 前"
        if age > 1800:
            label += " | WARNING：当前评测心跳超过 30 分钟未更新，可能处于任务切换/加载或停滞"
    else:
        label += " | 等待当前阶段心跳/模型加载"
elif phase == "Judge" and root:
    files = list((root / "local_judge").glob("*.jsonl")) + list((root / "local_judge").glob("*.log"))
    if files:
        age = int(time.time() - max(p.stat().st_mtime for p in files))
        label += f" | Judge 日志 {age}s 前"
label += f" | elapsed {elapsed//3600}h{elapsed%3600//60:02d}m | ETA 待估计"
print(label)
