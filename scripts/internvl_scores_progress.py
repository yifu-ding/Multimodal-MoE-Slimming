"""Display score-collection progress from the current shard logs, not eval caches."""
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
logs = ROOT / "storage/scores/internvl3_5-30b-a3b-mixed-512/logs"
done = set()
details = []
ages = []
fraction = 0.0
errors = []
for rank in range(4):
    path = logs / f"shard{rank}.log"
    if not path.exists():
        details.append(f"GPU{rank} 等待日志")
        continue
    text = path.read_text(errors="replace")
    ages.append(max(0, int(time.time() - path.stat().st_mtime)))
    saved = {int(v) for v in re.findall(r"\[calibration\] Layer (\d+):.*Have saved", text)}
    done.update(saved)
    matches = re.findall(r"Calibrating L(\d+):[^\r\n]*?\|\s*(\d+)/(\d+)", text)
    if matches:
        layer, batch, total = map(int, matches[-1])
        details.append(f"GPU{rank} L{layer} {batch}/{total} batch")
        if layer not in saved:
            fraction += batch / max(total, 1)
    else:
        details.append(f"GPU{rank} 加载模型/数据")
    if "Traceback (most recent call last)" in text:
        errors.append(f"shard{rank}")
try:
    pid = subprocess.check_output(["tmux", "display-message", "-p", "-t", "maes-todo-scores", "#{pane_pid}"], text=True).strip()
    elapsed = int(subprocess.check_output(["ps", "-o", "etimes=", "-p", pid], text=True).strip())
except (subprocess.SubprocessError, ValueError):
    elapsed = 0
progress = len(done) + fraction
eta = f"约 {(48-progress)*elapsed/max(progress, 1)/3600:.1f}h" if progress >= 4 and elapsed else "待估计"
label = "InternVL Scores 正在收集" if len(done) < 48 else "InternVL Scores 层收集完成，正在合并/校验/生成 EP4 Plans"
message = (f"{label} | 已保存 {len(done)}/48 层 ({len(done)/48:.1%}) | "
           + "; ".join(details) + f" | elapsed {elapsed//3600}h{elapsed%3600//60:02d}m | ETA {eta}")
if ages and len(done) < 48:
    message += f" | 最旧 shard 日志 {max(ages)}s 前"
    if max(ages) > 1800:
        message += " | WARNING：存在超过 30 分钟未更新的 shard"
if errors:
    message += " | WARNING：发现异常堆栈 " + ",".join(errors)
print(message)
