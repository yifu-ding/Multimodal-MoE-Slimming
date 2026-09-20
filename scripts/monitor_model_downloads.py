"""Lightweight download supervision; never restart or kill download workers."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
MODELS = [('mistral', 'Mistral-Small-4-119B-2603'),
          ('qwen235', 'Qwen3-VL-235B-A22B-Instruct-FP8')]


def progress(root):
    info = json.loads((root / 'mirror_manifest.json').read_text())
    weights = [x for x in info['siblings'] if x['rfilename'].endswith('.safetensors')
               and not x['rfilename'].startswith('consolidated')]
    total = done = count = 0
    for item in weights:
        size = item['lfs']['size']
        total += size
        target = root / item['rfilename']
        if target.exists() and target.stat().st_size == size:
            done += size
            count += 1
        else:
            partial = target.with_name(target.name + '.mirror-partial')
            if partial.exists():
                done += min(size, partial.stat().st_size)
    status = root / 'download_status.md'
    lines = status.read_text().splitlines() if status.exists() else []
    last = lines[-1] if lines else 'no status'
    complete = count == len(weights) and bool(weights) and 'DOWNLOAD_COMPLETE' in last
    return done, total, count, len(weights), last, complete


def inspect(state):
    now = time.time()
    result = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'],
                            capture_output=True, text=True, timeout=15)
    sessions = set(result.stdout.splitlines()) if result.returncode == 0 else None
    for key, name in MODELS:
        try:
            done, total, count, num, last, complete = progress(Path('/home/data3/dyf/models') / name)
            previous = state.get(key, {})
            changed = now if done != previous.get('bytes') else previous.get('changed', now)
            elapsed = now - previous.get('time', now)
            speed = (done - previous.get('bytes', done)) / elapsed if elapsed > 0 else 0
            status = 'DONE' if complete else ('UNKNOWN：会话检查失败' if sessions is None else
                     'ATTENTION：下载会话已停止' if f'maes-download-{key}' not in sessions else
                     'WARNING：一小时无新增字节，可能重试/校验中，需诊断' if now - changed >= 3600 else '运行中')
            eta = f'{(total-done)/speed/3600:.1f}h' if speed > 0 else '待估计'
            message = (f'{name}：{status}；权重 {done/2**30:.2f}/{total/2**30:.2f} GiB '
                       f'({done/total:.1%})，完整分片 {count}/{num}；区间速度 {max(speed,0)/2**20:.2f} MiB/s，'
                       f'ETA {eta}；最近事件：{last}')
            state[key] = {'bytes': done, 'changed': changed, 'time': now}
        except Exception as exc:
            message = f'{name}：UNKNOWN，检查失败：{exc}'
        line = f'- {time.strftime("%F %T %Z")} 下载巡检：{message}\n'
        print(line, end='', flush=True)
        with (REPO / 'docs/自动化执行结果.md').open('a') as handle:
            handle.write('\n' + line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    folder = REPO / 'artifacts/download-monitor'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        while True:
            inspect(state)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(state))
            temp.replace(path)
            if args.once:
                return
            time.sleep(1800)


if __name__ == '__main__':
    main()
