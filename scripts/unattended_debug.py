"""Fault-only Codex dispatcher. No blind restarts; persistent bounded recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from ours_campaign_state import MODELS, TASKS, run_status

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/unattended-debug"
REPORT = ROOT / "docs/自动化执行结果.md"
WORKERS = {"maes-todo-followup", "maes-todo-scores", "maes-todo-ours", "maes-todo-results"}
INTERVAL = 300  # cheap liveness check; Markdown progress remains every 30 minutes
SCOPES = [('internvl3_5-30b-a3b', 'p30'), *[(m, 'p50') for m in MODELS]]


def command(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          timeout=120, **kwargs)


def sessions():
    result = command(["tmux", "list-sessions", "-F", "#{session_name}"])
    if result.returncode:
        raise RuntimeError("无法读取 tmux；不得据此认定 worker 停止：" + result.stderr)
    return set(result.stdout.splitlines())


def progress():
    # Only existing aggregate validators count results, not file mtimes/log growth.
    scopes = [("internvl3_5-30b-a3b", "p30"), *[(m, "p50") for m in MODELS]]
    rows = [run_status(*s) for s in scopes]
    keys = {f"{r['model']}:{r['ratio']}:{t}" for r in rows
            for t in TASKS if t not in r['missing']}
    # Judge count comes from parsed, complete, zero-failure summaries.
    keys.update(f"{r['model']}:{r['ratio']}:judge:{i}"
                for r in rows for i in range(r['judge']))
    scores = ROOT / "storage/scores/internvl3_5-30b-a3b-mixed-512/scores.pt"
    if scores.is_file():
        check = command(["/home/dyf/miniconda/envs/maes/bin/python",
                         "scripts/validate_scores_artifact.py", "--scores", str(scores),
                         "--expected-layers", "0-47"])
        if check.returncode == 0:
            keys.add("internvl:validated-full-scores")
    return sorted(keys), all(r['complete'] for r in rows)


def record(message):
    line = f"\n> [!IMPORTANT]\n> 无人值守 Debug {time.strftime('%F %T %Z')}：{message}\n"
    with REPORT.open("a") as f:
        f.write(line)
    print(line, flush=True)


def save(state):
    tmp = BASE / "state.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(BASE / "state.json")


def reconcile(state, keys):
    if set(keys) - set(state.get("progress", keys)):
        state['failures'] = 0
        state['pending'] = False
    state['progress'] = keys
    if state.get('pending'):
        state['failures'] = state.get('failures', 0) + 1
        state['pending'] = False
    return state.get('failures', 0) >= 3


def choose_scope(state, rows):
    """Only exhaust the failing model/ratio, never latch the whole campaign."""
    for row in rows:
        key = f"{row['model']}:{row['ratio']}"
        if not row['complete'] and key not in state.get('deferred', {}):
            return key
    return None


def settle_round(state, keys):
    scope = state.get('current_scope')
    if not scope:
        return
    slot = state.setdefault('scopes', {}).setdefault(scope, {})
    if reconcile(slot, [k for k in keys if k.startswith(scope + ':')]):
        state.setdefault('deferred', {})[scope] = '连续3轮无新增有效完成产物；隔离该配置，继续其他配置'


def stale_worker(active):
    """No inference for a live worker unless stage-local evidence is stale."""
    metadata = ROOT / 'artifacts/ours-active-stage.json'
    if 'maes-todo-ours' not in active or not metadata.exists():
        return False
    stage = json.loads(metadata.read_text())
    started = stage.get('started_at', time.time())
    root = ROOT / 'results/vllm_ours' / stage['model'] / f"ep4-{stage['ratio']}-full"
    files = list(root.glob('watchdog/*/*/response_cache.json'))
    files += list((root / 'logs').glob('*.log'))
    files += list((root / 'local_judge').glob('*.log'))
    files += list((root / 'local_judge').glob('*.jsonl'))
    stamps = [p.stat().st_mtime for p in files if p.is_file()]
    return time.time() - max([started, *stamps]) > 1800


def clean_env():
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
        env.pop(key, None)
    return env


def codex_args(output):
    # --approve-for-me itself selects workspace-write; CLI rejects both flags together.
    return ["codex", "exec", "--approve-for-me",
            "-c", 'forced_login_method="chatgpt"', "-c", 'model_provider="openai"',
            "--cd", str(ROOT), "--json", "--output-schema",
            str(ROOT / "scripts/unattended_debug.schema.json"),
            "--output-last-message", str(output), "-"]


def quota_error(run):
    # Only classify executor error events, not arbitrary model/tool output.
    for line in (run / 'events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') not in ('error', 'turn.failed'):
            continue
        message = json.dumps(event).lower()
        if any(word in message for word in ('usage_limit', 'usage limit', 'rate_limit', 'rate limit')):
            return True
    return False


def tick():
    statefile = BASE / "state.json"
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    # Malformed state must fail closed, never silently reset the attempt limit.
    if state.get('attention'):  # auth/permission/unknown executor failure only
        print('ATTENTION: ' + state['attention'], flush=True)
        return
    active = sessions()
    suspect = bool(active & WORKERS) and stale_worker(active)
    if active & WORKERS and not suspect:
        print('HEALTHY: worker active; no Codex invocation', flush=True)
        return
    keys, done = progress()
    if done:
        print('DONE: validated benchmark/Judge scope complete', flush=True)
        return
    # Give the waiting p50 queue the handoff; never race it with another worker.
    if not active & WORKERS and 'maes-todo-p50-queue' in active:
        print('WAIT: p50 queue owns next handoff', flush=True)
        return
    before = set(state.get('deferred', {}))
    settle_round(state, keys)
    for key in set(state.get('deferred', {})) - before:
        record(f'ATTENTION：{key} 连续3轮无进展，暂停该配置；继续其他未完成配置。')
    rows = [run_status(*s) for s in SCOPES]
    scope = choose_scope(state, rows)
    if suspect:
        stage = json.loads((ROOT / 'artifacts/ours-active-stage.json').read_text())
        scope = f"{stage['model']}:{stage['ratio']}"
        if scope in state.get('deferred', {}):
            save(state)
            print('WAIT: deferred live worker; manual ownership check needed', flush=True)
            return
    if not scope:
        save(state)
        if not state.get('exhausted_reported'):
            record('ATTENTION：其余项已完成或均已达到各自修复上限；无可继续的独立配置，巡检保留。')
            state['exhausted_reported'] = True
            save(state)
        return
    state['current_scope'] = scope
    slot = state.setdefault('scopes', {}).setdefault(scope, {})
    slot.setdefault('progress', [k for k in keys if k.startswith(scope + ':')])
    save(state)
    if time.time() < state.get('not_before', 0):
        print('WAIT: cooldown / included quota reset', flush=True)
        return
    gpu = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if gpu.returncode:
        print('WAIT: GPU status unavailable; no restart', flush=True)
        return
    auth = command(['codex', 'login', 'status'], env=clean_env())
    if auth.returncode or 'Logged in using ChatGPT' not in auth.stdout + auth.stderr:
        state['attention'] = 'ChatGPT 认证不可用；禁止 API/credits fallback。'
        save(state)
        record('ATTENTION：' + state['attention'])
        return
    run = BASE / time.strftime('%Y%m%d-%H%M%S')
    run.mkdir()
    prompt = (ROOT / 'scripts/unattended_debug_prompt.md').read_text()
    prompt += (f'\n本轮证据目录：{run}\n本轮目标配置：{scope}\n'
               f'此前该配置连续无进展轮数：{slot.get("failures", 0)}\n'
               f'已隔离配置（不得重跑）：{json.dumps(state.get("deferred", {}), ensure_ascii=False)}\n'
               f'GPU 是否仍有计算进程（先查归属，禁止抢占）：{bool(gpu.stdout.strip())}\n'
               f'是否仍有活进程但本阶段日志/心跳停滞超过30分钟：{suspect}\n')
    # Persist intent before execution: a crashed executor cannot reset its budget.
    slot['pending'] = True
    state.update(not_before=time.time() + INTERVAL, last_run=str(run))
    save(state)
    record(f'检测到停止/疑似停滞，开始诊断 {scope}；证据 {run}。')
    with (run / 'events.jsonl').open('w') as out, (run / 'stderr.log').open('w') as err:
        result = subprocess.run(codex_args(run / 'result.json'), cwd=ROOT,
                                input=prompt, text=True, stdout=out, stderr=err,
                                env=clean_env(), timeout=3600)
    try:
        outcome = json.loads((run / 'result.json').read_text())
    except (OSError, ValueError):
        outcome = {}
    status = outcome.get('status')
    if result.returncode != 0 or not status:
        slot['pending'] = False
        if quota_error(run):
            state.update(not_before=time.time() + 1800)
            outcome = {'summary': '套餐额度/速率受限，30分钟后复查；不切换付费来源'}
        else:
            # Infrastructure errors get a separate bounded retry budget.
            state['executor_errors'] = state.get('executor_errors', 0) + 1
            state['not_before'] = time.time() + 1800
            if state['executor_errors'] >= 3:
                state['attention'] = f'执行器连续3次异常，需检查 {run}；正常巡检继续。'
            outcome = {'summary': f'执行器异常 {state["executor_errors"]}/3，保留证据 {run}'}
    elif status == 'quota_wait':
        slot['pending'] = False
        state.update(not_before=time.time() + 6 * 3600)
    elif status == 'permission_blocked':
        slot['pending'] = False
        state.update(attention=outcome.get('summary', status))
    elif status == 'resource_wait':
        slot['pending'] = False
        state['not_before'] = time.time() + 1800
    if result.returncode == 0 and status:
        state['executor_errors'] = 0
    # A model's attention/failed result is a failed round, not a global latch.
    # repaired/resumed and failed count only when next stopped check finds no progress.
    save(state)
    record(f'本轮结束：{outcome.get("summary", state.get("attention", "见日志"))}；状态={status or "executor_error"}。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'dispatcher.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another debug supervisor owns the lock')
        if args.preflight:
            # An explicitly requested installation check, never a pipeline recovery.
            with (BASE / 'preflight.jsonl').open('w') as out:
                result = subprocess.run(codex_args(BASE / 'preflight-result.json'),
                    cwd=ROOT, env=clean_env(), stdout=out, stderr=subprocess.STDOUT,
                    input='安装连通性测试。不要调用工具、读写文件、运行实验或启动子代理。仅返回 JSON：status=resource_wait，summary=ChatGPT后台执行器连通性检查通过。',
                    text=True, timeout=120)
            if result.returncode:
                raise SystemExit('Codex preflight failed; inspect preflight.jsonl')
            payload = json.loads((BASE / 'preflight-result.json').read_text())
            if payload.get('status') != 'resource_wait':
                raise SystemExit('Unexpected preflight response')
            print(payload['summary'])
            return
        last_error = None
        while True:
            try:
                tick()
                last_error = None
            except Exception as exc:
                # Preserve counters and keep checking; never infer from unreadable state.
                message = f'{type(exc).__name__}: {exc}'
                if message != last_error:
                    record(f'ATTENTION：本次检查安全暂停：{message}；下一轮继续只读检查。')
                last_error = message
            if args.once:
                return
            time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
