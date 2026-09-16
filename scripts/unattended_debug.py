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
INTERVAL = 1800


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


def tick():
    statefile = BASE / "state.json"
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    # Malformed state must fail closed, never silently reset the attempt limit.
    if state.get('attention'):
        print('ATTENTION: ' + state['attention'], flush=True)
        return
    active = sessions()
    if active & WORKERS:
        print('HEALTHY: worker active; no Codex invocation', flush=True)
        return
    keys, done = progress()
    if done:
        print('DONE: validated benchmark/Judge scope complete', flush=True)
        return
    if reconcile(state, keys):
        state['attention'] = '连续 3 轮 Debug/恢复无新增有效完成产物；停止自动尝试。'
        save(state)
        record('ATTENTION：' + state['attention'])
        return
    save(state)
    if time.time() < state.get('not_before', 0):
        print('WAIT: cooldown / included quota reset', flush=True)
        return
    gpu = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if gpu.returncode or gpu.stdout.strip():
        print('WAIT: GPU occupied or unavailable; no restart', flush=True)
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
    prompt += f'\n本轮证据目录：{run}\n此前连续无进展轮数：{state.get("failures", 0)}\n'
    # Persist intent before execution: a crashed executor cannot reset its budget.
    state.update(pending=True, not_before=time.time() + INTERVAL, last_run=str(run))
    save(state)
    record(f'检测到 worker 停止且范围未完成，开始诊断；证据 {run}。')
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
        # Do not repeatedly charge on authentication/network/unknown executor errors.
        state.update(pending=False, attention=f'执行器异常，需检查 {run}；不盲目重复调用。')
    elif status == 'quota_wait':
        state.update(pending=False, not_before=time.time() + 6 * 3600)
    elif status in ('permission_blocked', 'attention'):
        state.update(pending=False, attention=outcome.get('summary', status))
    elif status == 'resource_wait':
        state['pending'] = False
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
        while True:
            try:
                tick()
            except Exception as exc:
                # Preserve counters; halt inference on uncertain state.
                record(f'ATTENTION：执行器安全暂停：{type(exc).__name__}: {exc}')
                raise
            if args.once:
                return
            time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
