---
name: background-todo-runner
description: Execute a user-supplied TODO as a resumable background pipeline, monitor it in tmux, recover stopped work, debug repeated failures, and maintain a readable Markdown status report. Use when long-running experiments or batch jobs must continue while the foreground Codex remains available.
---

# Background TODO Runner

Turn a TODO into an idempotent background workflow. Keep the foreground Codex free after launch; do not run the long job in a blocking tool session.

## Required Inputs

Resolve these from the request or repository, asking only when a missing value changes scope:

- TODO Markdown path.
- Markdown report path.
- Workspace root.
- Authorized task scope, priority, resource limits, and retry policy.

Treat the TODO as requested scope, not proof of completion. Determine completion from concrete artifacts, validated outputs, and current process state. Preserve existing user work and never rerun a valid completed task.

## Build The Runner

Inspect the TODO, existing report, current tmux sessions, logs, and artifacts before launching anything. Convert the authorized unfinished items into a deterministic serial dispatcher, normally under `.automation/` in the target repository:

- `pipeline.sh`: runs the next unfinished item and skips validated completed items.
- `completion_check.sh`: returns `0` only when all authorized work is complete, `1` while work remains, and `2` when state is invalid or needs attention.
- `progress.sh`: prints one concise, single-line artifact summary.
- `resource_check.sh`: returns `0` only when the required GPUs, ports, disk, or other exclusive resources are available.

Make each stage resumable. Use stable run directories, atomic completion markers, response caches or checkpoints where supported, and bounded per-task retries. A completion marker is valid only when its signature/configuration matches the requested run and its result artifact passes a lightweight integrity check.

Do not encode task truth only in Markdown checkboxes. Do not delete failed outputs or overwrite useful caches merely to retry. Do not start deferred work such as a lower-priority pruning ratio unless it is explicitly in the authorized scope.

## Launch In Tmux

Use distinct, predictable names for the worker and supervisor. Start both detached, verify both panes remain alive, then return control to the user. Never attach the foreground session to a long job.

Use the bundled [scripts/tmux_supervisor.sh](scripts/tmux_supervisor.sh) with absolute paths. Provide these environment variables when starting the supervisor session:

```text
WORKSPACE, TODO_FILE, REPORT_FILE, PIPELINE_SESSION, PIPELINE_SCRIPT,
COMPLETION_CHECK_SCRIPT, PROGRESS_SCRIPT, RESOURCE_CHECK_SCRIPT
```

Set `INTERVAL_SECONDS=5400` for the default 90-minute inspection cadence. The supervisor accepts `COOLDOWN_SECONDS`, `MAX_RESTARTS`, `STATE_DIR`, and `PIPELINE_LOG` overrides.

Adapt quoting to the shell, but do not interpolate untrusted TODO text as shell code. Verify with `tmux list-panes`, the latest heartbeat/log timestamp, and one real progress update.

## Monitor And Recover

Default to a 90-minute interval unless the user specifies another interval. Each inspection must record a readable line containing the timestamp, stage/task, completed/total, percentage, elapsed time, ETA when defensible, and resource summary.

When the worker session disappears:

1. Run the completion checker and inspect concrete outputs.
2. If complete, record completion and keep the supervisor alive for later TODO changes.
3. If work remains, diagnose the failure, apply a scoped fix and validate it before resuming the same idempotent dispatcher when resources are available. Never rerun an unchanged failed command without evidence that its cause has been resolved.
4. Apply a cooldown and stop after three consecutive Debug-and-recovery rounds with no newly validated completed artifacts. Log growth, elapsed time and a live process are not artifact progress.
5. Write an `ATTENTION`/`WARNING` block to the report with the failed stage, evidence paths, restart count, and next action.

Do not automatically kill a live but stale process. Diagnose ownership and logs first. Never compete with unrelated resource users.

## Debug Loop

For a repeated failure, inspect the smallest relevant log, reproduce with a cheap smoke or one sample, identify the root cause, make a scoped fix, verify it, and resume only unfinished work. Record the cause, fix, validation, and recovery in the report. Skip an irreducible task only when the user authorized skipping or after the configured retry limit, and mark it explicitly rather than presenting the pipeline as complete.

A shell supervisor can restart known commands but cannot reason about a new bug. For unattended reasoning, create a Codex scheduled task at the same 90-minute cadence when that product capability is available. Its prompt should invoke this skill, name the TODO/report paths, request diagnosis of the current blocker, and forbid rerunning completed work. Do not launch recursive `codex exec` agents or grant unattended mutation privileges unless the user explicitly authorizes that execution model.

## Unattended Codex Debug (Explicit Opt-in)

After the user authorizes an independent background Codex executor, use a separate detached supervisor with a single-instance lock and persistent state. Preserve the agreed inspection cadence (MAES currently uses 30 minutes); healthy checks must be shell-only, without invoking Codex. The bundled shell supervisor defaults to monitoring only (`AUTO_RESTART=0`); do not enable blind restarts alongside a reasoning executor.

Only invoke Codex after confirming that the worker has stopped, authorized work remains, and exclusive resources are available. Treat failed status checks as unknown, not as proof of a stopped worker. For live-but-stale workers, report the evidence; do not automatically kill them.

Each round must inspect the latest failure and prior attempts, identify a cause, make a minimal fix, run a small smoke with semantic artifact checks, then resume only unfinished work in detached tmux. Successful exit and finite numbers alone are insufficient: verify expected coverage, meaningful data and downstream loadability. Save diagnosis, changes, validation and recovery evidence in the existing Markdown report and per-round logs.

Persist a pending-round marker before invoking Codex. Count each unsuccessful round once; stop at three consecutive rounds without new validated completed artifacts and mark `ATTENTION`. Reset only on new validated artifacts, not on another launch. Authentication, permission, quota and resource waits are not Debug failures. Corrupt state or unknown executor failures must fail closed, preserve evidence and request attention rather than reset counters. Keep ordinary monitoring independent so it can continue.

Use the current CLI's supported workspace-write and approval controls; do not bypass sandboxing. On CLI versions offering `--approve-for-me`, it already selects workspace-write and must not be combined with `--sandbox`. Scope the prompt to the authorized repository/tasks, preserve existing edits and artifacts, and forbid nested agents, changes to recovery counters/policies, or unrelated operations. Verify CLI compatibility and a small connectivity check before enabling the supervisor; do not claim recovery is proven merely because connectivity passed.

### Included Allowance Only — No Extra Credits

- Invoke Codex using the user's existing ChatGPT login and included allowance (including applicable five-hour and weekly limits). Never enable, purchase or consume extra credits, enable automatic top-ups, or fall back to API-key billing for this workflow.
- Do not modify billing settings or expose authentication secrets. Remove inherited API-key overrides from the executor environment and require ChatGPT authentication where the installed CLI supports it. This prevents API fallback, **not** use of an existing ChatGPT credit balance.
- ChatGPT login is not a credits-off switch. Before unattended use, obtain confirmation that account-side credit spending is disabled or verify an actual supported billing restriction. A prompt, a quota snapshot or a guessed CLI setting is not a spending hard limit. If this cannot be established, do not enable unattended inference.
- When included allowance is exhausted, pause inference, report the wait and resume only after allowance resets. Do not switch billing sources. Unknown quota/authentication errors should stop inference for diagnosis; script-only monitoring may continue.

### Existing MAES Implementation

In the MAES repository, reuse `scripts/unattended_debug.py`, `scripts/unattended_debug_prompt.md`, `scripts/unattended_debug.schema.json` and `tests/test_unattended_debug.py` rather than launching a second dispatcher. These are repository-specific, not bundled portable skill scripts; locate the repository and inspect its paths before using them elsewhere.

- Debug session: `maes-todo-debug`; normal monitor: `maes-todo-hourly`.
- Report: `docs/自动化执行结果.md`; state and round evidence: `artifacts/unattended-debug/`.
- `--once` performs a real inspection and can trigger authorized Debug on failure; it is not a dry run. `--preflight` invokes Codex for a small connectivity check and consumes included allowance.
- Stop only the Debug supervisor with `tmux kill-session -t maes-todo-debug`; keep the experiment worker and normal monitor running. Resolve recorded blockers before explicitly clearing an ATTENTION state.

## Reporting

Maintain one concise current-status section and append timestamped events. Use Markdown callouts for state transitions:

- `IMPORTANT`: launch or successful recovery.
- `WARNING`: stopped, stalled, resource-blocked, or retry limit reached.
- `DONE`: all authorized tasks have validated artifacts.

When the user asks for status or results, read artifacts directly and return only the requested aggregates. Include result paths in the report when reproducibility matters; do not inspect sample-level predictions unless requested.

## Handoff

Before finishing the foreground turn, report the worker and supervisor session names, TODO/report paths, current task, next task, and any unresolved warning. Ensure no required command is still attached to the foreground tool session.
