#!/usr/bin/env python3
"""Collect lmms-eval result JSON files into the TODO Markdown schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HEADER = (
    "| Model | Pruning | Task | n-shot | Metric | Value | Stderr | Evaluation time | "
    "total_gen_tokens (tokens) | avg_speed (tokens/s) | avg_tpot (seconds/token) | avg_ttft (seconds) |"
)
SEPARATOR = "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pruning", default="Unpruned")
    parser.add_argument("--model", default=None, help="Override the model name stored by lmms-eval.")
    parser.add_argument("--report", type=Path, default=None, help="Append a deduplicated table to this Markdown file.")
    return parser.parse_args()


def latest_result_files(run_dir: Path) -> list[Path]:
    newest: dict[Path, Path] = {}
    for path in run_dir.glob("tasks/**/*_results.json"):
        if "submissions" in path.parts:
            continue
        task_dir = path.relative_to(run_dir / "tasks").parts[0]
        key = run_dir / "tasks" / task_dir
        current = newest.get(key)
        if current is None or path.stat().st_mtime_ns > current.stat().st_mtime_ns:
            newest[key] = path
    return [newest[key] for key in sorted(newest)]


def model_name(payload: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    model_args = str(payload.get("config", {}).get("model_args", ""))
    match = re.search(r"(?:^|,)model=([^,]+)", model_args)
    return match.group(1) if match else "unknown"


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def result_rows(path: Path, pruning: str, override_model: str | None) -> list[list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    throughput = payload.get("throughput", {})
    model = model_name(payload, override_model)
    rows: list[list[str]] = []
    for task, task_results in payload.get("results", {}).items():
        if not isinstance(task_results, dict):
            continue
        nshot = payload.get("n-shot", {}).get(task, 0)
        for raw_key, value in task_results.items():
            if raw_key == "alias" or "_stderr" in raw_key or not isinstance(value, (int, float)):
                continue
            metric, separator, filter_name = raw_key.partition(",")
            if not separator:
                continue
            if metric == "bypass":
                continue
            stderr = task_results.get(f"{metric}_stderr,{filter_name}", "N/A")
            rows.append(
                [
                    model,
                    pruning,
                    task,
                    str(nshot),
                    metric,
                    fmt(value),
                    fmt(stderr),
                    fmt(throughput.get("total_elapsed_time"), 3),
                    fmt(throughput.get("total_gen_tokens"), 0),
                    fmt(throughput.get("avg_speed"), 4),
                    fmt(throughput.get("avg_tpot"), 6),
                    fmt(throughput.get("avg_ttft"), 6),
                ]
            )
    return rows


def judge_rows(
    run_dir: Path,
    result_files: list[Path],
    pruning: str,
    override_model: str | None,
) -> list[list[str]]:
    summaries = sorted((run_dir / "local_judge").glob("*_summary.json"))
    if not summaries:
        return []
    payload_by_outer_task = {
        path.relative_to(run_dir / "tasks").parts[0]: json.loads(path.read_text(encoding="utf-8"))
        for path in result_files
    }
    task_dir_candidates = {
        "mmvet": ("mmvet",),
        "mmbench": ("mmbench_en_dev_static_local", "mmbench_en_dev"),
        "video_mmmu": ("video_mmmu_local",),
    }
    rows: list[list[str]] = []
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        judge_task = str(summary.get("task", ""))
        payload = next(
            (
                payload_by_outer_task[name]
                for name in task_dir_candidates.get(judge_task, ())
                if name in payload_by_outer_task
            ),
            {},
        )
        throughput = payload.get("throughput", {})
        model = model_name(payload, override_model)
        metrics: list[tuple[str, Any]] = [("local_judge_score", summary.get("score"))]
        if judge_task == "video_mmmu":
            metrics.extend(
                (f"local_judge_{split}_score", split_result.get("score"))
                for split, split_result in sorted(summary.get("score_by_split", {}).items())
            )
        for metric, value in metrics:
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                [
                    model,
                    pruning,
                    f"{judge_task}_judge",
                    "0",
                    metric,
                    fmt(value),
                    "N/A",
                    fmt(throughput.get("total_elapsed_time"), 3),
                    fmt(throughput.get("total_gen_tokens"), 0),
                    fmt(throughput.get("avg_speed"), 4),
                    fmt(throughput.get("avg_tpot"), 6),
                    fmt(throughput.get("avg_ttft"), 6),
                ]
            )
    return rows


def markdown_table(rows: list[list[str]]) -> str:
    lines = [HEADER, SEPARATOR]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"error: RUN_DIR does not exist: {run_dir}")
    files = latest_result_files(run_dir)
    if not files:
        raise SystemExit(f"error: no lmms-eval result JSON files under {run_dir / 'tasks'}")
    rows = [row for path in files for row in result_rows(path, args.pruning, args.model)]
    rows.extend(judge_rows(run_dir, files, args.pruning, args.model))
    if not rows:
        raise SystemExit(f"error: no metric rows found in {len(files)} result files")
    table = markdown_table(rows)
    print(table)
    if args.report:
        report = args.report.resolve()
        marker = f"<!-- vllm-results:{run_dir} -->"
        existing = report.read_text(encoding="utf-8") if report.exists() else ""
        if marker in existing:
            print(f"report already contains {run_dir}; skipped append")
        else:
            with report.open("a", encoding="utf-8") as handle:
                handle.write(f"\n{marker}\n\n### {run_dir.name}\n\n{table}\n")
            print(f"appended {len(rows)} rows to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
