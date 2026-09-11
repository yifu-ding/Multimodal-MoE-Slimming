#!/usr/bin/env python3
"""Collect plot-ready summaries from a Qwen EP4 batch-size sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median

from collect_qwen_ep4_efficiency import percentile, read_gpu_trace

STRATEGIES = ("padded", "multi_kernel", "cross_layer")
KV_PATTERN = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
KV_GIB_PATTERN = re.compile(r"Available KV cache memory: ([0-9.]+) GiB")
MODEL_MEMORY_PATTERN = re.compile(r"Model loading took ([0-9.]+) GiB memory")


def read_batches(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_log_metrics(
    path: Path,
) -> tuple[int | None, float | None, float | None]:
    if not path.is_file():
        return None, None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    kv_matches = KV_PATTERN.findall(text)
    kv_gib_matches = KV_GIB_PATTERN.findall(text)
    model_matches = MODEL_MEMORY_PATTERN.findall(text)
    kv_tokens = int(kv_matches[0].replace(",", "")) if kv_matches else None
    kv_gib = float(kv_gib_matches[0]) if kv_gib_matches else None
    model_gib = float(model_matches[0]) if model_matches else None
    return kv_tokens, kv_gib, model_gib


def point_summary(strategy: str, batch_size: int, point: Path) -> dict:
    batches = read_batches(point / "batch_trace.jsonl")
    measured = [row for row in batches if not row.get("is_warmup", False)]
    if not measured:
        raise ValueError(f"no measured batches in {point}")
    gpu_rows = read_gpu_trace(point / "gpu_trace.csv")
    measure_start = measured[0]["wall_time_unix"] - measured[0]["batch_seconds"]
    measure_end = measured[-1]["wall_time_unix"]
    inference_rows = [
        row for row in gpu_rows if measure_start <= row["timestamp"] <= measure_end
    ]
    if not inference_rows:
        raise ValueError(f"no GPU samples in measured interval for {point}")
    rank_peaks = [
        max(
            row["memory_used_mib"]
            for row in inference_rows
            if row["gpu_index"] == rank
        )
        for rank in range(4)
    ]
    baseline_rows = [
        row
        for row in gpu_rows
        if measure_start - 2.0 <= row["timestamp"] < measure_start
    ]
    rank_baselines = []
    for rank in range(4):
        values = [
            row["memory_used_mib"]
            for row in baseline_rows
            if row["gpu_index"] == rank
        ]
        rank_baselines.append(median(values) if values else rank_peaks[rank])
    total_seconds = sum(float(row["batch_seconds"]) for row in measured)
    samples = sum(int(row["batch_requests"]) for row in measured)
    input_tokens = sum(int(row["batch_input_tokens"]) for row in measured)
    output_tokens = sum(int(row["batch_output_tokens"]) for row in measured)
    batch_latencies = [float(row["batch_seconds"]) for row in measured]
    task_logs = sorted((point / "eval" / "logs").glob("*.log"))
    kv_tokens, kv_gib, model_gib = parse_log_metrics(
        task_logs[0] if task_logs else point / "eval" / "logs" / "gqa.log"
    )
    mean_peak_mib = sum(rank_peaks) / 4
    max_peak_mib = max(rank_peaks)
    mean_baseline_mib = sum(rank_baselines) / 4
    max_activation_delta_mib = max(
        peak - baseline for peak, baseline in zip(rank_peaks, rank_baselines)
    )
    return {
        "strategy": strategy,
        "batch_size": batch_size,
        "measured_samples": samples,
        "measured_batches": len(measured),
        "total_measured_seconds": total_seconds,
        "requests_per_second": samples / total_seconds,
        "input_tokens_per_second": input_tokens / total_seconds,
        "output_tokens_per_second": output_tokens / total_seconds,
        "total_tokens_per_second": (input_tokens + output_tokens) / total_seconds,
        "batch_latency_p50_seconds": percentile(batch_latencies, 0.50),
        "batch_latency_p95_seconds": percentile(batch_latencies, 0.95),
        "gpu0_peak_memory_mib": rank_peaks[0],
        "gpu1_peak_memory_mib": rank_peaks[1],
        "gpu2_peak_memory_mib": rank_peaks[2],
        "gpu3_peak_memory_mib": rank_peaks[3],
        "mean_peak_memory_mib": mean_peak_mib,
        "max_peak_memory_mib": max_peak_mib,
        "peak_memory_imbalance_mib": max(rank_peaks) - min(rank_peaks),
        "gpu_kv_cache_tokens": kv_tokens,
        "gpu_kv_cache_gib": kv_gib,
        "mean_non_kv_peak_memory_mib": (
            mean_peak_mib - kv_gib * 1024 if kv_gib is not None else None
        ),
        "max_non_kv_peak_memory_mib": (
            max_peak_mib - kv_gib * 1024 if kv_gib is not None else None
        ),
        "pre_measure_baseline_mean_mib": mean_baseline_mib,
        "activation_workspace_delta_mean_mib": mean_peak_mib - mean_baseline_mib,
        "activation_workspace_delta_max_mib": max_activation_delta_mib,
        "rank0_model_loading_gib": model_gib,
    }


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    status_rows = read_status(root / "sweep_status.tsv")
    completed_points = {
        (row.get("strategy"), row.get("batch_size"))
        for row in status_rows
        if row.get("status") == "complete"
    }
    rows = []
    for strategy in STRATEGIES:
        strategy_dir = root / strategy
        if not strategy_dir.is_dir():
            continue
        for point in sorted(strategy_dir.glob("bs_*")):
            if not point.is_dir():
                continue
            try:
                batch_size = int(point.name.removeprefix("bs_"))
            except ValueError:
                continue
            if (strategy, str(batch_size)) not in completed_points:
                continue
            if (point / "batch_trace.jsonl").is_file() and (
                point / "gpu_trace.csv"
            ).is_file():
                rows.append(point_summary(strategy, batch_size, point))
    if not rows:
        raise SystemExit(f"no completed sweep points under {root}")
    rows.sort(key=lambda row: (STRATEGIES.index(row["strategy"]), row["batch_size"]))

    curve_path = root / "throughput_memory_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    best_rows = []
    for strategy in STRATEGIES:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        if not strategy_rows:
            continue
        best = max(strategy_rows, key=lambda row: row["requests_per_second"])
        max_stable = max(row["batch_size"] for row in strategy_rows)
        terminal = next(
            (
                row["status"]
                for row in reversed(status_rows)
                if row.get("strategy") == strategy
                and row.get("status") in {"oom", "capped"}
            ),
            "incomplete",
        )
        best_rows.append(
            {
                "strategy": strategy,
                "best_batch_size": best["batch_size"],
                "best_requests_per_second": best["requests_per_second"],
                "best_input_tokens_per_second": best["input_tokens_per_second"],
                "best_output_tokens_per_second": best["output_tokens_per_second"],
                "best_total_tokens_per_second": best["total_tokens_per_second"],
                "best_batch_latency_p95_seconds": best[
                    "batch_latency_p95_seconds"
                ],
                "best_max_peak_memory_mib": best["max_peak_memory_mib"],
                "best_max_non_kv_peak_memory_mib": best[
                    "max_non_kv_peak_memory_mib"
                ],
                "max_stable_batch_size": max_stable,
                "search_terminal_status": terminal,
                "gpu_kv_cache_tokens_at_best": best["gpu_kv_cache_tokens"],
                "gpu_kv_cache_gib_at_best": best["gpu_kv_cache_gib"],
                "rank0_model_loading_gib_at_best": best[
                    "rank0_model_loading_gib"
                ],
            }
        )
    best_path = root / "best_by_strategy.csv"
    with best_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(best_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(best_rows)
    (root / "sweep_summary.json").write_text(
        json.dumps(
            {"root": str(root), "curve": rows, "best_by_strategy": best_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"root": str(root), "best_by_strategy": best_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
