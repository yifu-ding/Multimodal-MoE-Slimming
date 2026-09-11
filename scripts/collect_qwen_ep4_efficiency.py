#!/usr/bin/env python3
"""Merge Qwen EP4 batch and GPU traces into plot-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median

STRATEGIES = ("padded", "multi_kernel", "cross_layer")


def parse_timestamp(value: str) -> float:
    value = value.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    raise ValueError(f"unsupported nvidia-smi timestamp: {value!r}")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def read_gpu_trace(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "timestamp": parse_timestamp(row["timestamp"]),
                    "gpu_index": int(row["gpu_index"]),
                    "memory_used_mib": float(row["memory_used_mib"]),
                    "memory_total_mib": float(row["memory_total_mib"]),
                    "gpu_utilization_percent": float(row["gpu_utilization_percent"]),
                    "power_watts": float(row["power_watts"]),
                }
            )
    return rows


def interval_gpu_stats(gpu_rows: list[dict], start: float, end: float) -> dict:
    selected = [row for row in gpu_rows if start <= row["timestamp"] <= end]
    if not selected:
        selected = sorted(gpu_rows, key=lambda row: abs(row["timestamp"] - end))[:4]
    per_rank_peak = []
    per_rank_mean = []
    for rank in range(4):
        rank_rows = [row for row in selected if row["gpu_index"] == rank]
        per_rank_peak.append(max(row["memory_used_mib"] for row in rank_rows))
        per_rank_mean.append(
            sum(row["memory_used_mib"] for row in rank_rows) / len(rank_rows)
        )
    return {
        "gpu_memory_mean_mib": sum(per_rank_mean) / 4,
        "gpu_memory_peak_max_mib": max(per_rank_peak),
        "gpu_memory_peak_min_mib": min(per_rank_peak),
        "gpu_memory_peak_imbalance_mib": max(per_rank_peak) - min(per_rank_peak),
        "gpu_utilization_mean_percent": sum(
            row["gpu_utilization_percent"] for row in selected
        )
        / len(selected),
        "gpu_power_mean_watts": sum(row["power_watts"] for row in selected)
        / len(selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    batch_output = []
    gpu_output = []
    summaries = []

    available_strategies = [
        strategy
        for strategy in STRATEGIES
        if (root / strategy / "batch_trace.jsonl").is_file()
        and (root / strategy / "gpu_trace.csv").is_file()
    ]
    if not available_strategies:
        raise SystemExit(f"no strategy traces found under {root}")

    for strategy in available_strategies:
        strategy_dir = root / strategy
        batch_path = strategy_dir / "batch_trace.jsonl"
        gpu_path = strategy_dir / "gpu_trace.csv"
        if not batch_path.is_file() or not gpu_path.is_file():
            raise SystemExit(f"missing traces for {strategy}: {strategy_dir}")
        batches = [
            json.loads(line)
            for line in batch_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        measured = [row for row in batches if not row.get("is_warmup", False)]
        if not measured:
            raise SystemExit(f"no measured batches for {strategy}")
        gpu_rows = read_gpu_trace(gpu_path)
        measure_start = measured[0]["wall_time_unix"] - measured[0]["batch_seconds"]
        completed = 0
        total_input = 0
        total_output = 0
        total_batch_seconds = 0.0
        for measured_index, row in enumerate(measured):
            completed += int(row["batch_requests"])
            total_input += int(row["batch_input_tokens"])
            total_output += int(row["batch_output_tokens"])
            total_batch_seconds += float(row["batch_seconds"])
            stats = interval_gpu_stats(
                gpu_rows,
                row["wall_time_unix"] - row["batch_seconds"],
                row["wall_time_unix"],
            )
            batch_output.append(
                {
                    "strategy": strategy,
                    "batch_index": measured_index,
                    "completed_samples": completed,
                    "elapsed_seconds": row["wall_time_unix"] - measure_start,
                    "batch_seconds": row["batch_seconds"],
                    "batch_requests_per_second": row["batch_requests_per_second"],
                    "batch_total_tokens_per_second": row[
                        "batch_total_tokens_per_second"
                    ],
                    "cumulative_requests_per_second": completed / total_batch_seconds,
                    "cumulative_total_tokens_per_second": (
                        total_input + total_output
                    )
                    / total_batch_seconds,
                    "batch_input_tokens": row["batch_input_tokens"],
                    "batch_output_tokens": row["batch_output_tokens"],
                    **stats,
                }
            )

        inference_rows = [row for row in gpu_rows if row["timestamp"] >= measure_start]
        rank_peaks = [
            max(
                row["memory_used_mib"]
                for row in inference_rows
                if row["gpu_index"] == rank
            )
            for rank in range(4)
        ]
        batch_seconds = [float(row["batch_seconds"]) for row in measured]
        summaries.append(
            {
                "strategy": strategy,
                "measured_samples": completed,
                "measured_batches": len(measured),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_measured_seconds": total_batch_seconds,
                "requests_per_second": completed / total_batch_seconds,
                "total_tokens_per_second": (total_input + total_output)
                / total_batch_seconds,
                "batch_latency_p50_seconds": median(batch_seconds),
                "batch_latency_p95_seconds": percentile(batch_seconds, 0.95),
                "gpu0_peak_memory_mib": rank_peaks[0],
                "gpu1_peak_memory_mib": rank_peaks[1],
                "gpu2_peak_memory_mib": rank_peaks[2],
                "gpu3_peak_memory_mib": rank_peaks[3],
                "max_peak_memory_mib": max(rank_peaks),
                "mean_peak_memory_mib": sum(rank_peaks) / 4,
                "peak_memory_imbalance_mib": max(rank_peaks) - min(rank_peaks),
            }
        )
        for row in gpu_rows:
            gpu_output.append(
                {
                    "strategy": strategy,
                    "elapsed_seconds": row["timestamp"] - measure_start,
                    **{key: value for key, value in row.items() if key != "timestamp"},
                }
            )

    for name, rows in (
        ("batch_metrics.csv", batch_output),
        ("gpu_timeseries.csv", gpu_output),
        ("summary.csv", summaries),
    ):
        path = root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"root": str(root), "summary": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
