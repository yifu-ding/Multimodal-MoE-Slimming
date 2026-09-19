#!/usr/bin/env python3
"""Export placement ablation checkpoints as flat CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def scalar_records(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if value is None or isinstance(value, (bool, int, float, str))
        }
    )
    return keys, [{key: record.get(key) for key in keys} for record in records]


def format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def table6_markdown(records: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[float, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["prune_ratio"], record["layers"]), {})[
            record["method"]
        ] = record
    lines = [
        "| p | L | Greedy m! ΔΦ/Φ̄ (%) | Greedy swap ΔΦ/Φ̄ (%) | MILP UB | MILP LB | Gap | MILP time (s) | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for (prune_ratio, layers), methods in sorted(grouped.items()):
        full = methods.get("greedy_full_bijection", {})
        swap = methods.get("greedy_pairwise_swap", {})
        milp = methods.get("milp_assignment", {})
        status = (
            "optimal" if milp.get("solver_optimal") else milp.get("solver_status", "-")
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    format_value(prune_ratio),
                    str(layers),
                    format_value(full.get("relative_spread_percent")),
                    format_value(swap.get("relative_spread_percent")),
                    format_value(milp.get("incumbent")),
                    format_value(milp.get("dual_bound")),
                    format_value(milp.get("mip_gap")),
                    format_value(milp.get("median_time_seconds")),
                    str(status),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def depth_sweep_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "| p | L | Layers | Method | DeltaPhi/Phi_bar (%) | Time (s) | UB | LB | Gap | Status |",
        "|---:|---:|:---|:---|---:|---:|---:|---:|---:|:---|",
    ]
    for record in sorted(
        records,
        key=lambda item: (
            item["prune_ratio"],
            item["layers"],
            item["layer_start"],
            item["method"],
        ),
    ):
        status = (
            "optimal"
            if record.get("solver_optimal")
            else record.get("solver_status", "-")
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    format_value(record["prune_ratio"]),
                    str(record["layers"]),
                    f"{record['layer_start']}-{record['layer_end'] - 1}",
                    record["method"],
                    format_value(record.get("relative_spread_percent")),
                    format_value(record.get("median_time_seconds")),
                    format_value(record.get("incumbent")),
                    format_value(record.get("dual_bound")),
                    format_value(record.get("mip_gap")),
                    str(status),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def m_sweep_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "| m | Method | Limit (s) | Initial ΔΦ/Φ̄ (%) | Final ΔΦ/Φ̄ (%) | Time (s) | UB | LB | Certified gap | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for record in sorted(
        records,
        key=lambda item: (
            item["m"],
            item["method"],
            -1.0 if item.get("time_limit") is None else item["time_limit"],
        ),
    ):
        status = (
            "optimal"
            if record.get("solver_optimal")
            else record.get("solver_status", "-")
        )
        elapsed = record.get("median_time_seconds", record.get("time_seconds"))
        initial = record.get("initial_relative_spread")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(record["m"]),
                    record["method"],
                    format_value(record.get("time_limit")),
                    format_value(None if initial is None else 100.0 * initial),
                    format_value(record.get("relative_spread_percent")),
                    format_value(elapsed),
                    format_value(record.get("incumbent")),
                    format_value(record.get("dual_bound")),
                    format_value(record.get("certified_relative_gap")),
                    str(status),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = payload["records"]
    csv_path = (args.csv or input_path.with_suffix(".csv")).expanduser().resolve()
    markdown_path = (
        (args.markdown or input_path.with_suffix(".md")).expanduser().resolve()
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = scalar_records(records)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if payload["experiment"] == "table6":
        markdown = table6_markdown(records)
    elif payload["experiment"] == "depth-sweep":
        markdown = depth_sweep_markdown(records)
    elif payload["experiment"] == "m-sweep":
        markdown = m_sweep_markdown(records)
    else:
        raise ValueError(f"unsupported experiment: {payload['experiment']}")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"csv={csv_path}")
    print(f"markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
