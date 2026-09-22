#!/usr/bin/env python3
"""Enrich the 43 unproven grid cells with constructive fallback placements."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_placement_ablation import checkpoint, environment_metadata
from scripts.run_placement_e0 import sha256
from scripts.run_placement_e0_followup import attach_constructive_fallback
from scripts.run_placement_grid import build_case_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("artifacts/placement-grid/cases")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/placement_grid_fallback")
    )
    return parser.parse_args()


def load_unproven(source_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(source_dir.expanduser().resolve().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload["ladder"]["optimality_proven"]:
            rows.append((path, payload))
    if len(rows) != 43:
        raise RuntimeError(f"expected 43 unproven source cells, found {len(rows)}")
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ratios = [
        float(row["ladder"]["certified_approximation_ratio_upper_bound"])
        for row in rows
    ]
    absolute_gaps = [
        float(row["ladder"]["certified_absolute_gap_upper_bound"]) for row in rows
    ]
    fallback_times = [
        float(row["ladder"]["constructive_fallback"]["elapsed_seconds"])
        for row in rows
    ]
    by_m: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_m[int(row["case"]["m"])].append(row)
    lines = [
        "# Experiment C constructive fallback validation",
        "",
        f"- Source unproven cells: {len(rows)}",
        f"- Returned constructive placements: {sum(row['ladder']['fallback_used'] for row in rows)}/{len(rows)}",
        f"- Median fallback construction time: {statistics.median(fallback_times):.6g} s",
        f"- Maximum fallback construction time: {max(fallback_times):.6g} s",
        f"- Median certified approximation-ratio upper bound: {statistics.median(ratios):.6g}x",
        f"- Maximum certified approximation-ratio upper bound: {max(ratios):.6g}x",
        f"- Median certified absolute gap upper bound: {statistics.median(absolute_gaps):.6g}",
        "- Existing MILP ladder attempts were reused byte-for-byte; they were not rerun.",
        "",
        "| m | Cases | Median G0 spread | Median lower bound | Median ratio bound | Max ratio bound | Median G0 time (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ep_size, group in sorted(by_m.items()):
        spreads = [float(row["ladder"]["spread"]) for row in group]
        bounds = [float(row["ladder"]["certified_lower_bound"]) for row in group]
        group_ratios = [
            float(row["ladder"]["certified_approximation_ratio_upper_bound"])
            for row in group
        ]
        times = [
            float(row["ladder"]["constructive_fallback"]["elapsed_seconds"])
            for row in group
        ]
        lines.append(
            f"| {ep_size} | {len(group)} | {statistics.median(spreads):.6g} | "
            f"{statistics.median(bounds):.6g} | {statistics.median(group_ratios):.6g} | "
            f"{max(group_ratios):.6g} | {statistics.median(times):.6g} |"
        )
    lines.extend(
        [
            "",
            "| Case | p | L | m | G0 spread | Certified lower bound | Absolute gap bound | Ratio bound | G0 time (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        case = row["case"]
        ladder = row["ladder"]
        lines.append(
            f"| {case['case_id']} | {case['prune_ratio']} | {case['layers']} | "
            f"{case['m']} | {ladder['spread']:.6g} | "
            f"{ladder['certified_lower_bound']:.6g} | "
            f"{ladder['certified_absolute_gap_upper_bound']:.6g} | "
            f"{ladder['certified_approximation_ratio_upper_bound']:.6g} | "
            f"{ladder['constructive_fallback']['elapsed_seconds']:.6g} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checkpoint(
        output_dir / "summary.json",
        {
            "experiment": "placement-grid-constructive-fallback",
            "num_cases": len(rows),
            "num_returned": sum(row["ladder"]["fallback_used"] for row in rows),
            "median_fallback_seconds": statistics.median(fallback_times),
            "max_fallback_seconds": max(fallback_times),
            "median_approximation_ratio_upper_bound": statistics.median(ratios),
            "max_approximation_ratio_upper_bound": max(ratios),
            "median_absolute_gap_upper_bound": statistics.median(absolute_gaps),
        },
    )


def main() -> int:
    args = parse_args()
    source_rows = load_unproven(args.source_dir)
    output_dir = args.output_dir.expanduser().resolve()
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_path, source in source_rows:
        counts, widths = build_case_tensors(source["case"])
        started = time.perf_counter()
        ladder = attach_constructive_fallback(counts, widths, source["ladder"])
        enrichment_seconds = time.perf_counter() - started
        if not ladder["fallback_used"] or ladder["spread"] is None:
            raise RuntimeError(f"fallback failed for {source['case']['case_id']}")
        result = {
            "experiment": "placement-grid-constructive-fallback-case",
            "case": source["case"],
            "source_artifact": str(source_path.relative_to(REPO_ROOT)),
            "source_sha256": sha256(source_path),
            "source_experiment_sha256": source["experiment_sha256"],
            "environment": environment_metadata(),
            "enrichment_seconds": enrichment_seconds,
            "ladder": ladder,
            "completed_at_unix": time.time(),
        }
        checkpoint(cases_dir / source_path.name, result)
        rows.append(result)
    write_summary(output_dir, rows)
    print(f"wrote constructive fallbacks for {len(rows)} cells to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
