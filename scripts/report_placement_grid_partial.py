#!/usr/bin/env python3
"""Create a validated partial report from completed placement-grid cases."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(manifest: dict[str, Any], cases_dir: Path) -> list[dict[str, Any]]:
    expected = {case["case_id"]: case for case in manifest["cases"]}
    rows = []
    for path in sorted(cases_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        case = result.get("case", {})
        case_id = case.get("case_id")
        if case_id not in expected or case != expected[case_id]:
            raise RuntimeError(f"case does not match manifest: {path}")
        if result.get("experiment_sha256") != manifest["experiment_sha256"]:
            raise RuntimeError(f"stale result: {path}")
        rows.append(result)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "prune_ratio",
        "layers",
        "m",
        "floor",
        "spread",
        "attempts",
        "retries",
        "elapsed_seconds",
        "optimality_proven",
        "stopped_reason",
        "source_kind",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            case = row["case"]
            ladder = row["ladder"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "prune_ratio": case["prune_ratio"],
                    "layers": case["layers"],
                    "m": case["m"],
                    "floor": case["arithmetic_spread_lower_bound"],
                    "spread": ladder["spread"],
                    "attempts": len(ladder["attempts"]),
                    "retries": ladder["retry_count"],
                    "elapsed_seconds": ladder["elapsed_seconds"],
                    "optimality_proven": ladder["optimality_proven"],
                    "stopped_reason": ladder["stopped_reason"],
                    "source_kind": row["source_kind"],
                }
            )


def plot_partial_heatmaps(
    output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    m_values = list(manifest["m_values"])
    layers = list(manifest["layers"])
    by_key = {
        (
            float(row["case"]["prune_ratio"]),
            int(row["case"]["layers"]),
            int(row["case"]["m"]),
        ): row
        for row in rows
    }
    outputs = []
    for ratio in manifest["prune_ratios"]:
        times = np.full((len(layers), len(m_values)), np.nan, dtype=np.float64)
        attempts = np.full_like(times, np.nan)
        proven = np.zeros_like(times, dtype=bool)
        complete = np.zeros_like(times, dtype=bool)
        for row_index, depth in enumerate(layers):
            for column_index, ep_size in enumerate(m_values):
                row = by_key.get((float(ratio), int(depth), int(ep_size)))
                if row is None:
                    continue
                ladder = row["ladder"]
                times[row_index, column_index] = max(
                    float(ladder["elapsed_seconds"]), 1e-4
                )
                attempts[row_index, column_index] = len(ladder["attempts"])
                proven[row_index, column_index] = bool(ladder["optimality_proven"])
                complete[row_index, column_index] = True

        valid_times = times[np.isfinite(times)]
        lower = float(valid_times.min())
        upper = max(float(valid_times.max()), lower * 1.001)
        time_cmap = plt.get_cmap("viridis").copy()
        attempt_cmap = plt.get_cmap("magma").copy()
        time_cmap.set_bad("#d9d9d9")
        attempt_cmap.set_bad("#d9d9d9")
        figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
        time_image = axes[0].imshow(
            np.ma.masked_invalid(times),
            aspect="auto",
            origin="lower",
            cmap=time_cmap,
            norm=LogNorm(vmin=lower, vmax=upper),
        )
        attempt_image = axes[1].imshow(
            np.ma.masked_invalid(attempts),
            aspect="auto",
            origin="lower",
            cmap=attempt_cmap,
        )
        for axis, title in zip(
            axes, ("Completed-cell wall time (s)", "Completed-cell target attempts")
        ):
            axis.set_title(title)
            axis.set_xlabel("EP size m")
            axis.set_ylabel("Depth L")
            axis.set_xticks(range(len(m_values)), labels=m_values)
            axis.set_yticks(range(len(layers)), labels=layers)
        figure.colorbar(time_image, ax=axes[0], label="seconds (log scale)")
        figure.colorbar(attempt_image, ax=axes[1], label="attempts")
        for row_index in range(len(layers)):
            for column_index in range(len(m_values)):
                if complete[row_index, column_index] and not proven[row_index, column_index]:
                    for axis in axes:
                        axis.text(
                            column_index,
                            row_index,
                            "x",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=11,
                            fontweight="bold",
                        )
        figure.suptitle(
            f"Partial Experiment C grid, prune ratio {ratio:.0%}; gray = pending, x = unproven"
        )
        path = output_dir / f"partial-heatmap-p{int(round(ratio * 100))}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    return outputs


def format_number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6g}"


def write_report(
    output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "partial-grid.csv", rows)
    figures = plot_partial_heatmaps(output_dir, manifest, rows)
    proven = [row for row in rows if row["ladder"]["optimality_proven"]]
    unproven = [row for row in rows if not row["ladder"]["optimality_proven"]]
    reused = [row for row in rows if row["source_kind"] == "reused"]
    times = [float(row["ladder"]["elapsed_seconds"]) for row in rows]
    by_key = {
        (
            float(row["case"]["prune_ratio"]),
            int(row["case"]["layers"]),
            int(row["case"]["m"]),
        ): row
        for row in rows
    }
    snapshot_unix = time.time()
    lines = [
        "# Experiment C placement-grid partial snapshot",
        "",
        "> This is an in-progress snapshot, not the final 144-cell result.",
        "",
        f"- Completed: {len(rows)}/{manifest['num_cases']} ({100.0 * len(rows) / manifest['num_cases']:.1f}%)",
        f"- Proven optimal: {len(proven)}",
        f"- Budget-ended/unproven: {len(unproven)}",
        f"- Reused E0/E0b artifacts: {len(reused)}",
        f"- Median completed-cell wall time: {statistics.median(times):.6g} s",
        f"- Maximum completed-cell wall time: {max(times):.6g} s",
        "- The 300-second HiGHS limit excludes Python model construction and solver preprocessing; total wall time can exceed 300 seconds.",
        "- E=128 is fixed. m is EP size/placement-group count; L is prefix depth.",
        "",
    ]
    max_depth = max(manifest["layers"])
    for ratio in manifest["prune_ratios"]:
        lines.extend(
            [
                f"## Available EP-size scan at L={max_depth}, p={ratio:.0%}",
                "",
                "| m | Time (s) | Spread | Attempts | Status | Source |",
                "| ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for ep_size in manifest["m_values"]:
            row = by_key.get((float(ratio), int(max_depth), int(ep_size)))
            if row is None:
                lines.append(f"| {ep_size} | - | - | - | pending | - |")
                continue
            ladder = row["ladder"]
            status = "proven optimal" if ladder["optimality_proven"] else "budget-ended/unproven"
            lines.append(
                f"| {ep_size} | {float(ladder['elapsed_seconds']):.6g} | "
                f"{format_number(ladder['spread'])} | {len(ladder['attempts'])} | "
                f"{status} | {row['source_kind']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `partial-grid.csv`: completed cells only.",
            *[f"- `{path.name}`" for path in figures],
            "- `cases/`: validated per-cell JSON copied at snapshot time.",
            "- `manifest.json`: full 144-cell experiment definition.",
            "",
        ]
    )
    (output_dir / "partial-summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (output_dir / "partial-summary.json").write_text(
        json.dumps(
            {
                "experiment": "placement-grid-partial-snapshot",
                "snapshot_unix": snapshot_unix,
                "experiment_sha256": manifest["experiment_sha256"],
                "completed": len(rows),
                "total": manifest["num_cases"],
                "proven": len(proven),
                "unproven": len(unproven),
                "reused": len(reused),
                "median_seconds": statistics.median(times),
                "max_seconds": max(times),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    rows = load_rows(manifest, args.cases_dir.resolve())
    if not rows:
        raise RuntimeError("no completed cases found")
    write_report(args.output_dir.resolve(), manifest, rows)
    print(f"wrote partial snapshot for {len(rows)}/{manifest['num_cases']} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
