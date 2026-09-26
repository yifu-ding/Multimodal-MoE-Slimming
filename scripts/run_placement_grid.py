#!/usr/bin/env python3
"""Run the Experiment C feasibility-MILP grid over EP size and depth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_placement_ablation import checkpoint, environment_metadata, warm_up_highs
from scripts.run_placement_e0 import canonical_digest, load_plan, manifest_digest, sha256
from scripts.run_placement_e0_followup import solve_feasibility_ladder
from src.generate_mask.ep4_intplan import (
    _build_layer_placement_groups,
    _placement_spread_arithmetic_lower_bound,
)

DEFAULT_PLANS = (
    Path("storage/ep4_plans/qwen3-vl-30b-a3b-p30.pt"),
    Path("storage/ep4_plans/qwen3-vl-30b-a3b-p50.pt"),
)
DEFAULT_M_VALUES = (4, 6, 8, 12, 16, 24, 32, 48, 64)
DEFAULT_LAYERS = (4, 8, 12, 16, 24, 32, 40, 48)
DEFAULT_REUSE_DIRS = (
    Path("artifacts/placement-e0/cases"),
    Path("artifacts/placement-e0-followup/cases"),
)
DEFAULT_MANIFEST = Path("results/placement_grid/manifest.json")
DEFAULT_OUTPUT_DIR = Path("results/placement_grid/cases")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--plans", nargs="+", type=Path, default=list(DEFAULT_PLANS))
    manifest.add_argument(
        "--m-values", nargs="+", type=int, default=list(DEFAULT_M_VALUES)
    )
    manifest.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    manifest.add_argument("--num-shards", type=int, default=4)
    manifest.add_argument(
        "--reuse-dirs", nargs="+", type=Path, default=list(DEFAULT_REUSE_DIRS)
    )
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    worker.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument("--total-time-limit", type=float, default=300.0)

    report = subparsers.add_parser("report")
    report.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    report.add_argument(
        "--output", type=Path, default=Path("results/placement_grid/summary.md")
    )
    return parser.parse_args()


def implementation_metadata() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256(path)
        for path in (
            REPO_ROOT / "src/generate_mask/ep4_intplan.py",
            REPO_ROOT / "scripts/run_placement_ablation.py",
            REPO_ROOT / "scripts/run_placement_e0.py",
            REPO_ROOT / "scripts/run_placement_e0_followup.py",
            Path(__file__).resolve(),
        )
    }


def case_key(case: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(case["plan_sha256"]),
        int(case.get("layer_start", 0)),
        int(case["layers"]),
        int(case["m"]),
    )


def build_case_tensors(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    plan = load_plan(Path(case["plan"]), case["plan_sha256"])
    start = int(case.get("layer_start", 0))
    end = start + int(case["layers"])
    expert_widths = torch.as_tensor(plan["expert_widths"], dtype=torch.int64)[start:end]
    groups = _build_layer_placement_groups(
        expert_widths,
        tuple(int(value) for value in plan["active_widths"]),
        ep_size=int(case["m"]),
    )
    return groups["placement_group_counts"], groups["placement_group_widths"]


def extract_reused_ladder(payload: dict[str, Any]) -> dict[str, Any]:
    experiment = payload.get("experiment")
    if experiment == "placement-e0-followup-case":
        ladder = payload.get("ladder")
        if not isinstance(ladder, dict):
            raise ValueError("E0b result has no ladder")
        return ladder
    if experiment == "placement-e0-case":
        arm_b = payload.get("arm_b")
        if not isinstance(arm_b, dict) or arm_b.get("incumbent") is None:
            raise ValueError("E0 result has no Arm B incumbent")
        proven = bool(arm_b.get("solver_optimal"))
        return {
            "attempts": [arm_b],
            "found_feasible": True,
            "retry_count": int(bool(arm_b.get("floor_retry"))),
            "final_target_spread": float(arm_b["target_spread"]),
            "spread": float(arm_b["spread"]),
            "elapsed_seconds": float(arm_b["elapsed_seconds"]),
            "optimality_proven": proven,
            "optimality_proof": arm_b.get("optimality_proof") if proven else None,
            "stopped_reason": "feasible" if proven else "unproven",
        }
    raise ValueError(f"unsupported reusable experiment: {experiment}")


def reusable_results(paths: Iterable[Path]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    reusable: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for directory in paths:
        resolved = directory.expanduser().resolve()
        if not resolved.is_dir():
            continue
        for path in sorted(resolved.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            case = payload.get("case")
            if not isinstance(case, dict):
                continue
            ladder = extract_reused_ladder(payload)
            if not ladder.get("found_feasible") or not ladder.get("optimality_proven"):
                continue
            key = case_key(case)
            candidate = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256(path),
                "experiment": payload["experiment"],
                "spread": float(ladder["spread"]),
                "elapsed_seconds": float(ladder["elapsed_seconds"]),
            }
            current = reusable.get(key)
            if current is None or candidate["experiment"] == "placement-e0-followup-case":
                reusable[key] = candidate
    return reusable


def validate_axes(
    plans: list[dict[str, Any]], m_values: list[int], layers: list[int]
) -> None:
    if not m_values or not layers:
        raise ValueError("m-values and layers must be non-empty")
    if len(set(m_values)) != len(m_values) or len(set(layers)) != len(layers):
        raise ValueError("m-values and layers must be unique")
    for plan in plans:
        expert_widths = torch.as_tensor(plan["payload"]["expert_widths"])
        active_widths = tuple(int(value) for value in plan["payload"]["active_widths"])
        max_layers, max_experts = expert_widths.shape
        present_tiers = max(
            sum(bool((row == width).any()) for width in active_widths)
            for row in expert_widths
        )
        if min(m_values) < present_tiers:
            raise ValueError(
                f"m must be >= {present_tiers}, the maximum non-empty width-tier count"
            )
        if max(m_values) > max_experts:
            raise ValueError(f"m cannot exceed {max_experts} active expert slots")
        if min(layers) <= 0 or max(layers) > max_layers:
            raise ValueError(f"layers must be in [1, {max_layers}]")


def assign_shards(cases: list[dict[str, Any]], num_shards: int) -> list[float]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    loads = [0.0] * num_shards
    counts = [0] * num_shards
    ordered = sorted(
        cases,
        key=lambda case: (
            case.get("reuse") is None,
            int(case["layers"]) * int(case["m"]) ** 2,
            case["case_id"],
        ),
        reverse=True,
    )
    for case in ordered:
        weight = 0.0 if case.get("reuse") else float(case["layers"] * case["m"] ** 2)
        shard = min(range(num_shards), key=lambda index: (loads[index], counts[index], index))
        case["shard_id"] = shard
        loads[shard] += weight
        counts[shard] += 1
    return loads


def build_manifest(args: argparse.Namespace) -> None:
    plan_rows = []
    for path in args.plans:
        resolved = path.expanduser().resolve()
        digest = sha256(resolved)
        payload = load_plan(resolved, digest)
        plan_rows.append({"path": resolved, "sha256": digest, "payload": payload})
    m_values = sorted(int(value) for value in args.m_values)
    layers = sorted(int(value) for value in args.layers)
    validate_axes(plan_rows, m_values, layers)
    reuse = reusable_results(args.reuse_dirs)

    cases: list[dict[str, Any]] = []
    for plan_row in plan_rows:
        plan = plan_row["payload"]
        ratio = float(plan["prune_ratio"])
        ratio_tag = f"p{int(round(ratio * 100))}"
        for depth in layers:
            for ep_size in m_values:
                case = {
                    "case_id": f"grid-{ratio_tag}-l{depth:02d}-m{ep_size:02d}",
                    "plan": str(plan_row["path"]),
                    "plan_sha256": plan_row["sha256"],
                    "prune_ratio": ratio,
                    "layer_start": 0,
                    "layers": depth,
                    "m": ep_size,
                    "num_experts_per_layer": int(
                        torch.as_tensor(plan["expert_widths"]).shape[1]
                    ),
                    "active_widths": [int(value) for value in plan["active_widths"]],
                }
                counts, widths = build_case_tensors(case)
                case.update(_placement_spread_arithmetic_lower_bound(counts, widths))
                source = reuse.get(case_key(case))
                if source is not None:
                    case["reuse"] = source
                cases.append(case)

    cases.sort(key=lambda case: case["case_id"])
    shard_loads = assign_shards(cases, int(args.num_shards))
    expected = len(plan_rows) * len(m_values) * len(layers)
    if len(cases) != expected:
        raise RuntimeError(f"expected {expected} cases, found {len(cases)}")
    implementation = implementation_metadata()
    cases_sha256 = manifest_digest(cases)
    payload = {
        "experiment": "placement-grid",
        "created_at_unix": time.time(),
        "m_values": m_values,
        "layers": layers,
        "prune_ratios": sorted(float(row["payload"]["prune_ratio"]) for row in plan_rows),
        "num_shards": int(args.num_shards),
        "num_cases": len(cases),
        "num_reused": sum(case.get("reuse") is not None for case in cases),
        "estimated_shard_loads": shard_loads,
        "cases_sha256": cases_sha256,
        "implementation": implementation,
        "experiment_sha256": canonical_digest(
            {"cases_sha256": cases_sha256, "implementation": implementation}
        ),
        "cases": cases,
    }
    checkpoint(args.output.expanduser().resolve(), payload)
    print(
        f"wrote {len(cases)} grid cases ({payload['num_reused']} reused) to {args.output}"
    )


def load_validated_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if manifest_digest(manifest["cases"]) != manifest["cases_sha256"]:
        raise RuntimeError("manifest case digest mismatch")
    if implementation_metadata() != manifest["implementation"]:
        raise RuntimeError("solver implementation differs from manifest")
    expected = canonical_digest(
        {
            "cases_sha256": manifest["cases_sha256"],
            "implementation": manifest["implementation"],
        }
    )
    if expected != manifest["experiment_sha256"]:
        raise RuntimeError("manifest experiment digest mismatch")
    return manifest


def load_reused_result(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = REPO_ROOT / case["reuse"]["path"]
    if sha256(source) != case["reuse"]["sha256"]:
        raise RuntimeError(f"reusable artifact changed: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if case_key(payload["case"]) != case_key(case):
        raise RuntimeError(f"reusable artifact case mismatch: {source}")
    ladder = extract_reused_ladder(payload)
    if not ladder.get("optimality_proven"):
        raise RuntimeError(f"reusable artifact is not proven optimal: {source}")
    return ladder, payload


def run_worker(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard configuration")
    manifest = load_validated_manifest(args.manifest)
    if int(args.num_shards) != int(manifest["num_shards"]):
        raise ValueError("worker shard count differs from manifest")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warm_up_highs()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None

    for case in manifest["cases"]:
        if int(case["shard_id"]) != int(args.shard_id):
            continue
        output = output_dir / f"{case['case_id']}.json"
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("experiment_sha256") == manifest["experiment_sha256"]
                and existing.get("total_time_limit") == float(args.total_time_limit)
                and existing.get("worker_cpu_affinity") == affinity
            ):
                print(f"skip {case['case_id']}", flush=True)
                continue
            raise RuntimeError(f"stale grid output: {output}")

        started = time.perf_counter()
        if case.get("reuse") is not None:
            ladder, source_payload = load_reused_result(case)
            source_kind = "reused"
            execution_environment = source_payload.get("environment")
            execution_affinity = source_payload.get("cpu_affinity")
        else:
            counts, widths = build_case_tensors(case)
            arithmetic = _placement_spread_arithmetic_lower_bound(counts, widths)
            expected = {
                key: case[key]
                for key in (
                    "arithmetic_quantum",
                    "total_rank_weight_load",
                    "total_load_quanta",
                    "arithmetic_spread_lower_bound",
                )
            }
            if arithmetic != expected:
                raise RuntimeError(f"arithmetic metadata changed for {case['case_id']}")
            ladder = solve_feasibility_ladder(
                counts, widths, total_time_limit=float(args.total_time_limit)
            )
            source_kind = "computed"
            execution_environment = environment_metadata()
            execution_affinity = affinity

        result = {
            "experiment": "placement-grid-case",
            "cases_sha256": manifest["cases_sha256"],
            "experiment_sha256": manifest["experiment_sha256"],
            "case": case,
            "source_kind": source_kind,
            "source_artifact": case.get("reuse"),
            "execution_environment": execution_environment,
            "execution_cpu_affinity": execution_affinity,
            "worker_cpu_affinity": affinity,
            "total_time_limit": float(args.total_time_limit),
            "materialization_seconds": time.perf_counter() - started,
            "ladder": ladder,
            "completed_at_unix": time.time(),
        }
        checkpoint(output, result)
        print(
            json.dumps(
                {
                    "case_id": case["case_id"],
                    "source": source_kind,
                    "attempts": len(ladder["attempts"]),
                    "spread": ladder["spread"],
                    "seconds": ladder["elapsed_seconds"],
                    "proven": ladder["optimality_proven"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def load_rows(manifest: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for case in manifest["cases"]:
        path = output_dir / f"{case['case_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"missing grid result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("experiment_sha256") != manifest["experiment_sha256"]:
            raise RuntimeError(f"stale grid result: {path}")
        rows.append(result)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "prune_ratio",
        "layers",
        "m",
        "num_experts_per_layer",
        "arithmetic_quantum",
        "arithmetic_spread_lower_bound",
        "spread",
        "attempts",
        "retries",
        "elapsed_seconds",
        "optimality_proven",
        "stopped_reason",
        "source_kind",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
                    "num_experts_per_layer": case["num_experts_per_layer"],
                    "arithmetic_quantum": case["arithmetic_quantum"],
                    "arithmetic_spread_lower_bound": case[
                        "arithmetic_spread_lower_bound"
                    ],
                    "spread": ladder["spread"],
                    "attempts": len(ladder["attempts"]),
                    "retries": ladder["retry_count"],
                    "elapsed_seconds": ladder["elapsed_seconds"],
                    "optimality_proven": ladder["optimality_proven"],
                    "stopped_reason": ladder["stopped_reason"],
                    "source_kind": row["source_kind"],
                }
            )


def plot_heatmaps(
    output_dir: Path,
    rows: list[dict[str, Any]],
    m_values: list[int],
    layers: list[int],
    prune_ratios: list[float],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    outputs = []
    by_key = {
        (float(row["case"]["prune_ratio"]), int(row["case"]["layers"]), int(row["case"]["m"])): row
        for row in rows
    }
    for ratio in prune_ratios:
        times = np.empty((len(layers), len(m_values)), dtype=np.float64)
        attempts = np.empty_like(times)
        proven = np.empty_like(times, dtype=bool)
        for row_index, depth in enumerate(layers):
            for column_index, ep_size in enumerate(m_values):
                row = by_key[(ratio, depth, ep_size)]
                ladder = row["ladder"]
                times[row_index, column_index] = max(
                    float(ladder["elapsed_seconds"]), 1e-4
                )
                attempts[row_index, column_index] = len(ladder["attempts"])
                proven[row_index, column_index] = bool(ladder["optimality_proven"])

        figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
        positive_min = float(times[times > 0].min())
        time_image = axes[0].imshow(
            times,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            norm=LogNorm(vmin=positive_min, vmax=float(times.max())),
        )
        attempt_image = axes[1].imshow(
            attempts, aspect="auto", origin="lower", cmap="magma"
        )
        for axis, title in zip(
            axes, ("Feasibility-ladder solve time (s)", "Number of target attempts")
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
                if not proven[row_index, column_index]:
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
            f"Experiment C feasibility MILP grid, prune ratio {ratio:.0%}; x = unproven"
        )
        path = output_dir / f"heatmap-p{int(round(ratio * 100))}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    return outputs


def format_value(value: Any) -> str:
    if value is None:
        return "OOT"
    number = float(value)
    return f"{number:.6g}"


def write_report(args: argparse.Namespace) -> None:
    manifest = load_validated_manifest(args.manifest)
    output_dir = args.output_dir.expanduser().resolve()
    rows = load_rows(manifest, output_dir)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_name("grid.csv")
    write_csv(csv_path, rows)
    figures = plot_heatmaps(
        output.parent,
        rows,
        list(manifest["m_values"]),
        list(manifest["layers"]),
        list(manifest["prune_ratios"]),
    )

    proven = [row for row in rows if row["ladder"]["optimality_proven"]]
    reused = [row for row in rows if row["source_kind"] == "reused"]
    computed = [row for row in rows if row["source_kind"] == "computed"]
    times = [float(row["ladder"]["elapsed_seconds"]) for row in rows]
    lines = [
        "# Experiment C EP-size/depth grid",
        "",
        f"- Cases: {len(rows)}",
        f"- Proven optimal: {len(proven)}/{len(rows)}",
        f"- Reused exact E0/E0b artifacts: {len(reused)}",
        f"- Newly computed: {len(computed)}",
        f"- Median solve time: {statistics.median(times):.6g} s",
        f"- Maximum solve time: {max(times):.6g} s",
        "- Each newly computed case used one isolated four-core CPU affinity set.",
        "- E=128 experts per layer is fixed; m is the EP rank/placement-group count.",
        "",
    ]
    by_key = {
        (float(row["case"]["prune_ratio"]), int(row["case"]["layers"]), int(row["case"]["m"])): row
        for row in rows
    }
    max_depth = max(manifest["layers"])
    for ratio in manifest["prune_ratios"]:
        lines.extend(
            [
                f"## EP-size scan at L={max_depth}, p={ratio:.0%}",
                "",
                "| m | Time (s) | Spread | Attempts | Status | Source |",
                "| ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for ep_size in manifest["m_values"]:
            row = by_key[(float(ratio), int(max_depth), int(ep_size))]
            ladder = row["ladder"]
            status = "proven optimal" if ladder["optimality_proven"] else ladder["stopped_reason"]
            lines.append(
                f"| {ep_size} | {float(ladder['elapsed_seconds']):.6g} | "
                f"{format_value(ladder['spread'])} | {len(ladder['attempts'])} | "
                f"{status} | {row['source_kind']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Figures",
            "",
            *[f"- `{path.name}`" for path in figures],
            "- `grid.csv` contains all cells and provenance.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    checkpoint(
        output.with_suffix(".json"),
        {
            "experiment": "placement-grid-summary",
            "cases_sha256": manifest["cases_sha256"],
            "experiment_sha256": manifest["experiment_sha256"],
            "num_cases": len(rows),
            "num_proven": len(proven),
            "num_reused": len(reused),
            "num_computed": len(computed),
            "median_seconds": statistics.median(times),
            "max_seconds": max(times),
            "figures": [path.name for path in figures],
        },
    )


def main() -> int:
    args = parse_args()
    if args.command == "manifest":
        build_manifest(args)
    elif args.command == "worker":
        run_worker(args)
    else:
        write_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
