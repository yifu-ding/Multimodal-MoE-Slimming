#!/usr/bin/env python3
"""Run the multi-instance supplement for Experiment C."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_placement_ablation import (
    checkpoint,
    load_output,
    no_incumbent_summary,
    placement_summary,
    plan_metadata,
    timed_repeats,
    warm_up_highs,
)
from src.generate_mask.ep4_intplan import (
    PlacementMilpNoIncumbentError,
    _build_layer_placement_groups,
    _solve_placement_groups_greedy,
    _solve_placement_groups_milp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--plan", type=Path, required=True)
    pilot.add_argument("--cases", nargs="+", required=True)
    pilot.add_argument("--time-limits", nargs="+", type=float, default=(60.0, 300.0))
    pilot.add_argument("--output", type=Path, required=True)

    depth = subparsers.add_parser("depth")
    depth.add_argument("--plans", nargs="+", type=Path, required=True)
    depth.add_argument("--layers", nargs="+", type=int, default=(8, 12, 16, 24, 32, 48))
    depth.add_argument("--stride", type=int, default=4)
    depth.add_argument("--greedy-repeats", type=int, default=5)
    depth.add_argument("--milp-time-limit", type=float, default=60.0)
    depth.add_argument("--pilot", type=Path, required=True)
    depth.add_argument("--output", type=Path, required=True)

    sweep = subparsers.add_parser("m-sweep")
    sweep.add_argument("--plans", nargs="+", type=Path, required=True)
    sweep.add_argument(
        "--m-values", nargs="+", type=int, default=(4, 5, 6, 7, 8, 9, 10, 11, 12, 16)
    )
    sweep.add_argument("--full-bijection-m-values", nargs="+", type=int, default=(5, 6, 7))
    sweep.add_argument("--greedy-repeats", type=int, default=5)
    sweep.add_argument("--milp-time-limit", type=float, default=300.0)
    sweep.add_argument("--output", type=Path, required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--pilot", type=Path, required=True)
    report.add_argument("--depth", type=Path, required=True)
    report.add_argument("--m-sweep", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    plan = torch.load(resolved, map_location="cpu", weights_only=False)
    return plan, plan_metadata(resolved, plan)


def run_milp(
    counts: torch.Tensor, widths: torch.Tensor, time_limit: float
) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    try:
        summary = placement_summary(
            _solve_placement_groups_milp(counts, widths, time_limit=time_limit)
        )
    except PlacementMilpNoIncumbentError as error:
        summary = no_incumbent_summary(error)
    return time.perf_counter() - started, summary


def parse_case(value: str) -> tuple[int, int]:
    try:
        start, layers = (int(part) for part in value.split(":", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid case {value!r}; expected START:LAYERS") from error
    return start, layers


def run_pilot(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    payload = load_output(output, "supplement-pilot", False)
    completed = {
        (record["plan_sha256"], record["layer_start"], record["layers"], record["time_limit"])
        for record in payload["records"]
    }
    plan, metadata = load_plan(args.plan)
    expert_widths = torch.as_tensor(plan["expert_widths"], dtype=torch.int64)
    active_widths = tuple(int(value) for value in plan["active_widths"])
    warm_up_highs()
    for start, layers in map(parse_case, args.cases):
        end = start + layers
        if start < 0 or end > expert_widths.shape[0]:
            raise ValueError(
                f"case {start}:{layers} exceeds {expert_widths.shape[0]} layers"
            )
        groups = _build_layer_placement_groups(
            expert_widths[start:end], active_widths, ep_size=4
        )
        counts = groups["placement_group_counts"]
        widths = groups["placement_group_widths"]
        for limit in args.time_limits:
            key = (metadata["sha256"], start, layers, float(limit))
            if key in completed:
                continue
            duration, summary = run_milp(counts, widths, limit)
            record = {
                "plan": metadata["path"],
                "plan_sha256": metadata["sha256"],
                "model": metadata["model"],
                "prune_ratio": metadata["prune_ratio"],
                "layer_start": start,
                "layer_end": end,
                "layers": layers,
                "m": 4,
                "method": "milp_assignment",
                "time_limit": float(limit),
                "time_seconds": duration,
                **summary,
            }
            payload["records"].append(record)
            checkpoint(output, payload)
            print(json.dumps(record, sort_keys=True), flush=True)

    grouped: dict[tuple[int, int], dict[float, dict[str, Any]]] = {}
    for record in payload["records"]:
        grouped.setdefault((record["layer_start"], record["layers"]), {})[
            float(record["time_limit"])
        ] = record
    limits = sorted(float(value) for value in args.time_limits)
    comparisons = []
    passed = True
    for case, records in sorted(grouped.items()):
        spreads = [records.get(limit, {}).get("spread") for limit in limits]
        same = all(value is not None for value in spreads) and max(spreads) == min(spreads)
        comparisons.append({"layer_start": case[0], "layers": case[1], "spreads": spreads, "same": same})
        passed = passed and same
    payload["validation"] = {"passed": passed, "time_limits": limits, "comparisons": comparisons}
    checkpoint(output, payload)
    if not passed:
        raise SystemExit("60-second MILP pilot changed solution quality; extended sweep blocked")


def sliding_windows(total_layers: int, layers: int, stride: int) -> list[tuple[int, int]]:
    if layers <= 0 or stride <= 0 or layers > total_layers:
        return []
    return [(start, start + layers) for start in range(0, total_layers - layers + 1, stride)]


def require_pilot(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("validation", {}).get("passed"):
        raise RuntimeError(f"pilot has not validated the shorter MILP limit: {path}")


def run_depth(args: argparse.Namespace) -> None:
    require_pilot(args.pilot.resolve())
    output = args.output.resolve()
    payload = load_output(output, "supplement-depth", False)
    completed = {
        (record["plan_sha256"], record["layer_start"], record["layers"], record["method"])
        for record in payload["records"]
    }
    warm_up_highs()
    for plan_path in args.plans:
        plan, metadata = load_plan(plan_path)
        expert_widths = torch.as_tensor(plan["expert_widths"], dtype=torch.int64)
        active_widths = tuple(int(value) for value in plan["active_widths"])
        for layers in args.layers:
            for start, end in sliding_windows(
                expert_widths.shape[0], layers, args.stride
            ):
                groups = _build_layer_placement_groups(
                    expert_widths[start:end], active_widths, ep_size=4
                )
                case_counts = groups["placement_group_counts"]
                case_widths = groups["placement_group_widths"]
                methods: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
                    (
                        "greedy_pairwise_swap",
                        lambda: _solve_placement_groups_greedy(
                            case_counts, case_widths, refinement_neighborhood="pairwise_swap"
                        ),
                    ),
                    (
                        "milp_assignment",
                        lambda: _solve_placement_groups_milp(
                            case_counts, case_widths, time_limit=args.milp_time_limit
                        ),
                    ),
                )
                for method, function in methods:
                    key = (metadata["sha256"], start, layers, method)
                    if key in completed:
                        continue
                    if method == "milp_assignment":
                        duration, summary = run_milp(
                            case_counts, case_widths, args.milp_time_limit
                        )
                        durations = [duration]
                    else:
                        durations, results = timed_repeats(
                            function, args.greedy_repeats, warmup=True
                        )
                        summary = placement_summary(results[-1])
                    record = {
                        "plan": metadata["path"],
                        "plan_sha256": metadata["sha256"],
                        "model": metadata["model"],
                        "prune_ratio": metadata["prune_ratio"],
                        "layer_start": start,
                        "layer_end": end,
                        "layers": layers,
                        "m": int(case_counts.shape[1]),
                        "method": method,
                        "time_limit": args.milp_time_limit if method == "milp_assignment" else None,
                        "times_seconds": durations,
                        "median_time_seconds": statistics.median(durations),
                        **summary,
                    }
                    payload["records"].append(record)
                    checkpoint(output, payload)
                    print(json.dumps(record, sort_keys=True), flush=True)


def run_m_sweep(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    payload = load_output(output, "supplement-m-sweep", False)
    completed = {
        (record["plan_sha256"], record["m"], record["method"])
        for record in payload["records"]
    }
    full_values = set(args.full_bijection_m_values)
    warm_up_highs()
    for plan_path in args.plans:
        plan, metadata = load_plan(plan_path)
        expert_widths = torch.as_tensor(plan["expert_widths"], dtype=torch.int64)
        active_widths = tuple(int(value) for value in plan["active_widths"])
        for ep_size in args.m_values:
            groups = _build_layer_placement_groups(expert_widths, active_widths, ep_size=ep_size)
            counts = groups["placement_group_counts"]
            widths = groups["placement_group_widths"]
            methods = ["greedy_pairwise_swap"]
            if ep_size in full_values:
                methods.append("greedy_full_bijection")
            methods.append("milp_assignment")
            for method in methods:
                key = (metadata["sha256"], ep_size, method)
                if key in completed:
                    continue
                if method == "milp_assignment":
                    duration, summary = run_milp(counts, widths, args.milp_time_limit)
                    durations = [duration]
                else:
                    neighborhood = method.removeprefix("greedy_")
                    durations, results = timed_repeats(
                        lambda: _solve_placement_groups_greedy(
                            counts, widths, refinement_neighborhood=neighborhood
                        ),
                        args.greedy_repeats,
                        warmup=True,
                    )
                    summary = placement_summary(results[-1])
                record = {
                    "plan": metadata["path"],
                    "plan_sha256": metadata["sha256"],
                    "model": metadata["model"],
                    "prune_ratio": metadata["prune_ratio"],
                    "m": ep_size,
                    "method": method,
                    "time_limit": args.milp_time_limit if method == "milp_assignment" else None,
                    "times_seconds": durations,
                    "median_time_seconds": statistics.median(durations),
                    **summary,
                }
                payload["records"].append(record)
                checkpoint(output, payload)
                print(json.dumps(record, sort_keys=True), flush=True)


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run_report(args: argparse.Namespace) -> None:
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in (("pilot", args.pilot), ("depth", args.depth), ("m-sweep", args.m_sweep))
    }
    lines = [
        "# Experiment C supplement results",
        "",
        f"- Pilot validation: `{payloads['pilot']['validation']['passed']}`",
        f"- Sliding-window records: `{len(payloads['depth']['records'])}`",
        f"- Multi-model m-sweep records: `{len(payloads['m-sweep']['records'])}`",
        "",
        "## M sweep",
        "",
        "| Model | p | m | Method | Spread | Relative (%) | Time (s) | Optimal |",
        "|---|---:|---:|---|---:|---:|---:|:---:|",
    ]
    for record in sorted(
        payloads["m-sweep"]["records"],
        key=lambda item: (item["model"], item["prune_ratio"], item["m"], item["method"]),
    ):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(record["model"]),
                    format_number(record["prune_ratio"]),
                    str(record["m"]),
                    record["method"],
                    format_number(record.get("spread")),
                    format_number(record.get("relative_spread_percent")),
                    format_number(record.get("median_time_seconds")),
                    str(record.get("solver_optimal", "-")),
                )
            )
            + " |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.experiment == "pilot":
        run_pilot(args)
    elif args.experiment == "depth":
        run_depth(args)
    elif args.experiment == "m-sweep":
        run_m_sweep(args)
    else:
        run_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
