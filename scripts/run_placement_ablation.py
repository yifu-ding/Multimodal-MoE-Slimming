#!/usr/bin/env python3
"""Run the greedy-versus-MILP placement ablations without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_mask.ep4_intplan import (
    PlacementMilpNoIncumbentError,
    _build_layer_placement_groups,
    _solve_placement_groups_greedy,
    _solve_placement_groups_milp,
)

DEFAULT_PLANS = (
    Path("/home/dyf/data/MARS-results/storage/ep4_plans/qwen3-vl-30b-a3b-p30.pt"),
    Path("/home/dyf/data/MARS-results/storage/ep4_plans/qwen3-vl-30b-a3b-p50.pt"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    table = subparsers.add_parser("table6")
    table.add_argument("--plans", nargs="+", type=Path, default=list(DEFAULT_PLANS))
    table.add_argument("--layers", nargs="+", type=int, default=(8, 24, 48))
    table.add_argument("--repeats", type=int, default=5)
    table.add_argument("--milp-time-limit", type=float, default=3600.0)
    table.add_argument(
        "--output", type=Path, default=Path("results/placement_ablation/table6.json")
    )
    table.add_argument("--force", action="store_true")

    depth = subparsers.add_parser("depth-sweep")
    depth.add_argument("--plans", nargs="+", type=Path, default=list(DEFAULT_PLANS))
    depth.add_argument(
        "--layers", nargs="+", type=int, default=(4, 8, 12, 16, 24, 32, 48)
    )
    depth.add_argument("--greedy-repeats", type=int, default=5)
    depth.add_argument("--milp-time-limit", type=float, default=300.0)
    depth.add_argument(
        "--output",
        type=Path,
        default=Path("results/placement_ablation/depth_sweep.json"),
    )
    depth.add_argument("--force", action="store_true")

    sweep = subparsers.add_parser("m-sweep")
    sweep.add_argument("--plan", type=Path, default=DEFAULT_PLANS[1])
    sweep.add_argument("--m-values", nargs="+", type=int, default=(4, 6, 8, 12, 16))
    sweep.add_argument(
        "--time-limits", nargs="+", type=float, default=(60.0, 600.0, 3600.0)
    )
    sweep.add_argument("--greedy-repeats", type=int, default=5)
    sweep.add_argument(
        "--output", type=Path, default=Path("results/placement_ablation/m_sweep.json")
    )
    sweep.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer() and isinstance(value, int):
        return int(number)
    return number


def environment_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cpu_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def load_output(path: Path, experiment: str, force: bool) -> dict[str, Any]:
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") != experiment:
            raise ValueError(f"output belongs to a different experiment: {path}")
        return payload
    return {
        "experiment": experiment,
        "created_at_unix": time.time(),
        "environment": environment_metadata(),
        "records": [],
    }


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def placement_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank_weight_loads": result["rank_weight_loads"].tolist(),
        "spread": float(result["rank_weight_spread"]),
        "relative_spread": float(result["relative_rank_weight_spread"]),
        "relative_spread_percent": 100.0 * float(result["relative_rank_weight_spread"]),
        "solver_status": result.get("solver_status"),
        "solver_success": result.get("solver_success"),
        "solver_optimal": result.get("solver_optimal"),
        "solver_message": result.get("solver_message"),
        "incumbent": finite_or_none(result.get("solver_objective")),
        "solver_reported_objective": finite_or_none(
            result.get("solver_reported_objective")
        ),
        "dual_bound": finite_or_none(result.get("mip_dual_bound")),
        "mip_gap": finite_or_none(result.get("mip_gap")),
        "mip_node_count": finite_or_none(result.get("mip_node_count")),
        "milp_binary_variables": result.get("milp_binary_variables"),
    }


def no_incumbent_summary(error: PlacementMilpNoIncumbentError) -> dict[str, Any]:
    diagnostics = error.diagnostics
    return {
        "rank_weight_loads": None,
        "spread": None,
        "relative_spread": None,
        "relative_spread_percent": None,
        "solver_status": diagnostics["solver_status"],
        "solver_success": diagnostics["solver_success"],
        "solver_optimal": False,
        "solver_message": diagnostics["solver_message"],
        "incumbent": None,
        "solver_reported_objective": None,
        "dual_bound": finite_or_none(diagnostics["mip_dual_bound"]),
        "mip_gap": finite_or_none(diagnostics["mip_gap"]),
        "mip_node_count": finite_or_none(diagnostics["mip_node_count"]),
        "milp_binary_variables": diagnostics.get("milp_binary_variables"),
    }


def timed_repeats(
    function: Callable[[], dict[str, Any]], repeats: int, warmup: bool
) -> tuple[list[float], list[dict[str, Any]]]:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if warmup:
        function()
    durations = []
    results = []
    for _ in range(repeats):
        started = time.perf_counter()
        results.append(function())
        durations.append(time.perf_counter() - started)
    return durations, results


def timed_milp_repeats(
    function: Callable[[], dict[str, Any]], repeats: int
) -> tuple[list[float], list[dict[str, Any]]]:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    try:
        function()
    except PlacementMilpNoIncumbentError:
        pass
    durations = []
    summaries = []
    for _ in range(repeats):
        started = time.perf_counter()
        try:
            summaries.append(placement_summary(function()))
        except PlacementMilpNoIncumbentError as error:
            summaries.append(no_incumbent_summary(error))
        durations.append(time.perf_counter() - started)
    return durations, summaries


def aggregate_milp_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    incumbents = [item for item in summaries if item["incumbent"] is not None]
    if incumbents:
        aggregate = dict(min(incumbents, key=lambda item: item["incumbent"]))
    else:
        aggregate = dict(summaries[-1])
    dual_bounds = [
        float(item["dual_bound"])
        for item in summaries
        if item["dual_bound"] is not None
    ]
    aggregate["dual_bound"] = max(dual_bounds) if dual_bounds else None
    if aggregate["incumbent"] is not None and aggregate["dual_bound"] is not None:
        difference = max(0.0, aggregate["incumbent"] - aggregate["dual_bound"])
        aggregate["mip_gap"] = (
            difference / aggregate["incumbent"] if aggregate["incumbent"] > 0.0 else 0.0
        )
    aggregate["solver_optimal"] = all(item["solver_optimal"] for item in summaries)
    aggregate["repeat_statuses"] = summaries
    return aggregate


def plan_metadata(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "model": plan.get("model"),
        "prune_ratio": float(plan["prune_ratio"]),
        "actual_prune_ratio": float(plan["actual_prune_ratio"]),
        "source_scores": plan.get("source_scores"),
        "source_scores_sha256": plan.get("source_scores_sha256"),
    }


def non_overlapping_layer_windows(
    total_layers: int, window_layers: int
) -> list[tuple[int, int]]:
    if window_layers <= 0 or window_layers > total_layers:
        raise ValueError(
            f"invalid layer count {window_layers} for {total_layers} total layers"
        )
    return [
        (start, start + window_layers)
        for start in range(0, total_layers - window_layers + 1, window_layers)
    ]


def run_table6(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    payload = load_output(output, "table6", args.force)
    completed = {
        (record["plan_sha256"], record["layers"], record["method"])
        for record in payload["records"]
    }
    for plan_path_value in args.plans:
        plan_path = plan_path_value.expanduser().resolve()
        plan = torch.load(plan_path, map_location="cpu", weights_only=False)
        metadata = plan_metadata(plan_path, plan)
        counts = torch.as_tensor(plan["active_width_counts"], dtype=torch.int64)
        widths = torch.as_tensor(plan["active_widths"], dtype=torch.int64).expand_as(
            counts
        )
        for num_layers in args.layers:
            if num_layers <= 0 or num_layers > counts.shape[0]:
                raise ValueError(f"invalid layer count {num_layers} for {plan_path}")
            layer_counts = counts[:num_layers]
            layer_widths = widths[:num_layers]
            methods = (
                (
                    "greedy_pairwise_swap",
                    lambda: _solve_placement_groups_greedy(
                        layer_counts,
                        layer_widths,
                        refinement_neighborhood="pairwise_swap",
                    ),
                ),
                (
                    "greedy_full_bijection",
                    lambda: _solve_placement_groups_greedy(
                        layer_counts,
                        layer_widths,
                        refinement_neighborhood="full_bijection",
                    ),
                ),
                (
                    "milp_assignment",
                    lambda: _solve_placement_groups_milp(
                        layer_counts,
                        layer_widths,
                        time_limit=args.milp_time_limit,
                    ),
                ),
            )
            for method, function in methods:
                key = (metadata["sha256"], num_layers, method)
                if key in completed:
                    continue
                if method == "milp_assignment":
                    durations, repeat_statuses = timed_milp_repeats(
                        function, args.repeats
                    )
                    summary = aggregate_milp_summaries(repeat_statuses)
                else:
                    durations, results = timed_repeats(
                        function, args.repeats, warmup=True
                    )
                    summary = placement_summary(results[-1])
                    summary["repeat_statuses"] = [
                        placement_summary(result) for result in results
                    ]
                record = {
                    "plan": metadata["path"],
                    "plan_sha256": metadata["sha256"],
                    "prune_ratio": metadata["prune_ratio"],
                    "layers": num_layers,
                    "m": int(layer_counts.shape[1]),
                    "method": method,
                    "warmup_runs": 1,
                    "timing_repeats": args.repeats,
                    "times_seconds": durations,
                    "median_time_seconds": (
                        statistics.median(durations) if durations else None
                    ),
                    **summary,
                }
                payload["records"].append(record)
                payload["plans"] = list(
                    {
                        item["sha256"]: item
                        for item in payload.get("plans", []) + [metadata]
                    }.values()
                )
                checkpoint(output, payload)
                print(json.dumps(record, sort_keys=True), flush=True)


def run_depth_sweep(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    payload = load_output(output, "depth-sweep", args.force)
    completed = {
        (
            record["plan_sha256"],
            record["layer_start"],
            record["layers"],
            record["method"],
        )
        for record in payload["records"]
    }
    warm_up_highs()
    for plan_path_value in args.plans:
        plan_path = plan_path_value.expanduser().resolve()
        plan = torch.load(plan_path, map_location="cpu", weights_only=False)
        metadata = plan_metadata(plan_path, plan)
        counts = torch.as_tensor(plan["active_width_counts"], dtype=torch.int64)
        widths = torch.as_tensor(plan["active_widths"], dtype=torch.int64).expand_as(
            counts
        )
        for num_layers in args.layers:
            for layer_start, layer_end in non_overlapping_layer_windows(
                counts.shape[0], num_layers
            ):
                layer_counts = counts[layer_start:layer_end]
                layer_widths = widths[layer_start:layer_end]
                methods = (
                    (
                        "greedy_pairwise_swap",
                        lambda: _solve_placement_groups_greedy(
                            layer_counts,
                            layer_widths,
                            refinement_neighborhood="pairwise_swap",
                        ),
                    ),
                    (
                        "greedy_full_bijection",
                        lambda: _solve_placement_groups_greedy(
                            layer_counts,
                            layer_widths,
                            refinement_neighborhood="full_bijection",
                        ),
                    ),
                    (
                        "milp_assignment",
                        lambda: _solve_placement_groups_milp(
                            layer_counts,
                            layer_widths,
                            time_limit=args.milp_time_limit,
                        ),
                    ),
                )
                for method, function in methods:
                    key = (metadata["sha256"], layer_start, num_layers, method)
                    if key in completed:
                        continue
                    if method == "milp_assignment":
                        started = time.perf_counter()
                        try:
                            summary = placement_summary(function())
                        except PlacementMilpNoIncumbentError as error:
                            summary = no_incumbent_summary(error)
                        durations = [time.perf_counter() - started]
                    else:
                        durations, results = timed_repeats(
                            function, args.greedy_repeats, warmup=True
                        )
                        summary = placement_summary(results[-1])
                        summary["repeat_statuses"] = [
                            placement_summary(result) for result in results
                        ]
                    record = {
                        "plan": metadata["path"],
                        "plan_sha256": metadata["sha256"],
                        "prune_ratio": metadata["prune_ratio"],
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "layers": num_layers,
                        "m": int(layer_counts.shape[1]),
                        "method": method,
                        "warmup_runs": 0 if method == "milp_assignment" else 1,
                        "timing_repeats": len(durations),
                        "times_seconds": durations,
                        "median_time_seconds": statistics.median(durations),
                        **summary,
                    }
                    payload["records"].append(record)
                    payload["plans"] = list(
                        {
                            item["sha256"]: item
                            for item in payload.get("plans", []) + [metadata]
                        }.values()
                    )
                    checkpoint(output, payload)
                    print(json.dumps(record, sort_keys=True), flush=True)


def warm_up_highs() -> None:
    counts = torch.tensor([[2, 1], [1, 2]], dtype=torch.int64)
    widths = torch.tensor([[2, 1], [2, 1]], dtype=torch.int64)
    _solve_placement_groups_milp(counts, widths, time_limit=10.0)


def certified_gap(
    greedy_spread: float, dual_bound: float | None
) -> dict[str, float | None]:
    if dual_bound is None:
        return {"certified_absolute_gap": None, "certified_relative_gap": None}
    absolute = max(0.0, greedy_spread - dual_bound)
    relative = absolute / dual_bound if dual_bound > 0.0 else None
    return {"certified_absolute_gap": absolute, "certified_relative_gap": relative}


def run_m_sweep(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    payload = load_output(output, "m-sweep", args.force)
    plan_path = args.plan.expanduser().resolve()
    plan = torch.load(plan_path, map_location="cpu", weights_only=False)
    metadata = plan_metadata(plan_path, plan)
    expert_widths = torch.as_tensor(plan["expert_widths"], dtype=torch.int64)
    active_widths = tuple(int(value) for value in plan["active_widths"])
    completed = {
        (record["m"], record["method"], record.get("time_limit"))
        for record in payload["records"]
    }
    warm_up_highs()

    for ep_size in args.m_values:
        groups = _build_layer_placement_groups(
            expert_widths,
            active_widths,
            ep_size=ep_size,
        )
        counts = groups["placement_group_counts"]
        widths = groups["placement_group_widths"]
        greedy_results: dict[str, dict[str, Any]] = {}
        neighborhoods = ["pairwise_swap"]
        if ep_size == 4:
            neighborhoods.append("full_bijection")
        for neighborhood in neighborhoods:
            method = f"greedy_{neighborhood}"
            key = (ep_size, method, None)
            if key in completed:
                continue
            durations, results = timed_repeats(
                lambda neighborhood=neighborhood: _solve_placement_groups_greedy(
                    counts,
                    widths,
                    refinement_neighborhood=neighborhood,
                ),
                args.greedy_repeats,
                warmup=True,
            )
            result = results[-1]
            greedy_results[neighborhood] = result
            record = {
                "plan_sha256": metadata["sha256"],
                "m": ep_size,
                "method": method,
                "time_limit": None,
                "timing_repeats": args.greedy_repeats,
                "times_seconds": durations,
                "median_time_seconds": statistics.median(durations),
                "initialization_time_seconds": result["initialization_time_seconds"],
                "refinement_time_seconds": result["refinement_time_seconds"],
                "initial_spread": result["initial_rank_weight_spread"],
                "initial_relative_spread": result[
                    "initial_relative_rank_weight_spread"
                ],
                **placement_summary(result),
            }
            payload["records"].append(record)
            payload["plan"] = metadata
            checkpoint(output, payload)
            print(json.dumps(record, sort_keys=True), flush=True)

        selected_neighborhood = "full_bijection" if ep_size == 4 else "pairwise_swap"
        if selected_neighborhood not in greedy_results:
            selected_record = next(
                record
                for record in payload["records"]
                if record["m"] == ep_size
                and record["method"] == f"greedy_{selected_neighborhood}"
            )
            greedy_spread = float(selected_record["spread"])
        else:
            greedy_spread = float(
                greedy_results[selected_neighborhood]["rank_weight_spread"]
            )

        for time_limit in args.time_limits:
            method = "milp_assignment"
            key = (ep_size, method, float(time_limit))
            if key in completed:
                continue
            started = time.perf_counter()
            try:
                result = _solve_placement_groups_milp(
                    counts,
                    widths,
                    time_limit=time_limit,
                )
                summary = placement_summary(result)
            except PlacementMilpNoIncumbentError as error:
                summary = no_incumbent_summary(error)
            duration = time.perf_counter() - started
            record = {
                "plan_sha256": metadata["sha256"],
                "m": ep_size,
                "method": method,
                "time_limit": float(time_limit),
                "time_seconds": duration,
                **summary,
                **certified_gap(greedy_spread, summary["dual_bound"]),
            }
            payload["records"].append(record)
            payload["plan"] = metadata
            checkpoint(output, payload)
            print(json.dumps(record, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    if args.experiment == "table6":
        run_table6(args)
    elif args.experiment == "depth-sweep":
        run_depth_sweep(args)
    else:
        run_m_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
