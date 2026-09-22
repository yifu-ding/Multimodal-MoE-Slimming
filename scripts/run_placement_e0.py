#!/usr/bin/env python3
"""Run Experiment C E0 arithmetic-floor early-stop validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_placement_ablation import (
    checkpoint,
    environment_metadata,
    no_incumbent_summary,
    placement_summary,
    warm_up_highs,
)
from src.generate_mask.ep4_intplan import (
    PlacementMilpNoIncumbentError,
    _build_layer_placement_groups,
    _placement_spread_arithmetic_lower_bound,
    _solve_placement_groups_milp,
)

DEFAULT_DEPTH = Path("results/placement_ablation/depth_sweep.json")
DEFAULT_M_SWEEP = Path("results/placement_ablation/m_sweep_v2.json")
DEFAULT_MANIFEST = Path("results/placement_e0/manifest.json")
DEFAULT_OUTPUT_DIR = Path("results/placement_e0/cases")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--depth-results", type=Path, default=DEFAULT_DEPTH)
    manifest.add_argument("--m-sweep-results", type=Path, default=DEFAULT_M_SWEEP)
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    worker.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument(
        "--arm-a-limits",
        nargs="+",
        type=float,
        default=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    )
    worker.add_argument("--arm-b-time-limit", type=float, default=300.0)

    report = subparsers.add_parser("report")
    report.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    report.add_argument(
        "--output", type=Path, default=Path("results/placement_e0/summary.md")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(cases: list[dict[str, Any]]) -> str:
    encoded = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def implementation_metadata() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256(path)
        for path in (
            REPO_ROOT / "src/generate_mask/ep4_intplan.py",
            REPO_ROOT / "scripts/run_placement_ablation.py",
            Path(__file__).resolve(),
        )
    }


def load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    actual = sha256(resolved)
    if actual != expected_sha256:
        raise RuntimeError(
            f"plan hash mismatch for {resolved}: expected {expected_sha256}, got {actual}"
        )
    return torch.load(resolved, map_location="cpu", weights_only=False)


def build_case_tensors(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    plan = load_plan(Path(case["plan"]), case["plan_sha256"])
    if case["kind"] == "depth":
        counts = torch.as_tensor(plan["active_width_counts"], dtype=torch.int64)
        widths = torch.as_tensor(plan["active_widths"], dtype=torch.int64).expand_as(
            counts
        )
        start = int(case["layer_start"])
        end = start + int(case["layers"])
        return counts[start:end], widths[start:end]
    if case["kind"] == "m-sweep":
        groups = _build_layer_placement_groups(
            torch.as_tensor(plan["expert_widths"], dtype=torch.int64),
            tuple(int(value) for value in plan["active_widths"]),
            ep_size=int(case["m"]),
        )
        return groups["placement_group_counts"], groups["placement_group_widths"]
    raise ValueError(f"unknown case kind: {case['kind']}")


def old_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "method",
        "spread",
        "relative_spread_percent",
        "median_time_seconds",
        "time_seconds",
        "time_limit",
        "solver_status",
        "solver_optimal",
        "dual_bound",
    )
    return {key: record.get(key) for key in keys}


def build_manifest(args: argparse.Namespace) -> None:
    depth_path = args.depth_results.expanduser().resolve()
    sweep_path = args.m_sweep_results.expanduser().resolve()
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))

    depth_greedy = {
        (
            row["plan_sha256"],
            row["layer_start"],
            row["layers"],
        ): row
        for row in depth["records"]
        if row["method"] == "greedy_full_bijection"
    }
    cases: list[dict[str, Any]] = []
    plan_cache: dict[str, dict[str, Any]] = {}
    for old_milp in depth["records"]:
        if old_milp["method"] != "milp_assignment" or old_milp["layers"] < 8:
            continue
        plan_path = Path(old_milp["plan"]).expanduser().resolve()
        plan = plan_cache.setdefault(
            old_milp["plan_sha256"],
            load_plan(plan_path, old_milp["plan_sha256"]),
        )
        counts = torch.as_tensor(plan["active_width_counts"], dtype=torch.int64)
        widths = torch.as_tensor(plan["active_widths"], dtype=torch.int64).expand_as(
            counts
        )
        start = int(old_milp["layer_start"])
        layers = int(old_milp["layers"])
        arithmetic = _placement_spread_arithmetic_lower_bound(
            counts[start : start + layers], widths[start : start + layers]
        )
        if arithmetic["arithmetic_spread_lower_bound"] != 128:
            continue
        if float(old_milp["spread"]) != 128.0:
            raise RuntimeError(
                f"expected floor-reaching old incumbent for {plan_path} "
                f"layers {start}:{start + layers}, got {old_milp['spread']}"
            )
        ratio_tag = f"p{int(round(float(old_milp['prune_ratio']) * 100))}"
        key = (old_milp["plan_sha256"], start, layers)
        cases.append(
            {
                "case_id": f"depth-{ratio_tag}-s{start:02d}-l{layers:02d}",
                "kind": "depth",
                "plan": str(plan_path),
                "plan_sha256": old_milp["plan_sha256"],
                "prune_ratio": old_milp["prune_ratio"],
                "layer_start": start,
                "layers": layers,
                "m": 4,
                **arithmetic,
                "old_milp": old_record_summary(old_milp),
                "old_greedy": old_record_summary(depth_greedy[key]),
            }
        )

    sweep_plan = sweep["plan"]
    old_sweep = {
        (row["m"], row["method"], row.get("time_limit")): row
        for row in sweep["records"]
    }
    for ep_size in (12, 16):
        case = {
            "case_id": f"m-sweep-p50-m{ep_size:02d}",
            "kind": "m-sweep",
            "plan": sweep_plan["path"],
            "plan_sha256": sweep_plan["sha256"],
            "prune_ratio": sweep_plan["prune_ratio"],
            "layer_start": 0,
            "layers": 48,
            "m": ep_size,
        }
        counts, widths = build_case_tensors(case)
        arithmetic = _placement_spread_arithmetic_lower_bound(counts, widths)
        if arithmetic["arithmetic_spread_lower_bound"] != 0:
            raise RuntimeError(f"expected zero floor for {case['case_id']}")
        case.update(arithmetic)
        case["old_milp"] = old_record_summary(
            old_sweep[(ep_size, "milp_assignment", 300.0)]
        )
        case["old_greedy"] = old_record_summary(
            old_sweep[(ep_size, "greedy_pairwise_swap", None)]
        )
        cases.append(case)

    cases.sort(key=lambda item: item["case_id"])
    if len(cases) != 31:
        raise RuntimeError(f"E0 must contain 31 cases, found {len(cases)}")
    depth_cases = [case for case in cases if case["kind"] == "depth"]
    if len(depth_cases) != 29:
        raise RuntimeError("E0 manifest must contain 29 nonzero-floor cases")
    old_oot = sum(not bool(case["old_milp"]["solver_optimal"]) for case in depth_cases)
    if old_oot != 18:
        raise RuntimeError(f"E0 manifest must relabel 18 old OOT cases, found {old_oot}")
    implementation = implementation_metadata()
    cases_sha256 = manifest_digest(cases)
    payload = {
        "experiment": "placement-e0",
        "created_at_unix": time.time(),
        "source_depth_results": str(depth_path),
        "source_m_sweep_results": str(sweep_path),
        "cases_sha256": cases_sha256,
        "implementation": implementation,
        "experiment_sha256": canonical_digest(
            {"cases_sha256": cases_sha256, "implementation": implementation}
        ),
        "cases": cases,
    }
    checkpoint(args.output.expanduser().resolve(), payload)
    relabel_path = args.output.expanduser().resolve().with_name(
        "relabelled_depth_cases.md"
    )
    lines = [
        "# Experiment C arithmetic relabel",
        "",
        "All rows below have arithmetic floor 128 and an incumbent equal to 128. "
        "They are therefore optimal even when HiGHS ended at its time limit.",
        "",
        "| Case | p | Layers | Old HiGHS status | Spread | Relabel |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for case in cases:
        if case["kind"] != "depth":
            continue
        old = case["old_milp"]
        old_status = "optimal" if old["solver_optimal"] else "OOT"
        lines.append(
            f"| {case['case_id']} | {case['prune_ratio']} | "
            f"{case['layer_start']}-{case['layer_start'] + case['layers'] - 1} | "
            f"{old_status} | {old['spread']:g} | arithmetic-floor optimal |"
        )
    relabel_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} E0 cases to {args.output}")


def solve_once(
    counts: torch.Tensor,
    widths: torch.Tensor,
    *,
    time_limit: float,
    spread_upper_bound: float | None = None,
    feasibility_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = _solve_placement_groups_milp(
            counts,
            widths,
            time_limit=time_limit,
            spread_upper_bound=spread_upper_bound,
            feasibility_only=feasibility_only,
        )
        summary = placement_summary(result)
    except PlacementMilpNoIncumbentError as error:
        summary = no_incumbent_summary(error)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["time_limit"] = float(time_limit)
    return summary


def run_case(
    case: dict[str, Any],
    *,
    arm_a_limits: list[float],
    arm_b_time_limit: float,
    cases_sha256: str,
    experiment_sha256: str,
) -> dict[str, Any]:
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
        raise RuntimeError(
            f"arithmetic metadata changed for {case['case_id']}: {arithmetic} != {expected}"
        )
    floor = float(arithmetic["arithmetic_spread_lower_bound"])

    arm_a = []
    time_to_floor_upper_bound = None
    for limit in arm_a_limits:
        summary = solve_once(counts, widths, time_limit=limit)
        reached_floor = summary["incumbent"] is not None and float(
            summary["incumbent"]
        ) <= floor + 1e-6
        summary["reached_floor"] = reached_floor
        arm_a.append(summary)
        if reached_floor:
            time_to_floor_upper_bound = float(limit)
            break

    arm_b = solve_once(
        counts,
        widths,
        time_limit=arm_b_time_limit,
        spread_upper_bound=floor,
        feasibility_only=True,
    )
    arm_b["target_spread"] = floor
    arm_b["floor_retry"] = False
    if arm_b["incumbent"] is None and arm_b["solver_status"] == 2:
        retry_target = floor + float(arithmetic["arithmetic_quantum"])
        arm_b = solve_once(
            counts,
            widths,
            time_limit=arm_b_time_limit,
            spread_upper_bound=retry_target,
            feasibility_only=True,
        )
        arm_b["target_spread"] = retry_target
        arm_b["floor_retry"] = True

    return {
        "experiment": "placement-e0-case",
        "cases_sha256": cases_sha256,
        "experiment_sha256": experiment_sha256,
        "case": case,
        "environment": environment_metadata(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "arm_a_limits": arm_a_limits,
        "arm_a": arm_a,
        "time_to_floor_upper_bound_seconds": time_to_floor_upper_bound,
        "arm_b_time_limit": arm_b_time_limit,
        "arm_b": arm_b,
        "completed_at_unix": time.time(),
    }


def run_worker(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard configuration")
    limits = [float(value) for value in args.arm_a_limits]
    if limits != sorted(set(limits)) or any(value <= 0.0 for value in limits):
        raise ValueError("Arm A limits must be unique positive values in ascending order")
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    cases = manifest["cases"]
    if manifest_digest(cases) != manifest["cases_sha256"]:
        raise RuntimeError("manifest case digest mismatch")
    if implementation_metadata() != manifest["implementation"]:
        raise RuntimeError("solver implementation differs from manifest")
    expected_experiment_sha256 = canonical_digest(
        {
            "cases_sha256": manifest["cases_sha256"],
            "implementation": manifest["implementation"],
        }
    )
    if expected_experiment_sha256 != manifest["experiment_sha256"]:
        raise RuntimeError("manifest experiment digest mismatch")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warm_up_highs()
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    for index, case in enumerate(cases):
        if index % args.num_shards != args.shard_id:
            continue
        output = output_dir / f"{case['case_id']}.json"
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("experiment_sha256") == manifest["experiment_sha256"]
                and existing.get("arm_a_limits") == limits
                and existing.get("arm_b_time_limit") == float(args.arm_b_time_limit)
                and existing.get("cpu_affinity") == affinity
            ):
                print(f"skip {case['case_id']}", flush=True)
                continue
            raise RuntimeError(f"stale E0 output: {output}")
        result = run_case(
            case,
            arm_a_limits=limits,
            arm_b_time_limit=float(args.arm_b_time_limit),
            cases_sha256=manifest["cases_sha256"],
            experiment_sha256=manifest["experiment_sha256"],
        )
        checkpoint(output, result)
        print(
            json.dumps(
                {
                    "case_id": case["case_id"],
                    "floor": case["arithmetic_spread_lower_bound"],
                    "arm_a_time_to_floor": result[
                        "time_to_floor_upper_bound_seconds"
                    ],
                    "arm_b_seconds": result["arm_b"]["elapsed_seconds"],
                    "arm_b_spread": result["arm_b"]["spread"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def write_report(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    output_dir = args.output_dir.expanduser().resolve()
    rows = []
    for case in manifest["cases"]:
        path = output_dir / f"{case['case_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"missing E0 result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("experiment_sha256") != manifest["experiment_sha256"]:
            raise RuntimeError(f"stale E0 result: {path}")
        rows.append(result)

    reached = [
        row["time_to_floor_upper_bound_seconds"]
        for row in rows
        if row["time_to_floor_upper_bound_seconds"] is not None
    ]
    arm_b_reached = [
        row for row in rows if row["arm_b"]["incumbent"] is not None
        and row["arm_b"]["spread"] <= row["arm_b"]["target_spread"] + 1e-6
    ]
    lines = [
        "# Experiment C E0 results",
        "",
        f"- Cases: {len(rows)}",
        f"- Arm A reached arithmetic floor: {len(reached)}/{len(rows)}",
        f"- Arm B reached target: {len(arm_b_reached)}/{len(rows)}",
        f"- Median Arm A time-to-floor upper bound: {statistics.median(reached) if reached else 'N/A'} s",
        "",
        "| Case | m | Floor | Old greedy | Old MILP | Arm A time-to-floor <= | Arm B spread | Arm B time | Proof |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        case = row["case"]
        arm_b = row["arm_b"]
        lines.append(
            "| {case_id} | {m} | {floor:g} | {greedy} | {old_milp} | {arm_a} | {arm_b_spread} | {arm_b_time:.6g} | {proof} |".format(
                case_id=case["case_id"],
                m=case["m"],
                floor=case["arithmetic_spread_lower_bound"],
                greedy=case["old_greedy"]["spread"],
                old_milp=case["old_milp"]["spread"],
                arm_a=row["time_to_floor_upper_bound_seconds"],
                arm_b_spread=arm_b["spread"],
                arm_b_time=arm_b["elapsed_seconds"],
                proof=arm_b["optimality_proof"],
            )
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checkpoint(
        output.with_suffix(".json"),
        {
            "experiment": "placement-e0-summary",
            "cases_sha256": manifest["cases_sha256"],
            "experiment_sha256": manifest["experiment_sha256"],
            "num_cases": len(rows),
            "arm_a_reached_floor": len(reached),
            "arm_b_reached_target": len(arm_b_reached),
            "median_arm_a_time_to_floor_upper_bound_seconds": (
                statistics.median(reached) if reached else None
            ),
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
