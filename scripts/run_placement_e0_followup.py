#!/usr/bin/env python3
"""Run Experiment C E0b feasibility-retry and EP-scaling validation."""

from __future__ import annotations

import argparse
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

from scripts.run_placement_ablation import checkpoint, environment_metadata, warm_up_highs
from scripts.run_placement_e0 import (
    canonical_digest,
    load_plan,
    manifest_digest,
    old_record_summary,
    sha256,
    solve_once,
)
from src.generate_mask.ep4_intplan import (
    _build_layer_placement_groups,
    _placement_spread_arithmetic_lower_bound,
)

DEFAULT_DEPTH = Path("results/placement_ablation/depth_sweep.json")
DEFAULT_M_SWEEP = Path("results/placement_ablation/m_sweep_v2.json")
DEFAULT_MANIFEST = Path("results/placement_e0_followup/manifest.json")
DEFAULT_OUTPUT_DIR = Path("results/placement_e0_followup/cases")


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
    worker.add_argument("--total-time-limit", type=float, default=300.0)

    report = subparsers.add_parser("report")
    report.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    report.add_argument(
        "--output", type=Path, default=Path("results/placement_e0_followup/summary.md")
    )
    return parser.parse_args()


def implementation_metadata() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256(path)
        for path in (
            REPO_ROOT / "src/generate_mask/ep4_intplan.py",
            REPO_ROOT / "scripts/run_placement_ablation.py",
            REPO_ROOT / "scripts/run_placement_e0.py",
            Path(__file__).resolve(),
        )
    }


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
    if case["kind"] == "scale":
        groups = _build_layer_placement_groups(
            torch.as_tensor(plan["expert_widths"], dtype=torch.int64),
            tuple(int(value) for value in plan["active_widths"]),
            ep_size=int(case["m"]),
        )
        return groups["placement_group_counts"], groups["placement_group_widths"]
    raise ValueError(f"unknown case kind: {case['kind']}")


def build_manifest(args: argparse.Namespace) -> None:
    depth_path = args.depth_results.expanduser().resolve()
    sweep_path = args.m_sweep_results.expanduser().resolve()
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))

    depth_greedy = {
        (row["plan_sha256"], row["layer_start"], row["layers"]): row
        for row in depth["records"]
        if row["method"] == "greedy_full_bijection"
    }
    plan_cache: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for old_milp in depth["records"]:
        if old_milp["method"] != "milp_assignment":
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
        if layers == 4:
            group = "retry-l4"
        elif layers >= 8 and arithmetic["arithmetic_spread_lower_bound"] == 0:
            group = "m4-floor0"
        else:
            continue
        if not bool(old_milp["solver_optimal"]):
            raise RuntimeError(f"E0b requires an exact old optimum: {old_milp}")
        ratio_tag = f"p{int(round(float(old_milp['prune_ratio']) * 100))}"
        key = (old_milp["plan_sha256"], start, layers)
        cases.append(
            {
                "case_id": f"{group}-{ratio_tag}-s{start:02d}-l{layers:02d}",
                "group": group,
                "kind": "depth",
                "plan": str(plan_path),
                "plan_sha256": old_milp["plan_sha256"],
                "prune_ratio": old_milp["prune_ratio"],
                "layer_start": start,
                "layers": layers,
                "m": 4,
                "expected_optimal_spread": float(old_milp["spread"]),
                **arithmetic,
                "old_milp": old_record_summary(old_milp),
                "old_greedy": old_record_summary(depth_greedy[key]),
            }
        )

    sweep_plan = sweep["plan"]
    for ep_size in (32, 64):
        case = {
            "case_id": f"scale-p50-l48-m{ep_size:02d}",
            "group": "ep-scale",
            "kind": "scale",
            "plan": sweep_plan["path"],
            "plan_sha256": sweep_plan["sha256"],
            "prune_ratio": sweep_plan["prune_ratio"],
            "layer_start": 0,
            "layers": 48,
            "m": ep_size,
            "expected_optimal_spread": None,
            "old_milp": None,
            "old_greedy": None,
        }
        counts, widths = build_case_tensors(case)
        case.update(_placement_spread_arithmetic_lower_bound(counts, widths))
        cases.append(case)

    cases.sort(key=lambda item: item["case_id"])
    group_counts = {
        group: sum(case["group"] == group for case in cases)
        for group in ("retry-l4", "m4-floor0", "ep-scale")
    }
    if group_counts != {"retry-l4": 24, "m4-floor0": 5, "ep-scale": 2}:
        raise RuntimeError(f"unexpected E0b case groups: {group_counts}")
    if any(
        case["arithmetic_spread_lower_bound"] != 0
        for case in cases
        if case["group"] in ("m4-floor0", "ep-scale")
    ):
        raise RuntimeError("all floor-zero and EP-scaling cases must have floor zero")

    implementation = implementation_metadata()
    cases_sha256 = manifest_digest(cases)
    payload = {
        "experiment": "placement-e0-followup",
        "created_at_unix": time.time(),
        "source_depth_results": str(depth_path),
        "source_m_sweep_results": str(sweep_path),
        "cases_sha256": cases_sha256,
        "implementation": implementation,
        "experiment_sha256": canonical_digest(
            {"cases_sha256": cases_sha256, "implementation": implementation}
        ),
        "group_counts": group_counts,
        "cases": cases,
    }
    checkpoint(args.output.expanduser().resolve(), payload)
    print(f"wrote {len(cases)} E0b cases to {args.output}")


def solve_feasibility_ladder(
    counts: torch.Tensor,
    widths: torch.Tensor,
    *,
    total_time_limit: float,
) -> dict[str, Any]:
    if total_time_limit <= 0.0:
        raise ValueError("total_time_limit must be positive")
    arithmetic = _placement_spread_arithmetic_lower_bound(counts, widths)
    floor = float(arithmetic["arithmetic_spread_lower_bound"])
    quantum = float(arithmetic["arithmetic_quantum"])
    started = time.perf_counter()
    deadline = started + total_time_limit
    target = floor
    attempts: list[dict[str, Any]] = []
    all_lower_targets_infeasible = True

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        result = solve_once(
            counts,
            widths,
            time_limit=remaining,
            spread_upper_bound=target,
            feasibility_only=True,
        )
        result["target_spread"] = target
        attempts.append(result)
        if result["incumbent"] is not None:
            spread = float(result["spread"])
            if spread > target + 1e-6:
                raise RuntimeError(f"feasibility incumbent {spread} exceeds target {target}")
            proof = (
                "arithmetic_lower_bound"
                if target <= floor + 1e-6
                else "quantum_feasibility_ladder"
            )
            return {
                "attempts": attempts,
                "found_feasible": True,
                "retry_count": len(attempts) - 1,
                "final_target_spread": target,
                "spread": spread,
                "elapsed_seconds": time.perf_counter() - started,
                "optimality_proven": all_lower_targets_infeasible,
                "optimality_proof": proof if all_lower_targets_infeasible else None,
                "stopped_reason": "feasible",
            }
        if int(result["solver_status"]) != 2:
            all_lower_targets_infeasible = False
            break
        target += quantum

    return {
        "attempts": attempts,
        "found_feasible": False,
        "retry_count": max(0, len(attempts) - 1),
        "final_target_spread": attempts[-1]["target_spread"] if attempts else floor,
        "spread": None,
        "elapsed_seconds": time.perf_counter() - started,
        "optimality_proven": False,
        "optimality_proof": None,
        "stopped_reason": "budget_or_unproven",
    }


def run_case(
    case: dict[str, Any],
    *,
    total_time_limit: float,
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
        raise RuntimeError(f"arithmetic metadata changed for {case['case_id']}")
    ladder = solve_feasibility_ladder(
        counts, widths, total_time_limit=total_time_limit
    )
    expected_spread = case["expected_optimal_spread"]
    if expected_spread is not None and (
        not ladder["optimality_proven"]
        or float(ladder["spread"]) != float(expected_spread)
    ):
        raise RuntimeError(
            f"E0b result disagrees with old optimum for {case['case_id']}: "
            f"{ladder['spread']} != {expected_spread}"
        )
    return {
        "experiment": "placement-e0-followup-case",
        "cases_sha256": cases_sha256,
        "experiment_sha256": experiment_sha256,
        "case": case,
        "environment": environment_metadata(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "total_time_limit": total_time_limit,
        "ladder": ladder,
        "completed_at_unix": time.time(),
    }


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


def run_worker(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard configuration")
    manifest = load_validated_manifest(args.manifest)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warm_up_highs()
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    for index, case in enumerate(manifest["cases"]):
        if index % args.num_shards != args.shard_id:
            continue
        output = output_dir / f"{case['case_id']}.json"
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("experiment_sha256") == manifest["experiment_sha256"]
                and existing.get("total_time_limit") == float(args.total_time_limit)
                and existing.get("cpu_affinity") == affinity
            ):
                print(f"skip {case['case_id']}", flush=True)
                continue
            raise RuntimeError(f"stale E0b output: {output}")
        result = run_case(
            case,
            total_time_limit=float(args.total_time_limit),
            cases_sha256=manifest["cases_sha256"],
            experiment_sha256=manifest["experiment_sha256"],
        )
        checkpoint(output, result)
        ladder = result["ladder"]
        print(
            json.dumps(
                {
                    "case_id": case["case_id"],
                    "attempts": len(ladder["attempts"]),
                    "retries": ladder["retry_count"],
                    "spread": ladder["spread"],
                    "seconds": ladder["elapsed_seconds"],
                    "proof": ladder["optimality_proof"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def write_report(args: argparse.Namespace) -> None:
    manifest = load_validated_manifest(args.manifest)
    output_dir = args.output_dir.expanduser().resolve()
    rows = []
    for case in manifest["cases"]:
        path = output_dir / f"{case['case_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"missing E0b result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("experiment_sha256") != manifest["experiment_sha256"]:
            raise RuntimeError(f"stale E0b result: {path}")
        rows.append(result)

    proven = [row for row in rows if row["ladder"]["optimality_proven"]]
    retry_rows = [row for row in rows if row["ladder"]["retry_count"] > 0]
    seconds = [row["ladder"]["elapsed_seconds"] for row in rows]
    lines = [
        "# Experiment C E0b results",
        "",
        f"- Cases: {len(rows)}",
        f"- Proven optimal: {len(proven)}/{len(rows)}",
        f"- Retry branch exercised: {len(retry_rows)}/{len(rows)}",
        f"- Median feasibility-ladder time: {statistics.median(seconds):.6g} s",
        f"- Maximum feasibility-ladder time: {max(seconds):.6g} s",
        "",
        "| Case | Group | m | Floor | Known optimum | Attempts | Retries | Spread | Wall time | Status | Proof |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        case = row["case"]
        ladder = row["ladder"]
        lines.append(
            "| {case_id} | {group} | {m} | {floor} | {known} | {attempts} | "
            "{retries} | {spread} | {seconds:.6g} | {status} | {proof} |".format(
                case_id=case["case_id"],
                group=case["group"],
                m=case["m"],
                floor=case["arithmetic_spread_lower_bound"],
                known=case["expected_optimal_spread"],
                attempts=len(ladder["attempts"]),
                retries=ladder["retry_count"],
                spread=ladder["spread"],
                seconds=ladder["elapsed_seconds"],
                status=ladder["stopped_reason"],
                proof=ladder["optimality_proof"],
            )
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checkpoint(
        output.with_suffix(".json"),
        {
            "experiment": "placement-e0-followup-summary",
            "cases_sha256": manifest["cases_sha256"],
            "experiment_sha256": manifest["experiment_sha256"],
            "num_cases": len(rows),
            "optimality_proven": len(proven),
            "retry_branch_exercised": len(retry_rows),
            "median_seconds": statistics.median(seconds),
            "max_seconds": max(seconds),
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
