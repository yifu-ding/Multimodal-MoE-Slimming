#!/usr/bin/env python3
"""Re-place an existing EP4 plan with one permutation per layer block."""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vllm_ep4_plan import load_ep4_plan, validate_ep4_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=12)
    parser.add_argument("--method", choices=("cyclic", "optimized"), required=True)
    return parser.parse_args()


def _placement_objective(rank_loads: torch.Tensor) -> tuple[float, float]:
    centered = rank_loads.double() - rank_loads.double().mean()
    return float(centered.abs().max()), float(centered.square().sum())


def choose_block_permutations(
    active_width_counts: torch.Tensor,
    active_widths: tuple[int, ...],
    block_size: int,
    method: str,
) -> tuple[list[tuple[int, ...]], int]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    num_layers, num_tiers = active_width_counts.shape
    if num_tiers != 4 or len(active_widths) != 4:
        raise ValueError("block placement requires exactly four active width tiers")
    if num_layers % block_size != 0:
        raise ValueError(
            f"{num_layers} layers cannot be divided into blocks of {block_size}"
        )

    permutations = tuple(itertools.permutations(range(num_tiers)))
    identity = tuple(range(num_tiers))
    num_blocks = num_layers // block_size
    if method == "cyclic":
        return [
            tuple((rank + block) % num_tiers for rank in range(num_tiers))
            for block in range(num_blocks)
        ], 0

    widths = torch.tensor(active_widths, dtype=torch.int64)
    block_tier_loads = torch.stack(
        [
            active_width_counts[start : start + block_size].sum(dim=0) * widths
            for start in range(0, num_layers, block_size)
        ]
    )
    permutation_loads = torch.stack(
        [
            torch.stack([block_load[list(permutation)] for permutation in permutations])
            for block_load in block_tier_loads
        ]
    )

    # Keep the user's first-block convention (rank 0 owns the largest tier),
    # then exhaustively optimize the remaining block permutations. Parameter
    # balance is primary; fewer block-boundary width changes breaks ties.
    identity_id = permutations.index(identity)
    best_ids: tuple[int, ...] | None = None
    best_key: tuple[float, float, int, tuple[int, ...]] | None = None
    search_count = 0
    for suffix in itertools.product(range(len(permutations)), repeat=num_blocks - 1):
        ids = (identity_id, *suffix)
        rank_loads = sum(
            (permutation_loads[block, permutation_id] for block, permutation_id in enumerate(ids)),
            start=torch.zeros(num_tiers, dtype=torch.int64),
        )
        switches = sum(
            sum(a != b for a, b in zip(permutations[left], permutations[right]))
            for left, right in zip(ids, ids[1:])
        )
        key = (*_placement_objective(rank_loads), switches, ids)
        if best_key is None or key < best_key:
            best_key = key
            best_ids = ids
        search_count += 1
    assert best_ids is not None
    return [permutations[index] for index in best_ids], search_count


def build_block_plan(
    source: dict,
    source_path: Path,
    block_size: int,
    method: str,
) -> dict:
    plan = copy.deepcopy(source)
    counts = torch.as_tensor(plan["active_width_counts"], dtype=torch.int64)
    active_widths = tuple(int(width) for width in plan["active_widths"])
    block_permutations, search_count = choose_block_permutations(
        counts, active_widths, block_size, method
    )
    num_layers = counts.shape[0]
    tier_to_rank = torch.empty((num_layers, 4), dtype=torch.int64)
    rank_widths = torch.empty((num_layers, 4), dtype=torch.int64)
    permutation_ids = torch.empty(num_layers, dtype=torch.int64)
    all_permutations = tuple(itertools.permutations(range(4)))
    rank_loads = torch.zeros(4, dtype=torch.int64)

    for layer in range(num_layers):
        permutation = block_permutations[layer // block_size]
        permutation_ids[layer] = all_permutations.index(permutation)
        for rank, tier_index in enumerate(permutation):
            tier_to_rank[layer, tier_index] = rank
            rank_widths[layer, rank] = active_widths[tier_index]
            rank_loads[rank] += counts[layer, tier_index] * active_widths[tier_index]

    expert_widths = torch.as_tensor(plan["expert_widths"], dtype=torch.int64)
    expert_to_rank = torch.full_like(expert_widths, -1)
    expert_to_local_id = torch.full_like(expert_widths, -1)
    local_to_global: list[list[list[int]]] = []
    for layer in range(num_layers):
        width_to_rank = {
            active_widths[tier]: int(tier_to_rank[layer, tier]) for tier in range(4)
        }
        layer_mapping: list[list[int]] = [[], [], [], []]
        for expert, width_tensor in enumerate(expert_widths[layer]):
            width = int(width_tensor)
            if width == 0:
                continue
            rank = width_to_rank[width]
            expert_to_rank[layer, expert] = rank
            expert_to_local_id[layer, expert] = len(layer_mapping[rank])
            layer_mapping[rank].append(expert)
        local_to_global.append(layer_mapping)

    max_deviation, _ = _placement_objective(rank_loads)
    mean_load = float(rank_loads.double().mean())
    relative_deviation = max_deviation / mean_load if mean_load else 0.0
    tolerance = float(plan.get("tolerance", 0.01))
    plan.update(
        {
            "rank_widths": rank_widths,
            "tier_to_rank": tier_to_rank,
            "rank_weight_loads": rank_loads,
            "mean_rank_weight_load": mean_load,
            "max_rank_weight_deviation": max_deviation,
            "relative_max_rank_weight_deviation": relative_deviation,
            "tolerance_satisfied": relative_deviation <= tolerance + 1e-12,
            "permutation_ids": permutation_ids,
            "solver_status": 0,
            "solver_message": (
                f"block_size={block_size} method={method} "
                f"searched_combinations={search_count}"
            ),
            "solver_objective": max_deviation,
            "placement_method": f"block_{block_size}_{method}",
            "placement_block_size": block_size,
            "placement_block_permutations": [list(value) for value in block_permutations],
            "placement_source_plan": str(source_path),
            "expert_to_rank": expert_to_rank,
            "expert_to_local_id": expert_to_local_id,
            "local_to_global": local_to_global,
        }
    )
    plan.pop("plan_path", None)
    return validate_ep4_plan(plan)


def main() -> int:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing plan: {output_path}")
    source = load_ep4_plan(source_path)
    plan = build_block_plan(source, source_path, args.block_size, args.method)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(plan, handle)
    print(f"saved={output_path}")
    print(f"placement_method={plan['placement_method']}")
    print(f"block_permutations={plan['placement_block_permutations']}")
    print(f"rank_weight_loads={plan['rank_weight_loads'].tolist()}")
    print(
        "relative_max_rank_weight_deviation="
        f"{plan['relative_max_rank_weight_deviation']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
