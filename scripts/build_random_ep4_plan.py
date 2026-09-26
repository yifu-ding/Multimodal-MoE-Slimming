#!/usr/bin/env python3
"""Build a deterministic, performance-only EP4 plan without model scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_mask.ep4_intplan import solve_cross_layer_placement
from src.vllm_ep4_plan import MODEL_WIDTH_PRESETS, SCHEMA_VERSION, validate_ep4_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_WIDTH_PRESETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prune-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2603)
    parser.add_argument("--min-tier-experts", type=int, default=6)
    parser.add_argument("--max-tier-fraction", type=float, default=0.50)
    parser.add_argument("--placement-tolerance", type=float, default=0.01)
    return parser.parse_args()


def _model_shape(config: dict) -> tuple[int, int, int]:
    text_config = config.get("text_config", config)
    num_layers = int(text_config["num_hidden_layers"])
    num_experts = int(
        text_config.get("num_experts", text_config.get("n_routed_experts"))
    )
    width = int(text_config["moe_intermediate_size"])
    return num_layers, num_experts, width


def sample_nonuniform_tier_counts(
    *,
    num_layers: int,
    num_experts: int,
    active_widths: tuple[int, ...],
    target_keep_ratio: float,
    min_tier_experts: int,
    max_tier_fraction: float,
    generator: torch.Generator,
    candidates_per_layer: int = 4096,
) -> torch.Tensor:
    num_tiers = len(active_widths)
    if num_tiers != 4:
        raise ValueError("EP4 random plans require exactly four active widths")
    if min_tier_experts * num_tiers >= num_experts:
        raise ValueError("min_tier_experts leaves no experts for a nonuniform plan")
    max_tier_experts = int(num_experts * max_tier_fraction)
    if max_tier_experts < min_tier_experts:
        raise ValueError("max_tier_fraction is incompatible with min_tier_experts")
    if not 0.0 < target_keep_ratio <= 1.0:
        raise ValueError("target_keep_ratio must be in (0, 1]")

    widths = torch.tensor(active_widths, dtype=torch.float64)
    target = target_keep_ratio * num_experts * max(active_widths)
    remaining = num_experts - min_tier_experts * num_tiers
    minimum_spread = max(4, num_experts // 8)
    counts = torch.empty((num_layers, num_tiers), dtype=torch.int64)
    for layer in range(num_layers):
        best: tuple[tuple[float, int], torch.Tensor] | None = None
        for _ in range(candidates_per_layer):
            probabilities = torch.rand(num_tiers, generator=generator)
            probabilities = probabilities.square()
            probabilities /= probabilities.sum()
            extra = torch.multinomial(
                probabilities, remaining, replacement=True, generator=generator
            ).bincount(minlength=num_tiers)
            candidate = extra + min_tier_experts
            spread = int(candidate.max() - candidate.min())
            if spread < minimum_spread or int(candidate.max()) > max_tier_experts:
                continue
            budget_error = abs(float(candidate.to(torch.float64) @ widths) - target)
            score = (budget_error, -spread)
            if best is None or score < best[0]:
                best = (score, candidate)
                if budget_error == 0 and spread >= 2 * minimum_spread:
                    break
        if best is None:
            raise RuntimeError(f"could not sample a nonuniform tier split for layer {layer}")
        counts[layer] = best[1]
    return counts


def _build_mappings(
    expert_widths: torch.Tensor,
    active_widths: tuple[int, ...],
    tier_to_rank: torch.Tensor,
    full_width: int,
) -> dict:
    num_layers, num_experts = expert_widths.shape
    masks = torch.zeros((num_layers, num_experts, full_width), dtype=torch.bool)
    expert_to_rank = torch.full_like(expert_widths, -1)
    expert_to_local = torch.full_like(expert_widths, -1)
    local_to_global: list[list[list[int]]] = []
    width_to_tier = {width: tier for tier, width in enumerate(active_widths)}
    for layer in range(num_layers):
        local_ids: list[list[int]] = [[], [], [], []]
        for expert in range(num_experts):
            width = int(expert_widths[layer, expert])
            masks[layer, expert, :width] = True
            rank = int(tier_to_rank[layer, width_to_tier[width]])
            expert_to_rank[layer, expert] = rank
            expert_to_local[layer, expert] = len(local_ids[rank])
            local_ids[rank].append(expert)
        local_to_global.append(local_ids)
    return {
        "intermediate_masks": masks,
        "expert_to_rank": expert_to_rank,
        "expert_to_local_id": expert_to_local,
        "local_to_global": local_to_global,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.prune_ratio < 1.0:
        raise SystemExit("--prune-ratio must be in [0, 1)")
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing plan: {output_path}")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    num_layers, num_experts, full_width = _model_shape(config)
    preset = MODEL_WIDTH_PRESETS[args.model]
    if preset[0] != 0 or preset[-1] != full_width:
        raise SystemExit(
            f"width preset {preset} does not match model full width {full_width}"
        )
    active_widths = tuple(int(width) for width in preset[1:])
    if any(width % 128 for width in active_widths):
        raise SystemExit("FP8 performance tiers must be aligned to 128 channels")

    generator = torch.Generator().manual_seed(args.seed)
    width_counts = sample_nonuniform_tier_counts(
        num_layers=num_layers,
        num_experts=num_experts,
        active_widths=active_widths,
        target_keep_ratio=1.0 - args.prune_ratio,
        min_tier_experts=args.min_tier_experts,
        max_tier_fraction=args.max_tier_fraction,
        generator=generator,
    )
    expert_widths = torch.empty((num_layers, num_experts), dtype=torch.int64)
    for layer in range(num_layers):
        values = torch.repeat_interleave(
            torch.tensor(active_widths, dtype=torch.int64), width_counts[layer]
        )
        expert_widths[layer] = values[
            torch.randperm(num_experts, generator=generator)
        ]

    placement = solve_cross_layer_placement(
        width_counts,
        active_widths,
        tolerance=args.placement_tolerance,
        fix_first_layer=True,
    )
    mapping = _build_mappings(
        expert_widths,
        active_widths,
        placement["tier_to_rank"],
        full_width,
    )
    actual_keep = int(expert_widths.sum())
    total = num_layers * num_experts * full_width
    plan = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "model_layer_ids": list(range(num_layers)),
        "expert_widths": expert_widths,
        "rank_widths": placement["rank_widths"],
        "active_widths": active_widths,
        "width_counts": width_counts,
        "tier_to_rank": placement["tier_to_rank"],
        "rank_weight_loads": placement["rank_weight_loads"],
        "relative_max_rank_weight_deviation": placement[
            "relative_max_rank_weight_deviation"
        ],
        "prune_ratio": args.prune_ratio,
        "actual_prune_ratio": 1.0 - actual_keep / total,
        "source_model_path": str(model_path),
        "plan_purpose": "performance_only_random_no_scores",
        "random_seed": args.seed,
        "min_tier_experts": args.min_tier_experts,
        "max_tier_fraction": args.max_tier_fraction,
        "channel_selection": "contiguous_prefix_fp8_block_aligned",
        **mapping,
    }
    validate_ep4_plan(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(plan, handle)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "model": args.model,
                "seed": args.seed,
                "tier_count_min": width_counts.min(dim=0).values.tolist(),
                "tier_count_max": width_counts.max(dim=0).values.tolist(),
                "first_layer_tier_counts": width_counts[0].tolist(),
                "actual_prune_ratio": plan["actual_prune_ratio"],
                "relative_max_rank_weight_deviation": plan[
                    "relative_max_rank_weight_deviation"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
