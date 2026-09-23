#!/usr/bin/env python3
"""Build a performance-only EP4 plan with a random, seed-fixed expert->tier
assignment and forced-balanced tier counts.

This is for throughput/memory benchmarking only, never for accuracy claims.
Because the four deployment strategies (padded/multi_kernel/single_width/
cross_layer) are compared under identical tier counts and channel masks, and
because fused-MoE kernel cost depends only on *how many* channels/experts
land in each tier (not *which* expert IDs or *which* channels), a random
expert identity is exactly as valid as a real weight-magnitude proxy for this
purpose: see docs/efficiency_campaign_plan.md, which already establishes that
efficiency plans must not be decided by accuracy importance scores. This
script skips reading any checkpoint weights entirely, so it also works for
models with no local accuracy plan and no verified vLLM/EP4 support yet
(e.g. Mistral-Small-4-119B-2603, Qwen3-VL-235B-A22B-Instruct-FP8).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_mask.ep4_intplan import plan_ep4_intplan
from src.vllm_ep4_plan import MODEL_WIDTH_PRESETS, SCHEMA_VERSION, validate_ep4_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=sorted(MODEL_WIDTH_PRESETS))
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prune-ratio", type=float, required=True)
    parser.add_argument("--placement-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing plan: {output_path}")

    config = json.loads(args.config_path.expanduser().read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    num_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config.get("num_experts") or text_config["n_routed_experts"])
    width = int(text_config["moe_intermediate_size"])

    widths = MODEL_WIDTH_PRESETS[args.model_id]
    if widths[-1] != width:
        raise SystemExit(
            f"preset top tier {widths[-1]} does not match "
            f"moe_intermediate_size={width} for {args.model_id}"
        )

    generator = torch.Generator().manual_seed(args.seed)
    scores = torch.rand((num_layers, num_experts, width), generator=generator)
    expert_sensitivity = scores.mean(dim=-1)
    layer_sensitivity = expert_sensitivity.mean(dim=-1)

    # A forced-balanced tier-count budget is a Diophantine feasibility
    # question (see _quantize_balanced_widths_to_budget): not every
    # (num_layers, num_experts, widths, prune_ratio) combination admits an
    # exact solution, independent of the model. Per
    # docs/efficiency_campaign_plan.md, when the nominal ratio is infeasible
    # we search nearby ratios and record whichever one actually worked,
    # rather than silently redefining or forcing an incompatible budget.
    plan = None
    tried_ratios: list[float] = []
    for offset in [0] + [sign * step for step in range(1, 41) for sign in (-1, 1)]:
        candidate_ratio = round(args.prune_ratio + offset * 0.0025, 6)
        if not 0.0 < candidate_ratio < 1.0:
            continue
        tried_ratios.append(candidate_ratio)
        try:
            plan = plan_ep4_intplan(
                layer_sensitivity=layer_sensitivity,
                expert_sensitivity=expert_sensitivity,
                scores=scores,
                prune_ratio=candidate_ratio,
                widths=widths,
                placement_tolerance=args.placement_tolerance,
                strict_placement_tolerance=False,
                balance_tier_counts=True,
                verbose=True,
            )
        except ValueError as error:
            if "globally balanced active tiers" not in str(error):
                raise
            continue
        if candidate_ratio != args.prune_ratio:
            print(
                f"[balanced plan] requested prune_ratio={args.prune_ratio} is "
                f"infeasible under forced tier balance; using nearest feasible "
                f"prune_ratio={candidate_ratio} instead",
                flush=True,
            )
        break
    if plan is None:
        raise SystemExit(
            f"no feasible balanced-tier prune_ratio found near {args.prune_ratio} "
            f"(tried {tried_ratios})"
        )
    plan.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model": args.model_id,
            "model_layer_ids": list(range(num_layers)),
            "source_config_path": str(args.config_path),
            "plan_purpose": "performance_only_random_balanced",
            "score_proxy": "random_uniform_seeded",
            "random_seed": args.seed,
            "tier_count_policy": "balanced_exact_budget",
        }
    )
    validate_ep4_plan(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(plan, handle)
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
