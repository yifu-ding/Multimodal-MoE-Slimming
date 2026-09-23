#!/usr/bin/env python3
"""Compile MAES mixed-calibration scores into a validated EP4 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_mask import generate_masks, plan_ep4_from_masks
from src.generate_mask.stages.prepare_scores import prepare_scores
from src.vllm_ep4_plan import MODEL_WIDTH_PRESETS, SCHEMA_VERSION, validate_ep4_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_WIDTH_PRESETS))
    parser.add_argument("--prune-ratio", type=float, required=True)
    parser.add_argument("--placement-tolerance", type=float, default=0.01)
    parser.add_argument("--sparse-tier-max-experts", type=int, default=5)
    parser.add_argument("--inter-method", default="loss_smooth_2")
    parser.add_argument("--smooth-fn", default="sqrt")
    parser.add_argument("--intra-method", default="second_attr_coverage")
    parser.add_argument("--intra-expert-metric", default="gateup_act")
    parser.add_argument("--layerwise-loss-key", default="layerwise_loss")
    parser.add_argument("--ema-source-key", default="ema_matrix")
    parser.add_argument("--modality-aware", type=int, choices=(0, 1), default=1)
    parser.add_argument("--shared-protect", type=int, choices=(0, 1), default=1)
    parser.add_argument("--text-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--visual-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--normalize", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expertwise-budget-normalize", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use-ema", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--align-inter",
        type=int,
        default=0,
        help="Round each expert's kept channel count to a multiple of this value "
        "(0 disables; see src/generate_mask/adjusters/align.py).",
    )
    parser.add_argument(
        "--min-per-expert",
        type=int,
        default=0,
        help="Minimum kept channels for an active expert (0 disables); "
        "should itself be a multiple of --align-inter when both are set.",
    )
    parser.add_argument(
        "--adjust-method",
        choices=("largest_channel", "largest_score_sum"),
        default="largest_score_sum",
        help="largest_channel uses adjusters/align.py (round every expert to the "
        "align grid, redistribute leftover budget by channel count); "
        "largest_score_sum uses adjusters/align_score.py instead.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    scores_path = args.scores.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not scores_path.is_file():
        raise SystemExit(f"scores file does not exist: {scores_path}")
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing plan: {output_path}")
    if args.text_only and args.visual_only:
        raise SystemExit("--text-only and --visual-only cannot both be enabled")

    scores_payload = torch.load(scores_path, map_location="cpu", weights_only=False)
    metadata = scores_payload.get("metadata", {}) if isinstance(scores_payload, dict) else {}
    expected_layers = {int(layer) for layer in metadata.get("layers", [])}
    raw_layerwise = scores_payload.get(args.layerwise_loss_key, {})
    actual_layers = {int(layer) for layer in raw_layerwise} if isinstance(raw_layerwise, dict) else set()
    if not expected_layers or actual_layers != expected_layers:
        raise SystemExit(
            f"{args.layerwise_loss_key} does not cover every scored MoE layer: "
            f"missing={sorted(expected_layers - actual_layers)}, "
            f"extra={sorted(actual_layers - expected_layers)}"
        )
    invalid_layers = sorted(
        int(layer)
        for layer, value in raw_layerwise.items()
        if not torch.isfinite(torch.as_tensor(value, dtype=torch.float32)).all()
        or float(value) <= 0.0
    )
    if invalid_layers:
        raise SystemExit(
            f"{args.layerwise_loss_key} must be finite and positive for every layer; "
            f"invalid layers={invalid_layers}"
        )

    mask_method_kwargs = {
        "inter_layer_method": args.inter_method,
        "intra_layer_method": args.intra_method,
        "intra_expert_metric": args.intra_expert_metric,
        "layerwise_loss_key": args.layerwise_loss_key,
    }
    pruning_config = {
        "prune_ratio": args.prune_ratio,
        "mask_method_kwargs": mask_method_kwargs,
        "adjust_masks_kwargs": {
            "align_inter": args.align_inter,
            "min_per_expert": args.min_per_expert,
            "adjust_method": args.adjust_method,
        },
        "smooth_fn": args.smooth_fn,
        "modality_aware": bool(args.modality_aware),
        "shared_protect": bool(args.shared_protect),
        "text_only": bool(args.text_only),
        "visual_only": bool(args.visual_only),
        "normalize": bool(args.normalize),
        "expertwise_budget_normalize": bool(args.expertwise_budget_normalize),
        "use_ema": bool(args.use_ema),
        "ema_source_key": args.ema_source_key,
    }
    mask_result = generate_masks(
        scores_dir=str(scores_path),
        prune_kwargs=pruning_config,
        device="cpu",
        verbose=True,
    )
    (
        intermediate_scores,
        expert_sensitivity,
        _,
        _,
        _,
        loss_kwargs,
        score_layers,
    ) = prepare_scores(
        scores_dir=str(scores_path),
        mask_method_kwargs=mask_method_kwargs,
        smooth_fn=args.smooth_fn,
        modality_aware=bool(args.modality_aware),
        ema_source_key=args.ema_source_key,
        normalize=bool(args.normalize),
        device="cpu",
        verbose=False,
    )
    channel_scores = intermediate_scores[0] if isinstance(intermediate_scores, tuple) else intermediate_scores
    layer_sensitivity = loss_kwargs.get("layerwise_loss")
    if layer_sensitivity is None:
        layer_sensitivity = torch.ones(channel_scores.shape[0], dtype=torch.float32)

    plan = plan_ep4_from_masks(
        intermediate_masks=mask_result["intermediate_masks"],
        layer_sensitivity=layer_sensitivity,
        expert_sensitivity=expert_sensitivity,
        scores=channel_scores,
        prune_ratio=args.prune_ratio,
        widths=MODEL_WIDTH_PRESETS[args.model],
        placement_tolerance=args.placement_tolerance,
        strict_placement_tolerance=True,
        sparse_tier_max_experts=args.sparse_tier_max_experts,
        layer_smooth_times=2,
        layer_smooth_fn=args.smooth_fn,
        verbose=True,
    )
    model_layers = [int(value) for value in mask_result.get("layers", score_layers)]
    plan.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "model_layer_ids": model_layers,
            "source_scores": str(scores_path),
            "source_scores_sha256": sha256(scores_path),
            "pruning_config": pruning_config,
        }
    )
    validate_ep4_plan(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(plan, handle)

    summary = {
        "output": str(output_path),
        "model": args.model,
        "requested_prune_ratio": args.prune_ratio,
        "actual_prune_ratio": plan["actual_prune_ratio"],
        "sparse_tier_max_experts": plan["sparse_tier_max_experts"],
        "sparse_tier_merge_count": len(plan["sparse_tier_merges"]),
        "post_merge_budget_delta": plan["post_merge_budget_delta"],
        "rank_weight_loads": plan["rank_weight_loads"].tolist(),
        "relative_max_rank_weight_deviation": plan[
            "relative_max_rank_weight_deviation"
        ],
        "tolerance_satisfied": plan["tolerance_satisfied"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
