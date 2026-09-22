#!/usr/bin/env python3
"""Save Kimi-VL's direct MAES pruning masks without width or expert adjustment."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.generate_mask import generate_masks
from src.vllm_mask_runtime import load_mask_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prune-ratio", type=float, required=True)
    parser.add_argument("--intra-method", default="second_attr_coverage")
    args = parser.parse_args()
    if not 0 <= args.prune_ratio < 1:
        parser.error("--prune-ratio must be in [0, 1)")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    config = {
        "prune_ratio": args.prune_ratio,
        "mask_method_kwargs": {
            "inter_layer_method": "loss_smooth_2",
            "intra_layer_method": args.intra_method,
            "intra_expert_metric": "gateup_act",
            "layerwise_loss_key": "layerwise_loss",
        },
        "adjust_masks_kwargs": {"align_inter": 0, "min_per_expert": 0},
        "smooth_fn": "sqrt",
        "modality_aware": True,
        "shared_protect": True,
        "text_only": False,
        "visual_only": False,
        "normalize": False,
        "expertwise_budget_normalize": True,
        "use_ema": True,
        "ema_source_key": "ema_matrix",
    }
    result = generate_masks(str(args.scores), prune_kwargs=config, device="cpu", verbose=True)
    masks = result["intermediate_masks"].bool().cpu()
    plan = {
        "schema_version": 1,
        "model": "moonshotai/Kimi-VL-A3B-Instruct",
        "model_layer_ids": [int(x) for x in result["layers"]],
        "intermediate_masks": masks,
        "prune_ratio": args.prune_ratio,
        "source_scores": str(args.scores.resolve()),
        "pruning_config": config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(plan, handle)
    load_mask_plan(args.output)
    widths = masks.sum(-1)
    print(json.dumps({"output": str(args.output), "shape": list(masks.shape),
                      "actual_prune_ratio": 1 - masks.float().mean().item(),
                      "min_width": int(widths.min()), "max_width": int(widths.max()),
                      "unique_widths": int(widths.unique().numel())}, indent=2))


if __name__ == "__main__":
    main()
