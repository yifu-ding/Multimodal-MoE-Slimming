#!/usr/bin/env python3
"""Collect channel prune budget-allocation results via generate_masks only."""
# 仅画图使用，analysis 部分

"""
Default behavior runs three variants in one command:
1. modality_aware=False, use_ema=False
2. modality_aware=True, use_ema=False
3. modality_aware=True, use_ema=True
"""

import argparse
import os
import sys
from datetime import datetime

import torch

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.generate_mask import generate_masks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Collect budget allocation result from src.generate_mask.pipeline.generate_masks "
            "without loading any model."
        )
    )
    p.add_argument(
        "--scores_path",
        type=str,
        # default="storage/data_distill_kimi/mixed-num_1024-token_2048-sample_at1.0-0423143941/teacher_hidden-scores.pt",
        default="storage/data_distill_kimi/mixed-num_342-token_2048-sample_at1.0-0421180638/teacher_hidden-new_ema.pt"
        # default="storage/scores/kimi-vl-a3b_gqa-num_256-token_1024-fill_0-0424-182029/scores.pt"
        # default="storage/scores/kimi-vl-a3b_gqa-num_16-token_1024-fill_0-0424-181932/scores.pt"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/dyf/code/distill/MAES/observations/a1/results/second",
    )
    p.add_argument("--output_name", type=str, default="")

    p.add_argument("--prune_ratio", type=float, default=0.50)
    p.add_argument("--thresholds_path", type=str, default=None)
    p.add_argument("--inter_method", type=str, default="uniform")
    p.add_argument("--intra_method", type=str, default="uniform")
    p.add_argument("--intra_expert_metric", type=str, default="gateup_act")
    p.add_argument("--align_inter", type=int, default=0)
    p.add_argument("--min_per_expert", type=int, default=0)
    p.add_argument("--smooth_fn", type=str, default="sqrt")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--modality_aware", action="store_true")
    p.add_argument("--ema_source_key", type=str, default="ema_matrix")
    p.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EMA to mix text/visual budgets in modality-aware planner.",
    )
    p.add_argument(
        "--run_all_variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ma0+ema0, ma1+ema0, ma1+ema1 in one command.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _auto_name(args: argparse.Namespace, modality_aware: bool, use_ema: bool) -> str:
    if args.output_name:
        stem, ext = os.path.splitext(args.output_name)
        ext = ext or ".pt"
        return f"{stem}_{'ma1' if modality_aware else 'ma0'}_{'ema1' if use_ema else 'ema0'}{ext}"
    ma = "ma1" if args.modality_aware else "ma0"
    ema = "ema1" if args.use_ema else "ema0"
    ratio = str(args.prune_ratio).replace(".", "p")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"budget_alloc_{'ma1' if modality_aware else 'ma0'}_{'ema1' if use_ema else 'ema0'}_pr{ratio}_{ts}.pt"


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(x) for x in obj)
    return obj


def _variant_list(args: argparse.Namespace) -> list[tuple[bool, bool]]:
    if args.run_all_variants:
        return [
            (False, False),
            (True, False),
            (True, True),
        ]
    return [(bool(args.modality_aware), bool(args.use_ema))]


def _save_variant(
    args: argparse.Namespace,
    modality_aware: bool,
    use_ema: bool,
) -> str:
    prune_kwargs = {
        "prune_ratio": args.prune_ratio,
        "thresholds_path": args.thresholds_path,
        "mask_method_kwargs": {
            "inter_layer_method": args.inter_method,
            "intra_layer_method": args.intra_method,
            "intra_expert_metric": args.intra_expert_metric,
        },
        "adjust_masks_kwargs": {
            "align_inter": args.align_inter,
            "min_per_expert": args.min_per_expert,
        },
        "modality_aware": modality_aware,
        "use_ema": use_ema,
        "normalize": args.normalize,
        "smooth_fn": args.smooth_fn,
        "ema_source_key": args.ema_source_key,
    }

    result = generate_masks(
        scores_dir=args.scores_path,
        prune_kwargs=prune_kwargs,
        device="cpu",
        verbose=args.verbose,
    )
    result = _to_cpu(result)
    result.setdefault("shared_masks", None)
    result.setdefault("text_K_E", None)
    result.setdefault("visual_K_E", None)
    result.setdefault("k_visual", None)
    result.setdefault("k_text", None)
    if not use_ema:
        result["ema_matrix"] = None
    else:
        result.setdefault("ema_matrix", None)
    result["scores_dir"] = args.scores_path

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scores_dir": args.scores_path,
            "modality_aware": modality_aware,
            "use_ema": use_ema,
            "shared_masks_present": result.get("shared_masks") is not None,
        },
        "args": {
            **vars(args),
            "modality_aware": modality_aware,
            "use_ema": use_ema,
        },
        "result": result,
        "ema_matrix": result.get("ema_matrix"),
    }

    out_name = _auto_name(args, modality_aware=modality_aware, use_ema=use_ema)
    out_path = os.path.join(args.output_dir, out_name)
    torch.save(payload, out_path)
    print(f"[Saved] {out_path}")
    return out_path


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for modality_aware, use_ema in _variant_list(args):
        _save_variant(args, modality_aware=modality_aware, use_ema=use_ema)


if __name__ == "__main__":
    main()
