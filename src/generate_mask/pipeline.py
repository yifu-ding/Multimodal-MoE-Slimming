from time import time
from typing import Any, Dict, Optional

import torch

from src.base.shared_utils import _print
from src.generate_mask.adjusters import trim_masks_to_layer_budget
from src.generate_mask.planners import build_modality_budget_masks
from src.generate_mask.stages import (
    adjust_masks,
    init_mask,
    load_modality_channel_scores,
    prepare_scores,
)

__all__ = [
    "generate_masks",
]


def generate_masks(
    scores_dir: str,
    mask_dir: Optional[str] = None,
    prune_kwargs: Dict[str, Any] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    if mask_dir is not None:
        masks = torch.load(mask_dir, map_location=device)
        if verbose:
            _print(f"[Mask Loading] Loaded masks from {mask_dir}")
        return masks if isinstance(masks, dict) else {"intermediate_masks": masks}

    prune_kwargs = prune_kwargs or {}
    prune_ratio = prune_kwargs.get("prune_ratio", 0.0)
    mask_method_kwargs = prune_kwargs.get("mask_method_kwargs", {})
    adjust_masks_kwargs = prune_kwargs.get("adjust_masks_kwargs", {})
    smooth_fn = prune_kwargs.get("smooth_fn", "sqrt")
    modality_aware = bool(prune_kwargs.get("modality_aware", False))

    (
        intermediate_scores,
        expertwise_scores,
        L,
        E,
        I,
        loss_based_kwargs,
        layers,
    ) = prepare_scores(
        scores_dir=scores_dir,
        mask_method_kwargs=mask_method_kwargs,
        prune_ratio=prune_ratio,
        smooth_fn=smooth_fn,
        device=device,
        verbose=verbose,
    )

    result = {}
    result["layers"] = layers
    inter_layer_method = loss_based_kwargs.get(
        "inter_layer_method",
        mask_method_kwargs.get("inter_layer_method", "uniform"),
    )

    start_time = time()
    mask_result = init_mask(
        intermediate_scores=intermediate_scores,
        expertwise_scores=expertwise_scores,
        prune_ratio=prune_ratio,
        inter_layer_method=inter_layer_method,
        loss_based_kwargs=loss_based_kwargs,
        intra_layer_method=mask_method_kwargs.get("intra_layer_method", "uniform"),
        L=L,
        E=E,
        I=I,
        verbose=verbose,
    )
    layerwise_keep_plan = mask_result["layerwise_keep_plan"]
    result["layerwise_keep_plan"] = layerwise_keep_plan

    if not modality_aware:
        result.update(mask_result)

    else:
        modality_scores = (
            load_modality_channel_scores(
                scores_dir,
                device=device,
                intra_expert_metric=mask_method_kwargs.get("intra_expert_metric", "activation"),
            )
            if modality_aware
            else None
        )
        if modality_scores is None:
            raise ValueError(f"modality-aware scores are required, but not found in {scores_dir}")
        if verbose:
            _print("[Mask Building] Applying modality-conditioned channel budgeting.")

        modality_masks, shared_masks = build_modality_budget_masks(
            modality_scores["text"],
            modality_scores["visual"],
            expertwise_scores=expertwise_scores,
            layerwise_keep_plan=layerwise_keep_plan,
            intra_layer_method=mask_method_kwargs.get("intra_layer_method", "uniform"),
            ema_matrix=modality_scores.get("ema_matrix", None),
        )
        result["intermediate_masks"] = trim_masks_to_layer_budget(
            masks=modality_masks,
            shared_masks=shared_masks,
            intermediate_scores=intermediate_scores,
            layerwise_keep_plan=layerwise_keep_plan,
            verbose=verbose,
        )
        result["K_E_inter"] = result["intermediate_masks"].sum(dim=-1)

    align_inter = adjust_masks_kwargs.get("align_inter", 0)
    min_per_expert = adjust_masks_kwargs.get("min_per_expert", 0)
    adjust_method = adjust_masks_kwargs.get("adjust_method", "largest_score_sum")

    if align_inter > 0 or min_per_expert > 0:
        result["intermediate_masks"], result["K_E_inter"] = adjust_masks(
            scores=intermediate_scores,
            masks=result["intermediate_masks"],
            K_E=result["K_E_inter"],
            L=L,
            E=E,
            I=I,
            align=align_inter,
            min_per_expert=min_per_expert,
            adjust_method=adjust_method,
            verbose=verbose,
        )
        
    if verbose:
        elapsed = (time() - start_time) * 1000.0
        keep_ratio = float(result["intermediate_masks"].float().mean().item())
        _print(f"[Mask Building] keep_ratio={keep_ratio:.4f}, elapsed={elapsed:.2f}ms")

    return result
