from time import time
from typing import Any, Dict, Optional

import torch

from src.base.shared_utils import _print
from src.generate_mask.adjusters import trim_masks_to_layer_budget
from src.generate_mask.planners import build_modality_budget_masks
from src.generate_mask.stages import (
    adjust_masks,
    init_mask,
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
    thresholds_path = prune_kwargs.get("thresholds_path")
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
        smooth_fn=smooth_fn,
        modality_aware=modality_aware, 
        device=device,
        verbose=verbose,
    )
    if modality_aware:
        modality_scores = intermediate_scores[1]
        intermediate_scores = intermediate_scores[0]

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

    use_modality = modality_aware or thresholds_path is not None

    if not use_modality:
        result.update(mask_result)

    else:
        if verbose:
            _print("[Mask Building] Applying modality-conditioned channel budgeting.")

        modality_masks, shared_masks = build_modality_budget_masks(
            modality_scores["text"],
            modality_scores["visual"],
            expertwise_scores=expertwise_scores,
            layerwise_keep_plan=layerwise_keep_plan,
            intra_layer_method=mask_method_kwargs.get("intra_layer_method", "uniform"),
            ema_matrix=modality_scores.get("ema_matrix", None),
            verbose=verbose,
        )

        if thresholds_path is not None:
            thresh_data = torch.load(
                thresholds_path, map_location=device, weights_only=False
            )

            n_override = 0
            for lid_pos, layer_idx in enumerate(layers):
                ratios = thresh_data["actual_keep_ratio"].get(layer_idx, None)  # 取实际的 keep_ratio
                if ratios is None:
                    continue

                k = int((ratios - prune_ratio).abs().argmin().item())  # 找到最接近的一组阈值
                if verbose:
                    _print(f"[Modality-aware Threshold] Layer {layer_idx} actual keep ratio: {ratios[k]:.4f}")
                    
                thresh_text = thresh_data["text_thresh"]
                thresh_visual = thresh_data["visual_thresh"]

                t_th = thresh_text[layer_idx][:, k].to(device)
                v_th = thresh_visual[layer_idx][:, k].to(device)
                ts = modality_scores["text"][lid_pos]
                vs = modality_scores["visual"][lid_pos]
                text_keep = ts >= t_th[:, None]
                visual_keep = vs >= v_th[:, None]
                modality_masks[lid_pos] = text_keep | visual_keep
                shared_masks[lid_pos] = text_keep & visual_keep
                n_override += 1

            if verbose:
                _print(
                    f"[Modality-aware Threshold] Overrode {n_override}/{len(layers)} layers with actual keep ratio"
                )

        result["intermediate_masks"] = trim_masks_to_layer_budget(
            masks=modality_masks,
            shared_masks=shared_masks,
            modality_scores=modality_scores,
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
