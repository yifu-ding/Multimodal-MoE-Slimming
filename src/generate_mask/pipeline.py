import math
from time import time
from typing import Any, Dict, Optional, Set

import torch

from src.base.shared_utils import _print
from src.generate_mask.stages import (
    adjust_masks,
    init_mask_for_I,
    load_modality_channel_scores,
    prepare_scores,
)

__all__ = [
    "generate_masks",
]


def _pick_topk(available_scores: torch.Tensor, available_idx: torch.Tensor, k: int) -> Set[int]:
    if k <= 0 or available_idx.numel() == 0:
        return set()
    k = min(k, int(available_idx.numel()))
    local = torch.topk(available_scores, k=k, largest=True).indices
    return {int(available_idx[idx].item()) for idx in local}


def _build_modality_budget_masks(
    text_scores: torch.Tensor,
    visual_scores: torch.Tensor,
    K_E: torch.Tensor,
) -> torch.Tensor:
    L, E, I = text_scores.shape
    masks = torch.zeros((L, E, I), dtype=torch.bool, device=text_scores.device)

    for lid in range(L):
        for eid in range(E):
            k = int(K_E[lid, eid].item())
            if k <= 0:
                continue

            t = text_scores[lid, eid].float().clamp_min(0.0)
            v = visual_scores[lid, eid].float().clamp_min(0.0)

            shared = torch.minimum(t, v)
            visual_specific = torch.clamp(v - t, min=0.0)
            text_specific = torch.clamp(t - v, min=0.0)

            shared_mass = float(shared.sum().item())
            visual_mass = float(visual_specific.sum().item())
            text_mass = float(text_specific.sum().item())
            total_mass = shared_mass + visual_mass + text_mass

            if total_mass <= 0.0:
                chosen = torch.topk(torch.maximum(t, v), k=min(k, I), largest=True).indices
                masks[lid, eid].index_fill_(0, chosen, True)
                continue

            k_shared = int(round(k * shared_mass / total_mass)) if shared_mass > 0 else 0
            k_shared = min(k, max(0, k_shared))
            remaining = k - k_shared

            if visual_mass + text_mass > 0 and remaining > 0:
                k_visual = int(round(remaining * visual_mass / (visual_mass + text_mass)))
                k_visual = min(remaining, max(0, k_visual))
                k_text = remaining - k_visual
                if visual_mass > 0 and text_mass > 0 and remaining >= 2:
                    if k_visual == 0:
                        k_visual = 1
                        k_text = remaining - 1
                    elif k_text == 0:
                        k_text = 1
                        k_visual = remaining - 1
            else:
                k_visual = 0
                k_text = 0

            chosen: Set[int] = set()

            shared_idx = torch.nonzero(shared > 0, as_tuple=False).flatten()
            chosen.update(_pick_topk(shared[shared_idx], shared_idx, k_shared))

            remaining_idx = torch.tensor(
                [idx for idx in range(I) if idx not in chosen],
                device=text_scores.device,
                dtype=torch.long,
            )
            if remaining_idx.numel() > 0:
                chosen.update(
                    _pick_topk(visual_specific[remaining_idx], remaining_idx, k_visual)
                )

            remaining_idx = torch.tensor(
                [idx for idx in range(I) if idx not in chosen],
                device=text_scores.device,
                dtype=torch.long,
            )
            if remaining_idx.numel() > 0:
                chosen.update(
                    _pick_topk(text_specific[remaining_idx], remaining_idx, k_text)
                )

            if len(chosen) < k:
                remaining_idx = torch.tensor(
                    [idx for idx in range(I) if idx not in chosen],
                    device=text_scores.device,
                    dtype=torch.long,
                )
                if remaining_idx.numel() > 0:
                    fallback = torch.maximum(t, v)
                    extra = _pick_topk(fallback[remaining_idx], remaining_idx, k - len(chosen))
                    chosen.update(extra)

            if chosen:
                chosen_idx = torch.tensor(sorted(chosen), device=text_scores.device, dtype=torch.long)
                masks[lid, eid].index_fill_(0, chosen_idx, True)

    return masks


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
        hidden_scores,
        expertwise_scores,
        L,
        E,
        I,
        H,
        loss_based_kwargs,
    ) = prepare_scores(
        scores_dir=scores_dir,
        mask_method_kwargs=mask_method_kwargs,
        HI_ratio_kwargs=prune_kwargs.get("HI_ratio_kwargs", {}),
        prune_ratio=prune_ratio,
        prune_hidden=prune_kwargs.get("prune_hidden", False),
        prune_gqa=prune_kwargs.get("prune_gqa", False),
        smooth_fn=smooth_fn,
        device=device,
        verbose=verbose,
    )

    result = {}
    layerwise_keep_plan = loss_based_kwargs["layerwise_keep_plan"]
    result["layers"] = loss_based_kwargs.get("layers", list(range(L)))
    result["layerwise_keep_plan"] = layerwise_keep_plan

    start_time = time()
    mask_result = init_mask_for_I(
        intermediate_scores=intermediate_scores,
        expertwise_scores=expertwise_scores,
        layerwise_keep_plan=layerwise_keep_plan,
        intra_layer_method=mask_method_kwargs.get("intra_layer_method", "uniform"),
        L=L,
        E=E,
        I=I,
        verbose=verbose,
    )
    result.update(mask_result)
    result["hidden_masks"] = None
    result["layerwise_inter_prune_ratio"] = None

    modality_scores = load_modality_channel_scores(scores_dir, device=device) if modality_aware else None
    if modality_scores is not None:
        if verbose:
            _print("[Mask Building] Applying modality-conditioned channel budgeting.")
        result["intermediate_masks"] = _build_modality_budget_masks(
            modality_scores["text"],
            modality_scores["visual"],
            result["K_E_inter"],
        )

    align_inter = adjust_masks_kwargs.get("align_inter", 0)
    min_per_expert = adjust_masks_kwargs.get("min_per_expert", 0)
    adjust_method = adjust_masks_kwargs.get("adjust_method", "largest_channel")

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
