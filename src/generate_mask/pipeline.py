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
    expertwise_scores: torch.Tensor,
    layerwise_keep_plan: torch.Tensor,
    intra_layer_method: str,
    ema_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    text_mask_result = init_mask_for_I(
        intermediate_scores=text_scores,
        expertwise_scores=expertwise_scores,
        layerwise_keep_plan=layerwise_keep_plan,
        intra_layer_method=intra_layer_method,
        L=text_scores.shape[0],
        E=text_scores.shape[1],
        I=text_scores.shape[2],
        verbose=False,
    )
    visual_mask_result = init_mask_for_I(
        intermediate_scores=visual_scores,
        expertwise_scores=expertwise_scores,
        layerwise_keep_plan=layerwise_keep_plan,
        intra_layer_method=intra_layer_method,
        L=visual_scores.shape[0],
        E=visual_scores.shape[1],
        I=visual_scores.shape[2],
        verbose=False,
    )
    text_tentative = text_mask_result["intermediate_masks"]
    visual_tentative = visual_mask_result["intermediate_masks"]

    L, E, I = text_scores.shape
    masks = torch.zeros((L, E, I), dtype=torch.bool, device=text_scores.device)

    for lid in range(L):
        for eid in range(E):
            k = int(K_E[lid, eid].item())
            if k <= 0:
                continue

            t = text_scores[lid, eid].float().clamp_min(0.0)
            v = visual_scores[lid, eid].float().clamp_min(0.0)

            text_mask = text_tentative[lid, eid]
            visual_mask = visual_tentative[lid, eid]
            shared_mask = text_mask & visual_mask
            text_only_mask = text_mask & (~visual_mask)
            visual_only_mask = visual_mask & (~text_mask)

            chosen: Set[int] = set(torch.nonzero(shared_mask, as_tuple=False).flatten().tolist())
            shared_count = len(chosen)
            if shared_count >= k:
                shared_idx = torch.tensor(
                    sorted(chosen), device=text_scores.device, dtype=torch.long
                )
                # shared 超预算时，按两模态共同强度取 top-k。
                shared_score = torch.maximum(t, v)[shared_idx]
                top_shared = torch.topk(shared_score, k=k, largest=True).indices
                keep_idx = shared_idx[top_shared]
                masks[lid, eid].index_fill_(0, keep_idx, True)
                continue
            remaining = max(0, k - shared_count)
            if remaining == 0:
                chosen_idx = torch.tensor(sorted(chosen), device=text_scores.device, dtype=torch.long)
                masks[lid, eid].index_fill_(0, chosen_idx, True)
                continue

            if ema_matrix is None:
                norm_vis_ema = 0.5
            else:
                affinity = float(ema_matrix[lid, eid].item())
                affinity = max(-1.0, min(1.0, affinity))
                norm_vis_ema = (affinity + 1.0) / 2.0
            norm_text_ema = 1.0 - norm_vis_ema

            k_visual = int(round(remaining * norm_vis_ema))
            k_visual = min(remaining, max(0, k_visual))
            k_text = remaining - k_visual

            visual_only_idx = torch.nonzero(visual_only_mask, as_tuple=False).flatten()
            text_only_idx = torch.nonzero(text_only_mask, as_tuple=False).flatten()

            chosen.update(_pick_topk(v[visual_only_idx], visual_only_idx, k_visual))
            chosen.update(_pick_topk(t[text_only_idx], text_only_idx, k_text))

            # 如果某一侧不够，使用另一侧补齐。
            if len(chosen) < k:
                remaining_idx = torch.tensor(
                    [idx for idx in visual_only_idx.tolist() if idx not in chosen],
                    device=text_scores.device,
                    dtype=torch.long,
                )
                chosen.update(_pick_topk(v[remaining_idx], remaining_idx, k - len(chosen)))
            if len(chosen) < k:
                remaining_idx = torch.tensor(
                    [idx for idx in text_only_idx.tolist() if idx not in chosen],
                    device=text_scores.device,
                    dtype=torch.long,
                )
                chosen.update(_pick_topk(t[remaining_idx], remaining_idx, k - len(chosen)))

            # 兜底：如果 tentative union 仍不足预算，用 max(text, visual) 补齐。
            if len(chosen) < k:
                remaining_idx = torch.tensor(
                    [idx for idx in range(I) if idx not in chosen],
                    device=text_scores.device,
                    dtype=torch.long,
                )
                chosen.update(_pick_topk(torch.maximum(t, v)[remaining_idx], remaining_idx, k - len(chosen)))

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
        expertwise_scores,
        L,
        E,
        I,
        loss_based_kwargs,
    ) = prepare_scores(
        scores_dir=scores_dir,
        mask_method_kwargs=mask_method_kwargs,
        prune_ratio=prune_ratio,
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
            expertwise_scores=expertwise_scores,
            layerwise_keep_plan=layerwise_keep_plan,
            intra_layer_method=mask_method_kwargs.get("intra_layer_method", "uniform"),
            ema_matrix=modality_scores.get("ema_matrix", None),
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
