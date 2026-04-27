from typing import Optional, Set

import torch

from src.generate_mask.planners.intra_layer import intra_layer_planner
from src.base.shared_utils import _print


def _pick_topk(available_scores: torch.Tensor, available_idx: torch.Tensor, k: int) -> Set[int]:
    if k <= 0 or available_idx.numel() == 0:
        return set()
    k = min(k, int(available_idx.numel()))
    local = torch.topk(available_scores, k=k, largest=True).indices
    return {int(available_idx[idx].item()) for idx in local}


def build_modality_budget_masks(
    text_scores: torch.Tensor,
    visual_scores: torch.Tensor,
    use_ema: bool,
    expertwise_scores: torch.Tensor | None,
    layerwise_keep_plan: torch.Tensor,
    intra_layer_method: str,
    ema_matrix: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    def _tentative(scores: torch.Tensor) -> torch.Tensor:
        if expertwise_scores is not None:
            weighted = torch.zeros_like(scores)
            for lid in range(scores.shape[0]):
                weighted[lid] = scores[lid] * expertwise_scores[lid][:, None]
            scores_to_use = weighted
        else:
            scores_to_use = scores
        masks_float, K_E = intra_layer_planner(
            scores=scores_to_use,
            expertwise_scores=expertwise_scores,
            keep_ratio=layerwise_keep_plan,
            method=intra_layer_method,
            L=scores.shape[0],
            E=scores.shape[1],
            I=scores.shape[2],
            verbose=False,
        )
        return masks_float.bool(), K_E
    
    # 计算文本和视觉各自的 tentative masks
    text_tentative, text_K_E = _tentative(text_scores)
    visual_tentative, visual_K_E = _tentative(visual_scores)
    # target_budget = (text_K_E + visual_K_E)/2
    
    if verbose:
        _print(f"[Modality-aware Budget] before: text: {text_K_E[0]}, \n visual: {visual_K_E[0]}")
        
    L, E, I = text_scores.shape
    masks = torch.zeros((L, E, I), dtype=torch.bool, device=text_scores.device)
    shared_masks = torch.zeros((L, E, I), dtype=torch.bool, device=text_scores.device)
    k_visual_tensor = torch.zeros((L, E), dtype=torch.int64, device=text_scores.device)
    k_text_tensor = torch.zeros((L, E), dtype=torch.int64, device=text_scores.device)

    for lid in range(L):
        for eid in range(E):
            # if eid == E-1:
            #     import ipdb; ipdb.set_trace()
                
            t = text_scores[lid, eid].float().clamp_min(0.0)
            v = visual_scores[lid, eid].float().clamp_min(0.0)

            text_mask = text_tentative[lid, eid]
            visual_mask = visual_tentative[lid, eid]
            shared_mask = text_mask & visual_mask
            text_only_mask = text_mask & (~visual_mask)
            visual_only_mask = visual_mask & (~text_mask)
            shared_masks[lid, eid] = shared_mask

            # chosen channel ids for shared mask
            chosen: Set[int] = set(torch.nonzero(shared_mask, as_tuple=False).flatten().tolist())

            if ema_matrix is None or not use_ema:
                norm_vis_ema = 0.5
            else:
                affinity = float(ema_matrix[lid, eid].item())
                assert affinity >= -1.0 and affinity <= 1.0, f"affinity should be in [-1.0, 1.0], but got {affinity}"
                norm_vis_ema = (affinity + 1.0) / 2.0
            norm_text_ema = 1.0 - norm_vis_ema

            target_budget = text_K_E[lid, eid] * norm_text_ema + visual_K_E[lid, eid] * norm_vis_ema

            visual_only_idx = torch.nonzero(visual_only_mask, as_tuple=False).flatten()
            text_only_idx = torch.nonzero(text_only_mask, as_tuple=False).flatten()

            # k_visual = int(round(float(visual_only_idx.numel()) * norm_vis_ema))
            # k_text = int(round(float(text_only_idx.numel()) * norm_text_ema))
            # k_visual = min(int(visual_only_idx.numel()), max(0, k_visual))
            # k_text = min(int(text_only_idx.numel()), max(0, k_text))
            remaining_budget = float(target_budget) - len(chosen)
            k_visual = min(int(round(remaining_budget * norm_vis_ema)), int(visual_only_idx.numel()))
            k_text = min(int(round(remaining_budget * norm_text_ema)), int(text_only_idx.numel()))
            k_visual_tensor[lid, eid] = int(k_visual)
            k_text_tensor[lid, eid] = int(k_text)

            chosen.update(_pick_topk(v[visual_only_idx], visual_only_idx, k_visual))
            chosen.update(_pick_topk(t[text_only_idx], text_only_idx, k_text))
            if chosen:
                chosen_idx = torch.tensor(sorted(chosen), device=text_scores.device, dtype=torch.long)
                masks[lid, eid].index_fill_(0, chosen_idx, True)


    if verbose:
        K_E = masks.sum(dim=-1)
        _print(f"[Modality-aware Budget] after modality-aware budget: {K_E[0]}")

    return masks, shared_masks, text_K_E, visual_K_E, k_visual_tensor, k_text_tensor
