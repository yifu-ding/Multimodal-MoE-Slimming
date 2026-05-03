from typing import Optional, Set, Dict, Any
import math
import torch
import os
from src.generate_mask.planners.intra_layer import intra_layer_planner
from src.base.shared_utils import _print


def _pick_topk(available_scores: torch.Tensor, available_idx: torch.Tensor, k: int) -> Set[int]:
    if k <= 0 or available_idx.numel() == 0:
        return set()
    k = min(k, int(available_idx.numel()))
    local = torch.topk(available_scores, k=k, largest=True).indices
    return {int(available_idx[idx].item()) for idx in local}


def _normalize_modality_weights(
    use_ema: bool,
    ema_matrix: Optional[torch.Tensor],
    lid: int,
    eid: int,
) -> tuple[float, float]:
    if ema_matrix is None or not use_ema:
        norm_vis_ema = 0.5
    else:
        affinity = float(ema_matrix[lid, eid].item())
        assert affinity >= -1.0 and affinity <= 1.0, f"affinity should be in [-1.0, 1.0], but got {affinity}"
        norm_vis_ema = (affinity + 1.0) / 2.0
    norm_text_ema = 1.0 - norm_vis_ema
    return norm_vis_ema, norm_text_ema


def _select_weighted_unique_channels(
    visual_scores: torch.Tensor,
    visual_idx: torch.Tensor,
    text_scores: torch.Tensor,
    text_idx: torch.Tensor,
    target_budget: int,
    vis_weight: float,
    text_weight: float,
    initial_chosen: Optional[Set[int]] = None,
) -> tuple[Set[int], int, int]:
    # visual/text 对应的是同一组 channel 的两套排序.
    # 如果直接分别取 topk 后做并集, 常会因为重复 channel 导致最终 unique 数量小于 target_budget.
    # 这里改成按权重逐步扩张两边的排序前缀, 再用二分搜索找到满足目标 unique 数量的最小步数.
    chosen_base = set() if initial_chosen is None else set(initial_chosen)
    target_budget = max(0, int(target_budget))
    if len(chosen_base) >= target_budget:
        return chosen_base, 0, 0

    remaining_target = target_budget - len(chosen_base)
    if remaining_target <= 0:
        return chosen_base, 0, 0

    vis_order_local = torch.argsort(visual_scores, descending=True)
    text_order_local = torch.argsort(text_scores, descending=True)
    vis_sorted_idx = [int(visual_idx[pos].item()) for pos in vis_order_local]
    text_sorted_idx = [int(text_idx[pos].item()) for pos in text_order_local]

    total_steps = len(vis_sorted_idx) + len(text_sorted_idx)
    if total_steps == 0:
        return chosen_base, 0, 0

    def _simulate(num_steps: int) -> tuple[Set[int], int, int]:
        # 给定扩张步数, 按 vis_weight:text_weight 的目标比例交替从两套排序里拿下一个 channel.
        # 即使某一步拿到的 channel 已经在 chosen 里, 也继续推进对应指针, 因为这代表该模态已经消耗了一次配额.
        chosen = set(chosen_base)
        vis_ptr = 0
        text_ptr = 0
        vis_taken = 0
        text_taken = 0

        for step in range(num_steps):
            vis_remaining = vis_ptr < len(vis_sorted_idx)
            text_remaining = text_ptr < len(text_sorted_idx)
            if not vis_remaining and not text_remaining:
                break

            if vis_remaining and not text_remaining:
                pick_visual = True
            elif text_remaining and not vis_remaining:
                pick_visual = False
            else:
                next_step = step + 1
                vis_gap = next_step * vis_weight - vis_taken
                text_gap = next_step * text_weight - text_taken
                pick_visual = vis_gap >= text_gap

            if pick_visual:
                chosen.add(vis_sorted_idx[vis_ptr])
                vis_ptr += 1
                vis_taken += 1
            else:
                chosen.add(text_sorted_idx[text_ptr])
                text_ptr += 1
                text_taken += 1

        return chosen, vis_taken, text_taken

    # 理论上最多只能拿到两套候选索引并集里的 unique channel 数.
    # 如果 target 超过这个上限, 就把目标截断到可达范围内.
    max_unique = len(set(vis_sorted_idx) | set(text_sorted_idx))
    if remaining_target > max_unique:
        remaining_target = max_unique
        target_budget = len(chosen_base) + remaining_target

    # 二分搜索最小步数.
    # 步数越大, chosen 的 unique 数量单调不减, 因此可以二分.
    left = 0
    right = total_steps
    best_steps = total_steps
    while left <= right:
        mid = (left + right) // 2
        chosen_mid, _, _ = _simulate(mid)
        unique_count = len(chosen_mid) - len(chosen_base)
        if unique_count >= remaining_target:
            best_steps = mid
            right = mid - 1
        else:
            left = mid + 1

    chosen_final, vis_taken, text_taken = _simulate(best_steps)
    return chosen_final, vis_taken, text_taken


def _get_layer_target_budgets(
    layerwise_keep_plan: torch.Tensor,
    L: int,
    E: int,
    I: int,
    device: torch.device,
) -> torch.Tensor:
    keep_ratio = layerwise_keep_plan
    if not isinstance(keep_ratio, torch.Tensor):
        keep_ratio = torch.tensor(keep_ratio, dtype=torch.float32, device=device)
    else:
        keep_ratio = keep_ratio.to(device=device, dtype=torch.float32)

    if keep_ratio.ndim == 0:
        keep_ratio = keep_ratio.expand(L)
    elif keep_ratio.ndim == 1 and keep_ratio.numel() == 1:
        keep_ratio = keep_ratio.expand(L)
    elif keep_ratio.ndim == 1 and keep_ratio.numel() == L:
        pass
    elif keep_ratio.ndim == 2 and keep_ratio.shape == (L, E):
        return torch.round(keep_ratio.clamp(0.0, 1.0) * I).sum(dim=1).to(dtype=torch.int64)
    else:
        raise ValueError(f"Unsupported layerwise_keep_plan shape: {tuple(keep_ratio.shape)}")

    keep_ratio = keep_ratio.clamp(0.0, 1.0)
    layer_total_channels = E * I
    targets = [
        min(
            layer_total_channels,
            max(0, int(math.ceil(layer_total_channels * float(keep_ratio[lid].item()) - 1e-6))),
        )
        for lid in range(L)
    ]
    return torch.tensor(targets, dtype=torch.int64, device=device)


def _allocate_layer_expert_budgets(
    raw_targets: torch.Tensor,
    min_budgets: torch.Tensor,
    max_budgets: torch.Tensor,
    target_total: int,
) -> torch.Tensor:
    raw_targets = raw_targets.to(dtype=torch.float32)
    min_budgets = min_budgets.to(dtype=torch.int64)
    max_budgets = max_budgets.to(dtype=torch.int64)

    if raw_targets.ndim != 1:
        raise ValueError(f"raw_targets must be 1D, got {raw_targets.shape}")

    budgets = min_budgets.clone()
    cap = (max_budgets - min_budgets).clamp_min(0)
    min_total = int(min_budgets.sum().item())
    max_total = int(max_budgets.sum().item())
    target_total = min(max(target_total, min_total), max_total)
    remaining = target_total - min_total
    if remaining <= 0:
        return budgets

    weights = (raw_targets - min_budgets.to(dtype=torch.float32)).clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = cap.to(dtype=torch.float32)
    if float(weights.sum().item()) <= 0.0:
        return budgets

    ideal_extra = weights / weights.sum() * float(remaining)
    extra = torch.minimum(torch.floor(ideal_extra).to(dtype=torch.int64), cap)
    budgets += extra
    left = target_total - int(budgets.sum().item())

    while left > 0:
        available = budgets < max_budgets
        if not bool(torch.any(available)):
            break
        frac = ideal_extra - extra.to(dtype=torch.float32)
        frac = torch.where(available, frac, torch.full_like(frac, float("-inf")))
        pick = int(torch.argmax(frac).item())
        budgets[pick] += 1
        extra[pick] += 1
        left -= 1

    return budgets


def build_modality_budget_masks(
    text_scores: torch.Tensor,
    visual_scores: torch.Tensor,
    use_ema: bool,
    shared_protect: bool, 
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
    layer_target_budgets = _get_layer_target_budgets(
        layerwise_keep_plan=layerwise_keep_plan,
        L=L,
        E=E,
        I=I,
        device=text_scores.device,
    )
    masks = torch.zeros((L, E, I), dtype=torch.bool, device=text_scores.device)
    shared_masks = torch.zeros((L, E, I), dtype=torch.bool, device=text_scores.device)
    k_visual_tensor = torch.zeros((L, E), dtype=torch.int64, device=text_scores.device)
    k_text_tensor = torch.zeros((L, E), dtype=torch.int64, device=text_scores.device)

    for lid in range(L):
        raw_targets = torch.zeros(E, dtype=torch.float32, device=text_scores.device)
        min_budgets = torch.zeros(E, dtype=torch.int64, device=text_scores.device)
        max_budgets = torch.zeros(E, dtype=torch.int64, device=text_scores.device)

        layer_items: list[Dict[str, Any]] = []
        for eid in range(E):
            t = text_scores[lid, eid].float().clamp_min(0.0)
            v = visual_scores[lid, eid].float().clamp_min(0.0)

            text_mask = text_tentative[lid, eid]
            visual_mask = visual_tentative[lid, eid]
            norm_vis_ema, norm_text_ema = _normalize_modality_weights(
                use_ema=use_ema,
                ema_matrix=ema_matrix,
                lid=lid,
                eid=eid,
            )

            if shared_protect:
                shared_mask = text_mask & visual_mask
                text_only_mask = text_mask & (~visual_mask)
                visual_only_mask = visual_mask & (~text_mask)
                shared_masks[lid, eid] = shared_mask

                # chosen channel ids for shared mask
                chosen: Set[int] = set(torch.nonzero(shared_mask, as_tuple=False).flatten().tolist())
                raw_target = float(
                    text_K_E[lid, eid] * norm_text_ema + visual_K_E[lid, eid] * norm_vis_ema
                )

                visual_only_idx = torch.nonzero(visual_only_mask, as_tuple=False).flatten()
                text_only_idx = torch.nonzero(text_only_mask, as_tuple=False).flatten()
                raw_targets[eid] = raw_target
                min_budgets[eid] = int(len(chosen))
                max_budgets[eid] = int(len(chosen) + visual_only_idx.numel() + text_only_idx.numel())
                layer_items.append({
                    "shared_protect": True,
                    "t": t,
                    "v": v,
                    "norm_vis_ema": norm_vis_ema,
                    "norm_text_ema": norm_text_ema,
                    "chosen": chosen,
                    "visual_only_idx": visual_only_idx,
                    "text_only_idx": text_only_idx,
                })
            else:
                all_idx = torch.arange(I, device=text_scores.device)
                raw_targets[eid] = float(
                    text_K_E[lid, eid] * norm_text_ema + visual_K_E[lid, eid] * norm_vis_ema
                )
                min_budgets[eid] = 0
                max_budgets[eid] = I
                layer_items.append({
                    "shared_protect": False,
                    "t": t,
                    "v": v,
                    "norm_vis_ema": norm_vis_ema,
                    "norm_text_ema": norm_text_ema,
                    "all_idx": all_idx,
                })

        expert_target_budgets = _allocate_layer_expert_budgets(
            raw_targets=raw_targets,
            min_budgets=min_budgets,
            max_budgets=max_budgets,
            target_total=int(layer_target_budgets[lid].item()),
        )

        for eid in range(E):
            item = layer_items[eid]
            target_budget = int(expert_target_budgets[eid].item())
            if item["shared_protect"]:
                chosen, k_visual, k_text = _select_weighted_unique_channels(
                    visual_scores=item["v"][item["visual_only_idx"]],
                    visual_idx=item["visual_only_idx"],
                    text_scores=item["t"][item["text_only_idx"]],
                    text_idx=item["text_only_idx"],
                    target_budget=target_budget,
                    vis_weight=item["norm_vis_ema"],
                    text_weight=item["norm_text_ema"],
                    initial_chosen=item["chosen"],
                )
            else:
                chosen, k_visual, k_text = _select_weighted_unique_channels(
                    visual_scores=item["v"],
                    visual_idx=item["all_idx"],
                    text_scores=item["t"],
                    text_idx=item["all_idx"],
                    target_budget=target_budget,
                    vis_weight=item["norm_vis_ema"],
                    text_weight=item["norm_text_ema"],
                    initial_chosen=set(),
                )

            k_visual_tensor[lid, eid] = int(k_visual)
            k_text_tensor[lid, eid] = int(k_text)

            if item["shared_protect"] and chosen:
                shared_chosen = torch.tensor(
                    sorted(item["chosen"]),
                    device=text_scores.device,
                    dtype=torch.long,
                )
                shared_masks[lid, eid].zero_()
                shared_masks[lid, eid].index_fill_(0, shared_chosen, True)

            if chosen:
                chosen_idx = torch.tensor(sorted(chosen), device=text_scores.device, dtype=torch.long)
                masks[lid, eid].index_fill_(0, chosen_idx, True)

    if verbose:
        K_E = masks.sum(dim=-1)
        _print(f"[Modality-aware Budget] target layer budgets: {layer_target_budgets.tolist()}")
        _print(f"[Modality-aware Budget] actual layer budgets: {K_E.sum(dim=-1).tolist()}")
        _print(f"[Modality-aware Budget] after modality-aware budget: {K_E[0]}")

    return masks, shared_masks, text_K_E, visual_K_E, k_visual_tensor, k_text_tensor
