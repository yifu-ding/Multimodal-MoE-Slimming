import torch
from typing import Any, Dict, Optional, Tuple

from src.generate_mask.planners import inter_layer_planner, intra_layer_planner
from src.base.shared_utils import _print

def init_mask(
    intermediate_scores: torch.Tensor,  # [L, E, I]
    expertwise_scores: Optional[torch.Tensor] = None,  # [L, E] or None
    prune_ratio: float = 0.0,
    inter_layer_method: str = "uniform",
    loss_based_kwargs: Optional[Dict[str, Any]] = None,
    intra_layer_method: str = "uniform",
    L: int = None,
    E: int = None,
    I: int = None,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    loss_based_kwargs = dict(loss_based_kwargs or {})

    layerwise_keep_plan = inter_layer_planner(
        intermediate_scores,
        p_target=prune_ratio,
        method=inter_layer_method,
        L=L,
        loss_based_importance_kwargs=loss_based_kwargs,
        tol=0.1,
        verbose=verbose,
    )
   
    # 如果没有提供 expertwise_scores，使用均匀权重
    if expertwise_scores is not None:
        weighted_scores = torch.zeros_like(intermediate_scores)
        for layer_idx in range(L):
            expert_weights = expertwise_scores[layer_idx]  # [E]
            # expert_weights = torch.softmax(layer_expert_scores / expert_temp, dim=0)  # [E]
            weighted_scores[layer_idx] = intermediate_scores[layer_idx] * expert_weights[:, None]
        scores_to_use = weighted_scores
    else:
        scores_to_use = intermediate_scores
    
    # 构建 intra-layer masks
    intermediate_masks_float, K_E = intra_layer_planner(
        scores=scores_to_use,
        expertwise_scores=expertwise_scores,
        keep_ratio=layerwise_keep_plan,
        method=intra_layer_method,
        L=L,
        E=E,
        I=I,
        verbose=verbose,
    )
    
    # 转换为 bool 类型
    intermediate_masks = intermediate_masks_float.bool()

    if verbose:
        inter_keep_ratio_actual = float(intermediate_masks.sum().item()) / float(L * E * I)
        _print(
            f"[I ratio planning] method: {intra_layer_method}, keep_ratio: {inter_keep_ratio_actual:.4f}, "
        )
    
    return {
        "intermediate_masks": intermediate_masks,
        "K_E_inter": K_E,
        "layerwise_keep_plan": layerwise_keep_plan,
    }
