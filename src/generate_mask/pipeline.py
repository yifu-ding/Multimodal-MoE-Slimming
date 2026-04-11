import torch
from time import time
import numpy as np
from typing import Dict, Any, Optional

from src.generate_mask.stages import (
    prepare_scores, 
    load_attention_head_scores,
    init_mask_for_HI, 
    init_mask_for_I, 
    init_mask_for_gqa,
    adjust_masks,
)
from src.base.shared_utils import _print

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

    ##############################################
    # 1. load masks from mask_dir (if provided)
    # 为了防止训的 mask 和测试的 mask 不一致, 训练时会保存 mask 到 mask_dir (和 checkpoint 的位置相同)
    ##############################################
    
    if mask_dir is not None:
        masks = torch.load(mask_dir, map_location=device)
        if verbose:
            _print(f"[Mask Loading] ✅ Loaded masks from {mask_dir}, skip mask generate pipeline. ")
        
        if isinstance(masks, dict):
            # 兼容旧代码: 将 drop_kv_idx_plan 重命名为 drop_head_idx_plan
            if "drop_kv_idx_plan" in masks and "drop_head_idx_plan" not in masks:
                masks["drop_head_idx_plan"] = masks["drop_kv_idx_plan"]
                del masks["drop_kv_idx_plan"]
            elif "drop_kv_idx_plan" in masks and "drop_head_idx_plan" in masks:
                if verbose:
                    _print(f"[WARNING] Both 'drop_kv_idx_plan' and 'drop_head_idx_plan' found in masks. "
                           f"Using 'drop_head_idx_plan' (preferred).")
                del masks["drop_kv_idx_plan"]
        elif isinstance(masks, torch.Tensor):
            if masks.ndim == 2:
                masks = None
            else:
                _print(f"[Mask Loading] Intermediate masks shape: {masks.shape}")
                masks = {
                    "intermediate_masks": masks,
                    "hidden_masks": None,
                    "drop_head_idx_plan": None,
                    "distill_layers": None,
                }        
        return masks
    
    ##############################################
    # 2. 没有提供 mask_dir, 则从 scores 中生成 masks
    ##############################################
    prune_ratio = prune_kwargs.get("prune_ratio", 0.0)
    mask_method_kwargs = prune_kwargs.get("mask_method_kwargs", {})
    adjust_masks_kwargs = prune_kwargs.get("adjust_masks_kwargs", {})
    smooth_fn = prune_kwargs.get("smooth_fn", "sqrt")

    # 2.1 准备 scores
    intermediate_scores, expertwise_scores, L, E, I, \
        loss_based_importance_kwargs = prepare_scores(
        scores_dir=scores_dir,      # scores 目录路径
        prune_ratio=prune_ratio,   # 总体目标剪枝率
        smooth_fn=smooth_fn,       # layerwise loss smoothing variant
        mask_method_kwargs=mask_method_kwargs,  # 剪枝算法参数
        device=device,
        verbose=verbose,
    )
    
    if verbose:
        _print("[Step 1-2] ✅ Prepare scores and layerwise_keep_plan")

    result = {}  # 结果返回
    layerwise_keep_plan = loss_based_importance_kwargs["layerwise_keep_plan"]
    result["layerwise_keep_plan"] = layerwise_keep_plan
    
    # 2.2 构建 masks
    start_time = time()

    mask_result = init_mask_for_I(
        intermediate_scores=intermediate_scores,
        expertwise_scores=expertwise_scores,
        layerwise_keep_plan=layerwise_keep_plan,
        intra_layer_method=mask_method_kwargs["intra_layer_method"],
        L=L,
        E=E,
        I=I,
        verbose=verbose,
    )
    # intermediate_masks, K_E_inter = mask_result["intermediate_masks"], mask_result["K_E_inter"]
    result.update(mask_result)
    result["hidden_masks"] = None
    result["layerwise_inter_prune_ratio"] = None
    
    if verbose:
        _print(f"[Prune Inter] ✅ Building masks for I")

    ##############################################
    # 3. 调整 masks, 对齐量化 kernel 形状限制和最少保留通道数
    ##############################################
    align_inter = adjust_masks_kwargs.get("align_inter", 0)
    min_per_expert = adjust_masks_kwargs.get("min_per_expert", 0)
    adjust_method = adjust_masks_kwargs.get("adjust_method", "largest_channel")

    if align_inter > 0 or min_per_expert > 0:
        K_E_inter = result["K_E_inter"]
        intermediate_masks = result["intermediate_masks"]
        
        if K_E_inter is not None:
            intermediate_masks, K_E_inter = adjust_masks(
                scores=intermediate_scores,
                masks=intermediate_masks,
                K_E=K_E_inter,
                L=L,
                E=E,
                I=I,
                align=align_inter,
                min_per_expert=min_per_expert,
                adjust_method=adjust_method,
                verbose=verbose,
            )
            result["intermediate_masks"] = intermediate_masks
            result["K_E_inter"] = K_E_inter
        
    intra_end_time = time()
    if verbose:
        _print(f"Intra-layer mask building time (including adjust): {((intra_end_time - start_time) * 1000):.2f} ms")
    
  
    
    return result
