# Mask adjustment utilities

from src.generate_mask.adjusters.align import adjust_with_align
from src.generate_mask.adjusters.align_score import adjust_with_align_score_sum
from src.generate_mask.adjusters.evict import adjust_with_caps_min
from src.base.shared_utils import _print

def adjust_masks(scores, masks, K_E, L, E, I, align=0, min_per_expert=0, verbose=False, **kwargs):
    layers = list(range(L))
    D_layer = {lid: E * I for lid in layers}
    adjust_method = kwargs.get("adjust_method", "largest_score_sum")
    
    if align is not None and align > 0:
        if adjust_method == "largest_channel":
            K_E, masks, info = adjust_with_align(
                K_E,
                scores,
                D_layer,
                cap_per_expert=I,
                align=align,
                min_per_expert=min_per_expert,
                layers=layers,
                prefer_keep_active=False,
                verbose=verbose,
            )
        elif adjust_method == "largest_score_sum":
            K_E, masks, info = adjust_with_align_score_sum(
                K_E,
                scores,
                D_layer,
                cap_per_expert=I,
                align=align,
                min_per_expert=min_per_expert,
                layers=layers,
                prefer_keep_active=False,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown adjust_method: {adjust_method}")
    elif min_per_expert is not None and min_per_expert > 0:
        K_E, masks, info = adjust_with_caps_min(K_E, scores, D_layer, cap_per_expert=I, min_per_expert=min_per_expert, layers=layers, verbose=verbose)
    else:
        K_E, masks, info = K_E, masks, None
        
    if verbose:
        _print(f"[Adjust Masks] ✅ adjust_masks info: {info}")
        _print(f"[Adjust Masks] adjust_method: {adjust_method}")
        _print(f"[Adjust Masks] example: layer0: {K_E[0]}")

    return masks, K_E
    

