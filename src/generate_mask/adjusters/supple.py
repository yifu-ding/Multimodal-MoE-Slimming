import torch
from src.base.shared_utils import _print

def trim_masks_to_layer_budget(
    masks: torch.Tensor,
    shared_masks: torch.Tensor,
    modality_scores: torch.Tensor,
    layerwise_keep_plan: torch.Tensor,
    verbose: bool = False,
) -> torch.Tensor:
    """Trim per-layer mask counts to layer budgets while protecting shared channels."""
    L, E, I = masks.shape
    total_per_layer = E * I
    
    # max_scores = torch.maximum(modality_scores["text"], modality_scores["visual"])
    _scores = modality_scores["text"] + modality_scores["visual"]
    _scores = _scores / 2.0

    for lid in range(L):
        keep_ratio = layerwise_keep_plan[lid]
        if isinstance(keep_ratio, torch.Tensor):
            keep_ratio = float(keep_ratio.item())
        else:
            keep_ratio = float(keep_ratio)
        target_keep = int(round(keep_ratio * float(total_per_layer)))
        target_keep = max(0, min(total_per_layer, target_keep))
        current_keep = int(masks[lid].sum().item())
        if current_keep == target_keep:
            continue

        if current_keep < target_keep:
            need_add = target_keep - current_keep
            addable_mask = ~masks[lid]
            addable_idx = torch.nonzero(addable_mask, as_tuple=False)
            if addable_idx.numel() == 0:
                continue
            addable_scores = _scores[lid, addable_idx[:, 0], addable_idx[:, 1]].float()
            order = torch.argsort(addable_scores, descending=True)
            add_cnt = min(int(need_add), int(addable_idx.shape[0]))
            to_add = addable_idx[order[:add_cnt]]
            masks[lid, to_add[:, 0], to_add[:, 1]] = True
            if verbose:
                after_keep = int(masks[lid].sum().item())
                _print(
                    f"[Mask Building] L{lid}: grow {current_keep}->{after_keep} "
                    f"(target={target_keep})."
                )
            continue

        removable_mask = masks[lid] & (~shared_masks[lid])
        removable_idx = torch.nonzero(removable_mask, as_tuple=False)
        if removable_idx.numel() == 0:
            if verbose:
                _print(
                    f"[Mask Building] L{lid}: over budget ({current_keep}>{target_keep}) "
                    "but no removable non-shared channels."
                )
            continue

        need_remove = current_keep - target_keep
        removable_scores = _scores[lid, removable_idx[:, 0], removable_idx[:, 1]].float()
        order = torch.argsort(removable_scores, descending=False)
        remove_cnt = min(int(need_remove), int(removable_idx.shape[0]))
        to_remove = removable_idx[order[:remove_cnt]]
        masks[lid, to_remove[:, 0], to_remove[:, 1]] = False

        if verbose:
            after_keep = int(masks[lid].sum().item())
            _print(
                f"[Mask Building] L{lid}: trim {current_keep}->{after_keep} "
                f"(target={target_keep}, protected_shared={int(shared_masks[lid].sum().item())})."
            )

    return masks
