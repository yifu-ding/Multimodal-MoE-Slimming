import torch


def create_mask_after_token(input_ids: torch.Tensor, special_token_id: int, offset: int = 0) -> torch.Tensor:
    """Return a boolean mask that starts after the first matched token plus offset."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_idx in range(input_ids.shape[0]):
        pos = (input_ids[row_idx] == special_token_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        start = int(pos[0].item()) + int(offset)
        if start < input_ids.shape[1]:
            mask[row_idx, start:] = True
    return mask


def create_mask_after_last_token(input_ids: torch.Tensor, special_token_id: int, offset: int = 0) -> torch.Tensor:
    """Return a boolean mask that starts after the last matched token plus offset."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_idx in range(input_ids.shape[0]):
        pos = (input_ids[row_idx] == special_token_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        start = int(pos[-1].item()) + int(offset)
        if start < input_ids.shape[1]:
            mask[row_idx, start:] = True
    return mask
