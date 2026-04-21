from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.base.shared_utils import angle_loss


def _is_qwen_like_model(model: nn.Module) -> bool:
    return (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    )


def compute_block_loss(
    pred: torch.Tensor,
    teacher_target: torch.Tensor,
    attn_mask: torch.Tensor,
    loss_fn: str = "rel_l2",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_f = attn_mask.float()
    rel_l2_inv_base_mean = None

    if loss_fn == "l2":
        token_mse = (pred.float() - teacher_target.float()).pow(2).mean(dim=-1)
        return (token_mse * mask_f).sum(), rel_l2_inv_base_mean

    if loss_fn == "rel_l2":
        pred_f = pred.float().view(-1, pred.size(-1))
        target_f = teacher_target.float().view(-1, teacher_target.size(-1))
        mask_flat = mask_f.view(-1)
        diff2 = (pred_f - target_f).pow(2).sum(dim=-1)
        base2 = target_f.pow(2).sum(dim=-1)
        loss_vec = diff2 / (base2 + eps) * mask_flat
        valid = mask_flat > 0
        if valid.any():
            rel_l2_inv_base_mean = (1.0 / (base2[valid] + eps)).mean().detach().float()
        return loss_vec.sum(), rel_l2_inv_base_mean

    if loss_fn == "cosine":
        return (angle_loss(pred, teacher_target) * mask_f).sum(), rel_l2_inv_base_mean

    raise ValueError(f"Unsupported loss_fn: {loss_fn}")


def teacher_blocks(bundle) -> Iterable[nn.Module]:
    if _is_qwen_like_model(bundle.model):
        return bundle.model.model.language_model.layers
    if hasattr(bundle.model, "language") and hasattr(bundle.model.language, "model") and hasattr(bundle.model.language.model, "layers"):
        return bundle.model.language.model.layers
    if hasattr(bundle.model, "language") and hasattr(bundle.model.language, "layers"):
        return bundle.model.language.layers
    if hasattr(bundle.model, "language_model") and hasattr(bundle.model.language_model, "layers"):
        return bundle.model.language_model.layers
    return bundle.model.language_model.model.layers


def teacher_block(bundle, layer_idx: int) -> nn.Module:
    return teacher_blocks(bundle)[layer_idx]


def resolve_special_token_tensor(bundle):
    model = bundle.model
    if _is_qwen_like_model(model):
        return getattr(model.model, "special_token_id_tensor", None)
    if hasattr(model, "language"):
        value = getattr(model.language, "special_token_id_tensor", None)
        if value is not None:
            return value
    return getattr(model, "special_token_id_tensor", None)


def resolve_media_token_ids(bundle) -> List[int]:
    config = bundle.model.config
    token_ids = []
    for attr in ("image_token_id", "video_token_id"):
        value = getattr(config, attr, None)
        if value is not None:
            token_ids.append(int(value))
    if not token_ids:
        value = getattr(config, "media_placeholder_token_id", None)
        if value is not None:
            token_ids.append(int(value))
    return token_ids


def fused_linear(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D fused expert weight, got shape={tuple(weight.shape)}")
    if weight.shape[-1] == hidden_states.shape[-1]:
        return F.linear(hidden_states, weight)
    if weight.shape[0] == hidden_states.shape[-1]:
        return hidden_states @ weight
    raise ValueError(
        f"Unsupported fused expert weight shape {tuple(weight.shape)} "
        f"for hidden size {hidden_states.shape[-1]}."
    )


def set_block_modality_masks(bundle, cnt_block: nn.Module, input_ids: torch.Tensor, attn_mask: torch.Tensor):
    flat_input_ids = input_ids.view(-1)
    special_ids = resolve_special_token_tensor(bundle)
    if special_ids is not None:
        special_ids = special_ids.to(flat_input_ids.device)
        moe_text_mask = ~torch.isin(flat_input_ids, special_ids)
    else:
        moe_text_mask = torch.ones_like(flat_input_ids, dtype=torch.bool)

    media_token_ids = resolve_media_token_ids(bundle)
    if media_token_ids:
        media_token_tensor = torch.tensor(
            media_token_ids, device=flat_input_ids.device, dtype=flat_input_ids.dtype
        )
        moe_media_mask = torch.isin(flat_input_ids, media_token_tensor)
    else:
        moe_media_mask = torch.zeros_like(flat_input_ids, dtype=torch.bool)

    cnt_block.mlp.moe_text_mask = moe_text_mask[:, None]
    cnt_block.mlp.moe_media_mask = moe_media_mask[:, None]
    if getattr(bundle, "family", None) == "qwen3":
        cnt_block.mlp.moe_padding_mask = (~attn_mask.to(torch.bool)).view(-1, 1)
    elif hasattr(cnt_block.mlp, "moe_padding_mask"):
        cnt_block.mlp.moe_padding_mask = None
    return moe_text_mask, moe_media_mask


def to_nested_expert_dict(layer_map: Dict[int, torch.Tensor], scalar: bool = False):
    nested = {}
    for layer_idx, tensor in layer_map.items():
        if scalar:
            nested[layer_idx] = {
                eid: float(tensor[eid].item())
                for eid in range(tensor.shape[0])
            }
        else:
            nested[layer_idx] = {
                eid: tensor[eid].detach().cpu().float()
                for eid in range(tensor.shape[0])
            }
    return nested
