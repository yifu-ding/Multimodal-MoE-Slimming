import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def kimi_rtp_enabled() -> bool:
    return os.getenv("FASTMMOE_KIMI_RTP_ENABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class KimiRTPConfig:
    reduction_layer_idx: int
    token_merge_strategy: str
    base_alpha: float
    base_beta: float
    merge_layer_locs: list[int]
    merge_token_nums: list[int]
    keep_token_ratios: Optional[list[float]]
    window_size: int
    merge_method: str
    merge_ratio: float


@dataclass
class KimiRTPResult:
    hidden_states: torch.Tensor
    residual: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    image_seq_mask: torch.Tensor
    applied: bool


def load_kimi_rtp_config() -> KimiRTPConfig:
    merge_layer_locs = [int(x) for x in os.getenv("MERGE_LAYER_LOCS", "2,5,8").split(",")]
    merge_token_nums = [int(x) for x in os.getenv("MERGE_TOKEN_NUMS", "500,400,200").split(",")]
    keep_token_raw = os.getenv("KEEP_TOKEN_RATIO", "none")
    keep_token_ratios = None
    if keep_token_raw != "none":
        keep_token_ratios = [float(x) for x in keep_token_raw.split(",")]
    alpha = float(os.getenv("BASE_ALPHA", "0.5"))
    return KimiRTPConfig(
        reduction_layer_idx=int(os.getenv("REDUCTION_LAYER_IDX", "15")),
        token_merge_strategy=os.getenv("TOKEN_MERGE_STRATEGY", "none"),
        base_alpha=alpha,
        base_beta=1.0 - alpha,
        merge_layer_locs=merge_layer_locs,
        merge_token_nums=merge_token_nums,
        keep_token_ratios=keep_token_ratios,
        window_size=int(os.getenv("ROUTING_SIMILARITY_WINDOW_SIZE", "3")),
        merge_method=os.getenv("TOKEN_MERGE_METHOD", "mean"),
        merge_ratio=float(os.getenv("MERGE_RATIO", "0.5")),
    )


def _grouped_mean(values: torch.Tensor, window_size: int) -> torch.Tensor:
    num_values = values.shape[0]
    num_groups = math.ceil(num_values / window_size)
    remainder = num_values % window_size
    if remainder > 0:
        pad = window_size - remainder
        values = torch.cat([values, values.new_zeros(pad)], dim=0)
        valid_mask = torch.ones_like(values)
        valid_mask[-pad:] = 0
    else:
        valid_mask = None
    grouped = values.view(num_groups, window_size)
    if valid_mask is None:
        return grouped.mean(dim=1, keepdim=True)
    grouped_mask = valid_mask.view(num_groups, window_size)
    valid_counts = grouped_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return (grouped * grouped_mask).sum(dim=1, keepdim=True) / valid_counts


def _routing_similarity_scores(routing_weights: torch.Tensor, window_size: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    num_tokens = routing_weights.shape[0]
    num_groups = math.ceil(num_tokens / window_size)
    grouped_indices = [
        torch.arange(i * window_size, min((i + 1) * window_size, num_tokens), device=routing_weights.device)
        for i in range(num_groups)
    ]
    remainder = num_tokens % window_size
    if remainder > 0:
        pad = window_size - remainder
        mean_weight = routing_weights.mean(dim=0, keepdim=True)
        routing_weights = torch.cat([routing_weights, mean_weight.repeat(pad, 1)], dim=0)
        padded = True
    else:
        pad = 0
        padded = False

    groups_tensor = routing_weights.view(num_groups, window_size, routing_weights.shape[-1])
    mean_routing_weight = groups_tensor.mean(dim=1, keepdim=True)
    similarities = F.cosine_similarity(groups_tensor, mean_routing_weight, dim=-1)

    if padded:
        if remainder < 2:
            scores = similarities.mean(dim=1, keepdim=True)
            scores[-1] = 0
        else:
            mask = torch.ones_like(similarities)
            mask[-1, -pad:] = 0
            valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
            scores = (similarities * mask).sum(dim=1, keepdim=True) / valid_counts
    else:
        scores = similarities.mean(dim=1, keepdim=True)
        for i, rel_indices in enumerate(grouped_indices):
            if rel_indices.numel() < 2:
                scores[i] = 0
    return scores, grouped_indices


def _merge_group_vectors(
    values: torch.Tensor,
    grouped_absolute_indices: list[torch.Tensor],
    method: str,
) -> torch.Tensor:
    merged = []
    for group_indices in grouped_absolute_indices:
        group_values = values[group_indices]
        if method == "mean":
            merged.append(group_values.mean(dim=0, keepdim=True))
            continue
        if method == "mlerp":
            mean_vec = group_values.mean(dim=0, keepdim=True)
            mean_norm = mean_vec.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-10)
            mean_normalized = mean_vec / mean_norm
            max_norm = group_values.norm(p=2, dim=-1).amax().view(1, 1)
            merged.append(mean_normalized * max_norm)
            continue
        raise ValueError(f"Unsupported TOKEN_MERGE_METHOD={method}")
    return torch.cat(merged, dim=0) if merged else values.new_empty((0, values.shape[-1]))


def apply_kimi_rtp(
    *,
    layer_idx: int,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    image_seq_mask: Optional[torch.Tensor],
    attn_scores: Optional[torch.Tensor],
    gate_weight: torch.Tensor,
    scoring_func: str,
) -> KimiRTPResult:
    if not kimi_rtp_enabled():
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)
    cfg = load_kimi_rtp_config()
    if cfg.token_merge_strategy == "none":
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)
    if hidden_states.shape[0] != 1 or hidden_states.shape[1] <= 1:
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)
    if image_seq_mask is None or not image_seq_mask.any():
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)
    if layer_idx not in cfg.merge_layer_locs:
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)
    if scoring_func != "sigmoid":
        raise NotImplementedError("Kimi RTP currently expects sigmoid MoE gate scoring.")

    flat_hidden = hidden_states[0]
    flat_residual = residual[0]
    flat_input_ids = input_ids[0]
    flat_mask = image_seq_mask[0].to(torch.bool)
    num_vision_tokens = int(flat_mask.sum().item())
    if num_vision_tokens <= 1:
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)

    logits = F.linear(flat_hidden.float(), gate_weight.float(), None)
    routing_weights = logits.sigmoid()

    loc_idx = cfg.merge_layer_locs.index(layer_idx)
    target_token_num = cfg.merge_token_nums[loc_idx] if loc_idx < len(cfg.merge_token_nums) else cfg.merge_token_nums[-1]
    if cfg.keep_token_ratios is not None:
        keep_ratio = cfg.keep_token_ratios[loc_idx] if loc_idx < len(cfg.keep_token_ratios) else cfg.keep_token_ratios[-1]
        if keep_ratio < 1.0:
            target_token_num = max(1, int(num_vision_tokens * keep_ratio))
    if num_vision_tokens <= target_token_num:
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)

    merged_token_count = int(target_token_num * cfg.merge_ratio)
    kept_original_token_count = target_token_num - merged_token_count
    num_pruning_groups = math.ceil(num_vision_tokens / cfg.window_size)
    max_merge_groups = min(merged_token_count, num_pruning_groups - 1)
    if max_merge_groups <= 0:
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)

    vision_indices = flat_mask.nonzero(as_tuple=False).flatten()
    vision_routing = routing_weights[vision_indices]
    s_v, grouped_relative_indices = _routing_similarity_scores(vision_routing, cfg.window_size)

    a_v = torch.zeros_like(s_v)
    if attn_scores is not None and cfg.token_merge_strategy in {"dynamic_attention", "hybrid"}:
        reduced_scores = attn_scores.mean(dim=1)[0]
        attn_scores_vision_text = reduced_scores[-1]
        attn_scores_vision_text = attn_scores_vision_text[flat_mask]
        a_v = _grouped_mean(attn_scores_vision_text, cfg.window_size)
        a_v = a_v / (a_v.max() + 1e-10)

    if cfg.token_merge_strategy == "dynamic_attention":
        c_v = -a_v
    elif cfg.token_merge_strategy == "routing_similarity":
        c_v = s_v
    else:
        c_v = cfg.base_alpha * s_v - cfg.base_beta * a_v

    _, sorted_group_indices = torch.sort(c_v.view(-1), descending=True)
    merge_group_indices = sorted_group_indices[:max_merge_groups].tolist()
    merge_group_abs = [vision_indices[grouped_relative_indices[idx]] for idx in merge_group_indices]

    token_importance = vision_routing.norm(dim=-1)
    if attn_scores is not None and cfg.token_merge_strategy in {"dynamic_attention", "hybrid"}:
        reduced_scores = attn_scores.mean(dim=1)[0]
        token_importance = reduced_scores[-1][flat_mask].clone()

    merge_token_relative_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=flat_hidden.device)
    for idx in merge_group_indices:
        rel_indices = grouped_relative_indices[idx]
        merge_token_relative_mask[rel_indices] = True

    non_merge_relative_indices = torch.where(~merge_token_relative_mask)[0]
    non_merge_importance = token_importance[non_merge_relative_indices]
    num_to_keep = min(kept_original_token_count, int(non_merge_relative_indices.numel()))
    keep_token_relative_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=flat_hidden.device)
    if num_to_keep > 0 and non_merge_relative_indices.numel() > 0:
        if num_to_keep < non_merge_relative_indices.numel():
            _, keep_indices = torch.topk(non_merge_importance, num_to_keep, largest=True)
            keep_token_relative_mask[non_merge_relative_indices[keep_indices]] = True
        else:
            keep_token_relative_mask[non_merge_relative_indices] = True

    keep_token_absolute_indices = vision_indices[keep_token_relative_mask]
    merge_token_absolute_indices = torch.cat(merge_group_abs) if merge_group_abs else vision_indices.new_empty((0,))
    merged_hidden = _merge_group_vectors(flat_hidden, merge_group_abs, cfg.merge_method)
    merged_residual = _merge_group_vectors(flat_residual, merge_group_abs, cfg.merge_method)
    merged_input_ids = torch.stack([flat_input_ids[group[0]] for group in merge_group_abs]) if merge_group_abs else flat_input_ids.new_empty((0,))
    merged_position_ids = None
    if position_ids is not None:
        flat_pos = position_ids[0]
        merged_position_ids = torch.stack([flat_pos[group[0]] for group in merge_group_abs]) if merge_group_abs else flat_pos.new_empty((0,))

    tokens_to_remove = torch.ones(flat_hidden.shape[0], dtype=torch.bool, device=flat_hidden.device)
    tokens_to_remove[~flat_mask] = False
    tokens_to_remove[keep_token_absolute_indices] = False
    tokens_to_remove[merge_token_absolute_indices] = True

    remaining_hidden = flat_hidden[~tokens_to_remove]
    remaining_residual = flat_residual[~tokens_to_remove]
    remaining_input_ids = flat_input_ids[~tokens_to_remove]
    remaining_image_mask = flat_mask[~tokens_to_remove]
    remaining_attn_mask = attention_mask[0][~tokens_to_remove] if attention_mask is not None else None
    remaining_pos = position_ids[0][~tokens_to_remove] if position_ids is not None else None

    first_token_indices = [group[0] for group in merge_group_abs if group.numel() > 0]
    if not first_token_indices:
        return KimiRTPResult(hidden_states, residual, input_ids, attention_mask, position_ids, image_seq_mask, False)
    insert_positions = torch.stack(first_token_indices)
    cumsum_removed = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=flat_hidden.device),
            tokens_to_remove.long().cumsum(dim=0)[:-1],
        ]
    )
    adjusted_positions = insert_positions - cumsum_removed[insert_positions]
    sorted_indices = adjusted_positions.argsort()
    sorted_positions = adjusted_positions[sorted_indices].tolist()
    sorted_merge_idx = sorted_indices.tolist()

    parts_hidden = []
    parts_residual = []
    parts_input_ids = []
    parts_mask = []
    parts_attn = [] if remaining_attn_mask is not None else None
    parts_pos = [] if remaining_pos is not None else None
    start = 0
    for out_idx, insert_pos in enumerate(sorted_positions):
        end = int(insert_pos)
        parts_hidden.append(remaining_hidden[start:end])
        parts_residual.append(remaining_residual[start:end])
        parts_input_ids.append(remaining_input_ids[start:end])
        parts_mask.append(remaining_image_mask[start:end])
        if parts_attn is not None:
            parts_attn.append(remaining_attn_mask[start:end])
        if parts_pos is not None:
            parts_pos.append(remaining_pos[start:end])

        merge_idx = sorted_merge_idx[out_idx]
        parts_hidden.append(merged_hidden[merge_idx : merge_idx + 1])
        parts_residual.append(merged_residual[merge_idx : merge_idx + 1])
        parts_input_ids.append(merged_input_ids[merge_idx : merge_idx + 1])
        parts_mask.append(torch.ones(1, dtype=torch.bool, device=flat_hidden.device))
        if parts_attn is not None:
            parts_attn.append(torch.ones(1, dtype=remaining_attn_mask.dtype, device=flat_hidden.device))
        if parts_pos is not None:
            parts_pos.append(merged_position_ids[merge_idx : merge_idx + 1])
        start = end

    parts_hidden.append(remaining_hidden[start:])
    parts_residual.append(remaining_residual[start:])
    parts_input_ids.append(remaining_input_ids[start:])
    parts_mask.append(remaining_image_mask[start:])
    if parts_attn is not None:
        parts_attn.append(remaining_attn_mask[start:])
    if parts_pos is not None:
        parts_pos.append(remaining_pos[start:])

    new_hidden = torch.cat(parts_hidden, dim=0).unsqueeze(0)
    new_residual = torch.cat(parts_residual, dim=0).unsqueeze(0)
    new_input_ids = torch.cat(parts_input_ids, dim=0).unsqueeze(0)
    new_image_mask = torch.cat(parts_mask, dim=0).unsqueeze(0)
    new_attn = torch.cat(parts_attn, dim=0).unsqueeze(0) if parts_attn is not None else None
    new_pos = torch.cat(parts_pos, dim=0).unsqueeze(0) if parts_pos is not None else None
    return KimiRTPResult(new_hidden, new_residual, new_input_ids, new_attn, new_pos, new_image_mask, True)
