from __future__ import annotations

from typing import Optional

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

try:
    from transformers.masking_utils import create_causal_mask as _hf_create_causal_mask
except Exception:
    _hf_create_causal_mask = None


def create_causal_mask(
    *,
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    past_key_values,
    position_ids: Optional[torch.Tensor] = None,
):
    """Compatibility wrapper for transformers causal-mask creation.

    Newer transformers expose `transformers.masking_utils.create_causal_mask`.
    Older releases such as 4.38 do not, so we fall back to the legacy
    `_prepare_4d_causal_attention_mask` helper when needed.
    """
    if _hf_create_causal_mask is not None:
        return _hf_create_causal_mask(
            config=config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    past_key_values_length = 0
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            past_key_values_length = int(past_key_values.get_seq_length())
        elif hasattr(past_key_values, "__len__") and len(past_key_values) > 0:
            first_layer_cache = past_key_values[0]
            if isinstance(first_layer_cache, (tuple, list)) and len(first_layer_cache) > 0:
                past_key_values_length = int(first_layer_cache[0].shape[-2])

    sliding_window = getattr(config, "sliding_window", None)
    return _prepare_4d_causal_attention_mask(
        attention_mask=attention_mask,
        input_shape=input_embeds.shape[:2],
        inputs_embeds=input_embeds,
        past_key_values_length=past_key_values_length,
        sliding_window=sliding_window,
    )
