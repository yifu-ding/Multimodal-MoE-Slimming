from typing import Optional

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from src.base.models.transformers_compat import create_causal_mask
from src.calibration.representation_distill.common import (
    build_position_ids_from_attention_mask,
    get_final_norm,
    make_compact_position_ids,
)


def _qwen3_forward_from_hidden(
    bundle,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    start_layer: int,
    end_layer: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    apply_final_norm: bool = False,
):
    lm = bundle.model.model.language_model
    if position_ids is None:
        position_ids = build_position_ids_from_attention_mask(attention_mask)
    if position_ids.ndim == 2:
        rope_position_ids = position_ids.unsqueeze(0).expand(3, position_ids.shape[0], -1)
    elif position_ids.ndim == 3:
        rope_position_ids = position_ids
        position_ids = position_ids[0]
    else:
        raise ValueError(
            f"Unsupported position_ids shape for qwen3: {tuple(position_ids.shape)}"
        )

    cache_position = torch.arange(
        hidden_states.shape[1],
        device=hidden_states.device,
    )
    causal_mask = create_causal_mask(
        config=lm.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = lm.rotary_emb(hidden_states, rope_position_ids)
    stop_layer = len(lm.layers) if end_layer is None else end_layer + 1

    for layer_idx in range(start_layer, stop_layer):
        hidden_states = lm.layers[layer_idx](
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            moe_layer_skip_flag=False,
            skip_modality=None,
            enable_tau_skip=False,
            tau=None,
        )
    if apply_final_norm:
        hidden_states = get_final_norm(bundle)(hidden_states)
    return hidden_states


def _kimi_forward_from_hidden(
    bundle,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    start_layer: int,
    end_layer: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    apply_final_norm: bool = False,
):
    lm = bundle.model.language_model.model
    if position_ids is None:
        position_ids = build_position_ids_from_attention_mask(attention_mask)
    else:
        position_ids = make_compact_position_ids(position_ids, attention_mask)

    if lm._use_flash_attention_2:
        layer_attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        layer_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (hidden_states.shape[0], hidden_states.shape[1]),
            hidden_states,
            0,
        )

    stop_layer = len(lm.layers) if end_layer is None else end_layer + 1
    for layer_idx in range(start_layer, stop_layer):
        layer_outputs = lm.layers[layer_idx](
            hidden_states=hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            moe_layer_skip_flag=False,
            skip_modality=None,
            skip_expert_idx=None,
            enable_tau_skip=False,
            tau={},
        )
        hidden_states = layer_outputs[0]
    if apply_final_norm:
        hidden_states = get_final_norm(bundle)(hidden_states)
    return hidden_states


def _deepseek_vl_forward_from_hidden(
    bundle,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    start_layer: int,
    end_layer: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    apply_final_norm: bool = False,
):
    lm = bundle.model.language.model
    if position_ids is None:
        position_ids = build_position_ids_from_attention_mask(attention_mask)
    else:
        position_ids = make_compact_position_ids(position_ids, attention_mask)

    if getattr(bundle.model, "_use_flash_attention_2", False):
        layer_attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        layer_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (hidden_states.shape[0], hidden_states.shape[1]),
            hidden_states,
            0,
        )

    stop_layer = len(lm.layers) if end_layer is None else end_layer + 1
    for layer_idx in range(start_layer, stop_layer):
        layer_outputs = lm.layers[layer_idx](
            hidden_states=hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
    if apply_final_norm:
        hidden_states = get_final_norm(bundle)(hidden_states)
    return hidden_states


def forward_from_hidden(
    bundle,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    start_layer: int,
    end_layer: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    apply_final_norm: bool = False,
):
    if bundle.family == "qwen3":
        return _qwen3_forward_from_hidden(
            bundle=bundle,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            start_layer=start_layer,
            end_layer=end_layer,
            position_ids=position_ids,
            apply_final_norm=apply_final_norm,
        )
    if bundle.family == "kimi":
        return _kimi_forward_from_hidden(
            bundle=bundle,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            start_layer=start_layer,
            end_layer=end_layer,
            position_ids=position_ids,
            apply_final_norm=apply_final_norm,
        )
    if bundle.family == "deepseek_vl":
        return _deepseek_vl_forward_from_hidden(
            bundle=bundle,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            start_layer=start_layer,
            end_layer=end_layer,
            position_ids=position_ids,
            apply_final_norm=apply_final_norm,
        )
    raise NotImplementedError(
        f"`forward_from_hidden` currently supports qwen3/kimi/deepseek_vl only, got family={bundle.family}."
    )
