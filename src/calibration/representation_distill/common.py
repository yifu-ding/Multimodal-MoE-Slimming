import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

import torch

from src.calibration.helpers.helpers import resolve_media_token_ids, teacher_blocks
from src.calibration.helpers.utils import unwrap_output


class EarlyStopForward(RuntimeError):
    """Internal control-flow exception used to stop forward after one block."""


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_decoder_layers(bundle) -> Iterable[torch.nn.Module]:
    return teacher_blocks(bundle)


def get_decoder_layer(bundle, layer_idx: int):
    return list(get_decoder_layers(bundle))[layer_idx]


def get_num_decoder_layers(bundle) -> int:
    return len(list(get_decoder_layers(bundle)))


def get_hidden_size(bundle) -> int:
    config = getattr(bundle.model, "config", None)
    text_config = getattr(config, "text_config", config)
    for attr in ("hidden_size", "d_model", "model_dim"):
        value = getattr(text_config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("Cannot infer hidden size from model config.")


def get_final_norm(bundle):
    if bundle.family == "qwen3":
        return bundle.model.model.language_model.norm
    if bundle.family == "kimi":
        return bundle.model.language_model.model.norm
    raise NotImplementedError(
        f"`forward_from_hidden` currently supports qwen3/kimi only, got family={bundle.family}."
    )


def discover_moe_layers(bundle) -> List[int]:
    layers = list(get_decoder_layers(bundle))
    if bundle.family == "qwen3":
        config = bundle.model.config.text_config
        return [
            layer_idx
            for layer_idx in range(len(layers))
            if (
                getattr(config, "num_experts", 0) > 0
                and (layer_idx + 1) % config.decoder_sparse_step == 0
                and layer_idx not in getattr(config, "mlp_only_layers", [])
            )
        ]
    if bundle.family == "kimi":
        config = bundle.model.config.text_config
        return [
            layer_idx
            for layer_idx in range(len(layers))
            if (
                getattr(config, "n_routed_experts", None) is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
        ]
    return []


def build_position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D attention_mask, got shape={tuple(attention_mask.shape)}"
        )
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.clamp_min(0)
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids


def _pool_single_sequence(hidden: torch.Tensor, target_length: int) -> torch.Tensor:
    seq_len, hidden_size = hidden.shape
    if target_length < 0:
        raise ValueError(f"target_length must be non-negative, got {target_length}")
    if target_length == 0:
        return hidden.new_zeros((0, hidden_size))
    if seq_len == 0:
        return hidden.new_zeros((target_length, hidden_size))
    if seq_len == target_length:
        return hidden
    if seq_len < target_length:
        indices = torch.linspace(
            0,
            seq_len - 1,
            steps=target_length,
            device=hidden.device,
        ).round().long()
        return hidden.index_select(0, indices)

    boundaries = torch.linspace(
        0,
        seq_len,
        steps=target_length + 1,
        device=hidden.device,
    ).floor().long()
    pooled = []
    for bucket_idx in range(target_length):
        start = int(boundaries[bucket_idx].item())
        end = int(boundaries[bucket_idx + 1].item())
        if end <= start:
            end = min(start + 1, seq_len)
        pooled.append(hidden[start:end].mean(dim=0))
    return torch.stack(pooled, dim=0)


def _sample_single_sequence(
    hidden: torch.Tensor,
    target_length: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample *target_length* tokens from a sequence **without replacement**.

    Returns
    -------
    sampled_hidden : (target_length, H)
    selected_indices : (target_length,) long – indices into the original sequence
    """
    seq_len, hidden_size = hidden.shape
    if target_length <= 0:
        return hidden.new_zeros((0, hidden_size)), torch.zeros(0, dtype=torch.long, device=hidden.device)
    if seq_len == 0:
        return hidden.new_zeros((target_length, hidden_size)), torch.zeros(
            target_length, dtype=torch.long, device=hidden.device
        )
    if seq_len <= target_length:
        indices = torch.arange(target_length, device=hidden.device) % seq_len
        return hidden[indices], indices
    # torch.randperm requires generator and tensor to be on the same device;
    # generate on CPU then move to avoid device mismatch.
    indices = torch.randperm(seq_len, generator=generator)[:target_length]
    indices, _ = indices.sort()
    indices = indices.to(hidden.device)
    return hidden[indices], indices


def _weighted_sample_single_sequence(
    hidden: torch.Tensor,
    target_length: int,
    importance: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample *target_length* tokens with probability proportional to *importance*.

    Parameters
    ----------
    hidden : (seq_len, H)
    importance : (seq_len,) non-negative weights.
    temperature : controls sharpness. Larger → more uniform; smaller → more greedy.
        ``inf`` degrades to uniform random sampling; ``0`` degrades to top-k.

    Returns (sampled_hidden, selected_indices) sorted by position.
    """
    # 按 importance 做 softmax 采样 target_length 个位置 (无放回), 再按原序列位置排序返回.
    seq_len, hidden_size = hidden.shape
    if target_length <= 0:
        return hidden.new_zeros((0, hidden_size)), torch.zeros(0, dtype=torch.long, device=hidden.device)
    if seq_len == 0:
        return hidden.new_zeros((target_length, hidden_size)), torch.zeros(
            target_length, dtype=torch.long, device=hidden.device,
        )
    # 有效 token 不足或刚好: 循环取位置填满长度.
    if seq_len <= target_length:
        indices = torch.arange(target_length, device=hidden.device) % seq_len
        return hidden[indices], indices

    # 由 importance 构造分布, temperature 越大分布越平, 越小越贴近 top importance.
    imp = importance.float().clamp(min=0)
    if temperature <= 0 or imp.sum() == 0:
        # Degenerate: fall back to top-k or uniform
        # 权重全 0: 退化为均匀随机; 否则退化为 importance 最大的 top-k 个下标.
        if imp.sum() == 0:
            indices = torch.randperm(seq_len, device=hidden.device)[:target_length]
        else:
            indices = imp.topk(target_length).indices
    else:
        log_probs = (imp + 1e-8).log() / temperature
        probs = torch.softmax(log_probs, dim=-1)
        indices = torch.multinomial(probs, num_samples=target_length, replacement=False)

    # 与 _sample_single_sequence 一致: 按原序列下标排序, 保证输出 token 随位置递增.
    indices, _ = indices.sort()
    return hidden[indices], indices


@dataclass
class CompressedResult:
    """Return type of :func:`compress_hidden_states`."""

    hidden_states: torch.Tensor   # (batch, target_length, H)
    modality_labels: torch.Tensor  # (batch, target_length) int8; 0=text, 1=image, 2=video


def _resolve_visual_token_masks(bundle, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    config = bundle.model.config
    image_token_id = getattr(config, "image_token_id", None)
    video_token_id = getattr(config, "video_token_id", None)

    image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    video_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    if image_token_id is not None:
        image_mask |= input_ids == int(image_token_id)
    if video_token_id is not None:
        video_mask |= input_ids == int(video_token_id)

    visual_mask = image_mask | video_mask
    if not visual_mask.any():
        media_token_ids = resolve_media_token_ids(bundle)
        if media_token_ids:
            media_token_tensor = torch.tensor(
                media_token_ids,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            visual_mask = torch.isin(input_ids, media_token_tensor)
            if not image_mask.any() and not video_mask.any():
                image_mask = visual_mask

    return {
        "image": image_mask,
        "video": video_mask,
        "visual": visual_mask,
    }


def build_compression_token_masks(
    bundle,
    inputs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    attention_mask = inputs["attention_mask"].to(torch.bool)
    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        return {
            "text": attention_mask,
            "image": torch.zeros_like(attention_mask),
            "video": torch.zeros_like(attention_mask),
            "visual": torch.zeros_like(attention_mask),
        }

    visual_masks = _resolve_visual_token_masks(bundle, input_ids)
    return {
        "text": attention_mask & ~visual_masks["visual"],
        "image": attention_mask & visual_masks["image"],
        "video": attention_mask & visual_masks["video"],
        "visual": attention_mask & visual_masks["visual"],
    }


def _compute_effective_modality_lengths(
    compression_masks: Dict[str, torch.Tensor],
    batch_idx: int,
    target_length: int,
    modality_lengths: Dict[str, int] | None = None,
) -> Dict[str, int]:
    """Resolve per-modality token budgets for a single sample."""
    modality_names = ("text", "image", "video")
    if modality_lengths is not None:
        return dict(modality_lengths)

    token_counts = {
        name: int(compression_masks[name][batch_idx].sum().item())
        for name in modality_names
    }
    total_tokens = sum(token_counts.values())
    if total_tokens <= 0:
        return {"text": target_length, "image": 0, "video": 0}

    raw_lengths = {
        name: target_length * token_counts[name] / total_tokens
        for name in modality_names
    }
    effective = {name: int(raw_lengths[name]) for name in modality_names}
    assigned = sum(effective.values())
    remainder = target_length - assigned
    if remainder > 0:
        ranked = sorted(
            modality_names,
            key=lambda n: (raw_lengths[n] - effective[n], token_counts[n]),
            reverse=True,
        )
        for idx in range(remainder):
            effective[ranked[idx % len(ranked)]] += 1
    elif remainder < 0:
        ranked = sorted(
            modality_names,
            key=lambda n: (effective[n] - raw_lengths[n], effective[n]),
            reverse=True,
        )
        to_remove = -remainder
        ri = 0
        while to_remove > 0 and ri < len(ranked) * max(target_length, 1):
            n = ranked[ri % len(ranked)]
            if effective[n] > 0:
                effective[n] -= 1
                to_remove -= 1
            ri += 1
    return effective


# Modality label constants
MODALITY_TEXT: int = 0
MODALITY_IMAGE: int = 1
MODALITY_VIDEO: int = 2


def compress_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    target_length: int,
    compression_masks: Dict[str, torch.Tensor] | None = None,
    modality_lengths: Dict[str, int] | None = None,
    mode: str = "pool",
    generator: torch.Generator | None = None,
    attn_importance: torch.Tensor | None = None,
    attn_temperature: float = 1.0,
) -> CompressedResult:
    """Compress hidden states to *target_length* tokens per sample.

    Parameters
    ----------
    mode :
        ``"pool"`` — mean-pooling (legacy).
        ``"sample"`` — uniform random token sampling.
        ``"attention_weighted"`` — importance-weighted sampling guided by
        attention scores.
    generator : optional torch.Generator for reproducible random sampling.
    attn_importance : (batch, seq_len) per-token importance scores.
        Required when ``mode="attention_weighted"``.
    attn_temperature : softmax temperature for importance-weighted sampling.
        Larger → more uniform; smaller → more greedy toward high-importance
        tokens.  Default 1.0.

    Returns a :class:`CompressedResult` with *hidden_states* and *modality_labels*.
    """
    # 将每条样本的 seq 维压到 target_length, 可选按模态分别压缩再拼接, 并输出逐 token 模态标签.
    valid_modes = ("pool", "sample", "attention_weighted")
    if mode not in valid_modes:
        raise ValueError(f"Unsupported compression mode: {mode!r}")
    if mode == "attention_weighted" and attn_importance is None:
        raise ValueError("attn_importance is required when mode='attention_weighted'.")

    batch_size = hidden_states.shape[0]
    mask = attention_mask.to(torch.bool)

    # 模态感知路径: 需提供 text/image/video 掩码, 若给 modality_lengths 则三模态长度之和须等于 target_length.
    if compression_masks is not None:
        for required_key in ("text", "image", "video"):
            if required_key not in compression_masks:
                raise ValueError(f"Missing compression mask for modality={required_key}.")
            if modality_lengths is not None and required_key not in modality_lengths:
                raise ValueError(f"Missing modality length for modality={required_key}.")
        if modality_lengths is not None:
            expected_length = sum(int(modality_lengths[name]) for name in ("text", "image", "video"))
            if expected_length != target_length:
                raise ValueError(
                    f"Sum of modality_lengths ({expected_length}) must equal target_length ({target_length})."
                )

    _modality_id = {"text": MODALITY_TEXT, "image": MODALITY_IMAGE, "video": MODALITY_VIDEO}
    compressed_batch: list[torch.Tensor] = []
    labels_batch: list[torch.Tensor] = []

    def _select(hidden_1d: torch.Tensor, length: int, imp_1d: torch.Tensor | None):
        """Dispatch to the right selection strategy."""
        # 单段序列 (已按掩码取出的有效 token) 上执行 pool / 均匀采样 / 注意力加权采样之一.
        if mode == "pool":
            return _pool_single_sequence(hidden_1d.float(), length)
        if mode == "attention_weighted" and imp_1d is not None:
            sampled, _ = _weighted_sample_single_sequence(
                hidden_1d.float(), length, imp_1d, temperature=attn_temperature,
            )
            return sampled
        # mode == "sample" or fallback
        sampled, _ = _sample_single_sequence(hidden_1d.float(), length, generator)
        return sampled

    for batch_idx in range(batch_size):
        batch_imp = attn_importance[batch_idx] if attn_importance is not None else None

        if compression_masks is None:
            # 非模态感知: 仅用 attention_mask 取有效 token, 整段压到 target_length, 标签全标为 TEXT.
            valid_mask = mask[batch_idx]
            valid_hidden = hidden_states[batch_idx][valid_mask]
            valid_imp = batch_imp[valid_mask] if batch_imp is not None else None
            compressed_batch.append(_select(valid_hidden, target_length, valid_imp))
            labels_batch.append(
                torch.full((target_length,), MODALITY_TEXT, dtype=torch.int8, device=hidden_states.device)
            )
            continue

        # 模态感知: 为各模态分配长度预算, 再分别压缩后沿 seq 维拼接.
        eff_lengths = _compute_effective_modality_lengths(
            compression_masks, batch_idx, target_length, modality_lengths,
        )

        parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        for modality_name in ("text", "image", "video"):
            mod_mask = compression_masks[modality_name][batch_idx]
            modality_hidden = hidden_states[batch_idx][mod_mask]
            mod_imp = batch_imp[mod_mask] if batch_imp is not None else None
            ml = int(eff_lengths[modality_name])
            parts.append(_select(modality_hidden, ml, mod_imp))
            label_parts.append(
                torch.full((ml,), _modality_id[modality_name], dtype=torch.int8, device=hidden_states.device)
            )
        compressed_batch.append(torch.cat(parts, dim=0))
        labels_batch.append(torch.cat(label_parts, dim=0))

    return CompressedResult(
        hidden_states=torch.stack(compressed_batch, dim=0),
        modality_labels=torch.stack(labels_batch, dim=0),
    )


def build_chat_messages(raw_samples: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    messages = []
    for sample in raw_samples:
        content: List[Dict[str, Any]] = []
        for _ in sample.get("images", []):
            content.append({"type": "image", "image": "placeholder"})
        for _ in sample.get("video_frames", []):
            content.append({"type": "image", "image": "placeholder"})
        content.append({"type": "text", "text": sample["text"]})
        messages.append([{"role": "user", "content": content}])
    return messages


def prepare_raw_batch_inputs(bundle, raw_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    processor = bundle.processor
    messages = build_chat_messages(raw_samples)
    prompts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    flat_images = []
    for sample in raw_samples:
        flat_images.extend(sample.get("images", []))
        flat_images.extend(sample.get("video_frames", []))

    processor_kwargs = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "padding_side": "left",
    }
    if flat_images:
        processor_kwargs["images"] = flat_images
    return processor(**processor_kwargs)


def move_inputs_to_model_device(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@dataclass
class BlockExtractionResult:
    """Return type of :func:`extract_block_output`."""

    hidden_states: torch.Tensor                       # (batch, seq_len, H)
    attn_importance: torch.Tensor | None = None       # (batch, seq_len) float


def extract_block_output(
    bundle,
    inputs: Dict[str, Any],
    layer_idx: int,
    capture_attn_importance: bool = False,
) -> BlockExtractionResult:
    """Extract the output of decoder block *layer_idx*.

    When *capture_attn_importance* is True, also hooks into the self-attention
    sub-layer to compute a per-token importance score (L2-norm of the attention
    output, i.e. how much information the attention layer writes into each
    token's residual stream).
    """
    state: Dict[str, Any] = {}
    block = get_decoder_layer(bundle, layer_idx)

    # --- optional: hook self_attn to capture per-token importance ----------
    attn_handle = None
    if capture_attn_importance:
        attn_module = getattr(block, "self_attn", None)
        if attn_module is not None:
            def _capture_attn(module, args, kwargs, output):
                attn_out = unwrap_output(output).detach().float()
                # L2 norm per token → importance score
                state["attn_importance"] = attn_out.norm(dim=-1)  # (batch, seq_len)

            attn_handle = attn_module.register_forward_hook(
                _capture_attn, with_kwargs=True,
            )

    # --- hook block output and early-stop ----------------------------------
    def _capture_and_stop(module, args, kwargs, output):
        state["output"] = unwrap_output(output).detach()
        raise EarlyStopForward()

    handle = block.register_forward_hook(_capture_and_stop, with_kwargs=True)
    try:
        with torch.no_grad():
            bundle.model(**inputs, use_cache=False, return_dict=True)
    except EarlyStopForward:
        pass
    finally:
        handle.remove()
        if attn_handle is not None:
            attn_handle.remove()

    if "output" not in state:
        raise RuntimeError(f"Failed to capture output of layer {layer_idx}.")
    return BlockExtractionResult(
        hidden_states=state["output"],
        attn_importance=state.get("attn_importance"),
    )


def build_sample_manifest(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest = []
    for sample in samples:
        manifest.append(
            {
                "dataset_name": sample["dataset_name"],
                "dataset_index": int(sample["dataset_index"]),
                "sample_id": sample["sample_id"],
                "task_type": sample.get("task_type"),
                "num_images": len(sample.get("images", [])),
                "num_video_frames": len(sample.get("video_frames", [])),
            }
        )
    return manifest
