import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

import torch

from src.calibration.helpers.helpers import teacher_blocks
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


def compress_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    pooled_batch = []
    mask = attention_mask.to(torch.bool)
    for batch_idx in range(hidden_states.shape[0]):
        valid_hidden = hidden_states[batch_idx][mask[batch_idx]]
        pooled_batch.append(_pool_single_sequence(valid_hidden.float(), target_length))
    return torch.stack(pooled_batch, dim=0)


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


def extract_block_output(bundle, inputs: Dict[str, Any], layer_idx: int) -> torch.Tensor:
    state: Dict[str, Any] = {}
    block = get_decoder_layer(bundle, layer_idx)

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

    if "output" not in state:
        raise RuntimeError(f"Failed to capture output of layer {layer_idx}.")
    return state["output"]


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
