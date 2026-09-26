import argparse
import importlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import random
import sys
from glob import glob
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets import config as datasets_config
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
if REPO_PARENT not in sys.path:
    sys.path.insert(0, REPO_PARENT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.coco import coco_transform
from tasks.dataset_paths import require_dataset_dir
from tasks.gqa import gqa_transform, load_gqa_instruction_rows, resolve_gqa_subdir
from tasks.star import load_star_subset_rows, star_transform
from src.base.masking import create_mask_after_last_token, create_mask_after_token


MODALITIES = ("text", "visual")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_tensor_dict(path: str, payload: Dict[str, Any]) -> None:
    torch.save(payload, path)


def _safe_top_expert_id(counts: torch.Tensor) -> Tuple[int, float]:
    if counts.numel() == 0:
        return -1, 0.0
    expert_idx = int(torch.argmax(counts).item())
    return expert_idx, float(counts[expert_idx].item())


def print_probe_snapshot(accumulator: "ObservationAccumulator", stage: str) -> None:
    if not accumulator.layers:
        print(f"[统计probe] {stage}：当前没有发现可观测的 MoE layer。", flush=True)
        return

    probe_layer = accumulator.layers[0]
    text_counts = accumulator.routing_counts[probe_layer]["text"]
    visual_counts = accumulator.routing_counts[probe_layer]["visual"]
    text_tokens = accumulator.token_counts[probe_layer]["text"]
    visual_tokens = accumulator.token_counts[probe_layer]["visual"]
    text_expert, text_hits = _safe_top_expert_id(text_counts)
    visual_expert, visual_hits = _safe_top_expert_id(visual_counts)
    active_text_experts = int((text_counts > 0).sum().item())
    active_visual_experts = int((visual_counts > 0).sum().item())

    print(
        f"[O1 - 统计probe] {stage}：观察 layer={probe_layer}。"
        f" 累计 text token={int(text_tokens)}，visual token={int(visual_tokens)}。",
        flush=True,
    )
    print(
        f"[O1 - 统计probe] layer={probe_layer} 上，text token 目前最常路由到 expert={text_expert} "
        f"(累计命中 {text_hits:.0f} 次)，已有 {active_text_experts}/{text_counts.numel()} 个 expert 收到过 text token。",
        flush=True,
    )
    print(
        f"[O1 - 统计probe] layer={probe_layer} 上，visual token 目前最常路由到 expert={visual_expert} "
        f"(累计命中 {visual_hits:.0f} 次)，已有 {active_visual_experts}/{visual_counts.numel()} 个 expert 收到过 visual token。",
        flush=True,
    )

    if text_expert >= 0:
        text_channel_visits = float(accumulator.channel_count[probe_layer]["text"][text_expert].item())
        print(
            f"[O2 - 统计probe] 以 text-top expert={text_expert} 为例，当前已有 {text_channel_visits:.0f} 个 "
            "text-routed token 被用于累计该 expert 的通道响应。",
            flush=True,
        )


def print_saved_artifact_message(path: str, description: str) -> None:
    print(f"[结果输出] 已保存 {description}: {path}", flush=True)


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    collated = {}
    for key in batch[0].keys():
        collated[key] = [row[key] for row in batch]
    return collated


class TransformedListDataset:
    def __init__(self, rows: List[Dict[str, Any]], transform: Callable[[Dict[str, List[Any]]], Dict[str, List[Any]]]):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        batch = {key: [value] for key, value in row.items()}
        transformed = self.transform(batch)
        item = {}
        for key, value in transformed.items():
            if isinstance(value, list) and len(value) == 1:
                item[key] = value[0]
            else:
                item[key] = value
        return item


def ensure_writable_datasets_cache() -> None:
    cache_root = os.environ.get("HF_DATASETS_CACHE") or datasets_config.HF_DATASETS_CACHE
    if cache_root and os.access(cache_root, os.W_OK):
        return

    fallback = os.path.join(".cache", "hf_datasets_cache")
    os.makedirs(fallback, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = fallback
    datasets_config.HF_DATASETS_CACHE = fallback

    downloaded = os.path.join(fallback, "downloads")
    extracted = os.path.join(fallback, "extracted")
    os.makedirs(downloaded, exist_ok=True)
    os.makedirs(extracted, exist_ok=True)
    datasets_config.DOWNLOADED_DATASETS_PATH = downloaded
    datasets_config.EXTRACTED_DATASETS_PATH = extracted


def infer_model_family(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    if "kimi-vl" in name:
        return "kimi"
    if "qwen3-vl" in name or "qwen3.5-35b-a3b" in name:
        return "qwen3"
    if "internvl" in name:
        return "internvl"
    if "deepseek-vl" in name:
        return "deepseek_vl"
    raise ValueError(f"Unsupported model family for path: {model_name_or_path}")


def _candidate_hf_cache_roots() -> List[str]:
    roots = []
    hf_home = os.environ.get("HF_HOME")
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    if hf_hub_cache:
        roots.append(hf_hub_cache)
    roots.append(os.path.join(".cache", "huggingface", "hub"))
    dedup = []
    for root in roots:
        if root and root not in dedup:
            dedup.append(root)
    return dedup


def _find_snapshot_from_hf_cache(model_id: str) -> Optional[str]:
    repo_dir = "models--" + model_id.replace("/", "--")
    for hub_root in _candidate_hf_cache_roots():
        repo_root = os.path.join(hub_root, repo_dir)
        snapshots_dir = os.path.join(repo_root, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        ref_main = os.path.join(repo_root, "refs", "main")
        if os.path.isfile(ref_main):
            with open(ref_main, "r", encoding="utf-8") as f:
                revision = f.read().strip()
            candidate = os.path.join(snapshots_dir, revision)
            if os.path.isdir(candidate):
                return candidate
        snapshots = sorted(
            [path for path in glob(os.path.join(snapshots_dir, "*")) if os.path.isdir(path)]
        )
        if snapshots:
            return snapshots[-1]
    return None


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    if "/" in model_name_or_path and not os.path.isabs(model_name_or_path):
        snapshot = _find_snapshot_from_hf_cache(model_name_or_path)
        if snapshot is not None:
            return snapshot
        return model_name_or_path
    if os.path.isabs(model_name_or_path):
        basename = os.path.basename(model_name_or_path.rstrip("/"))
        known_model_ids = [
            "moonshotai/Kimi-VL-A3B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3.5-35B-A3B",
            "OpenGVLab/InternVL-3.5-GPT-OSS-20B-A4B-Preview-HF",
            "OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF",
            "deepseek-ai/deepseek-vl2-small",
        ]
        for model_id in known_model_ids:
            if model_id.split("/")[-1] == basename:
                snapshot = _find_snapshot_from_hf_cache(model_id)
                if snapshot is not None:
                    return snapshot
        raise FileNotFoundError(
            f"Model path does not exist: {model_name_or_path}. "
            "If the model was downloaded to HF cache instead of --local-dir, "
            "pass the repo id (for example `moonshotai/Kimi-VL-A3B-Instruct`)."
        )
    return model_name_or_path


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    aliases = {
        "vmmmu": "video_mmmu",
        "m4": "m4_instruct",
        "m4-instruct": "m4_instruct",
        "m4_instruct_data": "m4_instruct",
        "star_train_subset_256": "star",
        "star_train_subset": "star",
    }
    return aliases.get(normalized, normalized)


def _resolve_attn_implementation(attn_implementation: str | None) -> str | None:
    if attn_implementation != "flash_attention_2":
        return attn_implementation
    if importlib.util.find_spec("flash_attn") is not None:
        try:
            # `flash_attn` may be installed while its compiled extension is ABI-incompatible
            # with the current torch build. Validate the actual CUDA extension import before use.
            importlib.import_module("flash_attn_2_cuda")
            return attn_implementation
        except Exception as exc:
            print(
                "[model-load] `flash_attn` is installed but unusable in the current environment; "
                f"falling back from `flash_attention_2` to `sdpa`. Import error: {exc!r}",
                flush=True,
            )
            return "sdpa"
    print(
        "[model-load] `flash_attn` is unavailable in the current environment; "
        "falling back from `flash_attention_2` to `sdpa`.",
        flush=True,
    )
    return "sdpa"


def _get_deepseek_decoder_layers(model):
    candidates = [
        ("language", "model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("language", "layers"),
        ("language_model", "layers"),
    ]
    for path in candidates:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            return current
    raise AttributeError(f"Cannot resolve DeepSeek decoder layers from model type {type(model)}")


def _get_deepseek_language_model(model):
    candidates = [
        ("language",),
        ("language_model",),
        ("model", "language_model"),
    ]
    for path in candidates:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            return current
    raise AttributeError(f"Cannot resolve DeepSeek language model from model type {type(model)}")


def _get_deepseek_text_config(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError(f"Model {type(model)} has no `config` for DeepSeek text config resolution.")
    candidates = [
        getattr(cfg, "text_config", None),
        getattr(cfg, "language_config", None),
        getattr(cfg, "lang_config", None),
        getattr(cfg, "llm_config", None),
        cfg,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if any(
            hasattr(candidate, attr)
            for attr in ("first_k_dense_replace", "moe_layer_freq", "n_routed_experts", "eos_token_id")
        ):
            return candidate
    return cfg


def _get_qwen3_text_config(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError(f"Model {type(model)} has no `config` for Qwen3 config resolution.")
    return getattr(cfg, "text_config", cfg)


def _get_internvl_text_config(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError(f"Model {type(model)} has no `config` for InternVL config resolution.")
    return getattr(cfg, "text_config", getattr(cfg, "llm_config", cfg))


def _get_internvl_language_model(model):
    candidates = [
        ("model", "language_model"),
        ("language_model",),
    ]
    for path in candidates:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            return current
    raise AttributeError(f"Cannot resolve InternVL language model from model type {type(model)}")


def _get_internvl_decoder_layers(model):
    return _get_internvl_language_model(model).layers


def _get_qwen3_decoder_layers(model):
    candidates = [
        ("model", "language_model", "layers"),  # Qwen3-VL
        ("model", "layers"),  # Qwen3 text
        ("language_model", "layers"),
        ("layers",),
    ]
    for path in candidates:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            return current
    raise AttributeError(f"Cannot resolve Qwen3 decoder layers from model type {type(model)}")


def build_dataset(dataset_name: str, model_family: str, **kwargs):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "gqa":
        return TransformedListDataset(load_gqa_instruction_rows(), gqa_transform)
    if dataset_name == "coco":
        data = load_dataset(
            require_dataset_dir("COCO-Caption2017", "data"), token=True
        )["validation"]
        data.set_transform(coco_transform)
        return data
    if dataset_name == "video_mmmu":
        from tasks.video_mmmu import videommmu_transform

        adaptation = load_dataset(
            require_dataset_dir("VideoMMMU", "Adaptation"), token=True
        )["test"]
        comprehension = load_dataset(
            require_dataset_dir("VideoMMMU", "Comprehension"), token=True
        )["test"]
        perception = load_dataset(
            require_dataset_dir("VideoMMMU", "Perception"), token=True
        )["test"]
        data = concatenate_datasets([adaptation, comprehension, perception])
        data.set_transform(videommmu_transform)
        return data
    if dataset_name == "m4_instruct":
        from tasks.m4_instruct import load_m4_instruct_rows, m4_instruct_transform

        max_rows = kwargs.get("max_rows", 1024)
        return TransformedListDataset(load_m4_instruct_rows(max_rows=max_rows), m4_instruct_transform)
    if dataset_name == "star":
        rows = [{"__raw_doc__": row} for row in load_star_subset_rows()]
        return TransformedListDataset(rows, star_transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


@dataclass
class ModelBundle:
    family: str
    model: Any
    processor: Any
    text_to_message: Callable[[str], List[Dict[str, Any]]]
    model_config: Dict[str, Any]


def load_model_bundle(
    model_name_or_path: str,
    *,
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
    max_decoder_layer: int | None = None,
) -> ModelBundle:
    resolved_name_or_path = resolve_model_name_or_path(model_name_or_path)
    family = infer_model_family(model_name_or_path)
    attn_implementation = _resolve_attn_implementation(attn_implementation)
    text_to_message = lambda text: [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    if family == "kimi":
        from src.base.models.kimi import load_model as load_kimi_model

        model, processor = load_kimi_model(resolved_name_or_path,
                                            device_map=device_map,
                                            attn_implementation=attn_implementation,
                                            max_decoder_layer=max_decoder_layer,
                                           )
        model_config = {
            "family": family,
            "get_lm": lambda m: m.language_model,
            "is_moe_layer": lambda cfg, idx: idx >= cfg.first_k_dense_replace
            and idx % cfg.moe_layer_freq == 0,
            "eos_token": "<|im_end|>[EOS]",
            "create_mask": partial(
                create_mask_after_token, special_token_id=163588, offset=3
            ),
        }
    elif family == "qwen3":
        from src.base.models.qwen3 import load_model as load_qwen3_model

        model, processor = load_qwen3_model(resolved_name_or_path, 
                                            device_map=device_map, 
                                            attn_implementation=attn_implementation)
        qwen_cfg = _get_qwen3_text_config(model)
        supports_vision = bool(
            hasattr(model, "visual")
            or hasattr(getattr(model, "model", None), "visual")
            or getattr(model.config, "vision_config", None) is not None
        )
        if not supports_vision:
            text_to_message = lambda text: [{"role": "user", "content": text}]
        eos_token = getattr(processor, "eos_token", None) or "<|im_end|>"
        model_config = {
            "family": family,
            "get_lm": lambda m: getattr(getattr(m, "model", None), "language_model", getattr(m, "model", m)),
            "is_moe_layer": lambda cfg, idx: (
                getattr(cfg, "num_experts", 0) > 0
                and (idx + 1) % getattr(cfg, "decoder_sparse_step", 1) == 0
                and idx not in getattr(cfg, "mlp_only_layers", [])
            ),
            "eos_token": eos_token,
            "create_mask": partial(
                create_mask_after_last_token, special_token_id=151644, offset=3
            ),
            "supports_vision": supports_vision,
            "qwen_text_config": qwen_cfg,
        }
    elif family == "deepseek_vl":
        from src.base.models.deepseek_vl import load_model as load_deepseek_vl_model

        model, processor = load_deepseek_vl_model(
            resolved_name_or_path,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        eos_token = getattr(processor.tokenizer, "eos_token", None) or ""
        text_config = _get_deepseek_text_config(model)
        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        model_config = {
            "family": family,
            "get_lm": _get_deepseek_language_model,
            "is_moe_layer": lambda cfg, idx: (
                idx >= getattr(cfg, "first_k_dense_replace", 0)
                and hasattr(_get_deepseek_decoder_layers(model)[idx].mlp, "experts")
            ),
            "eos_token": eos_token,
            "create_mask": partial(
                create_mask_after_last_token,
                special_token_id=eos_token_id if eos_token_id is not None else -1,
                offset=1,
            ),
        }
    else:
        from src.base.models.internvl import load_model as load_internvl_model

        model, processor = load_internvl_model(
            resolved_name_or_path,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        eos_token = getattr(processor.tokenizer, "eos_token", None) or ""
        text_config = _get_internvl_text_config(model)
        eos_token_id = getattr(text_config, "eos_token_id", None)
        model_config = {
            "family": family,
            "get_lm": _get_internvl_language_model,
            "is_moe_layer": lambda cfg, idx: getattr(cfg, "num_local_experts", 0) > 0,
            "eos_token": eos_token,
            "create_mask": partial(
                create_mask_after_last_token,
                special_token_id=eos_token_id if eos_token_id is not None else -1,
                offset=1,
            ),
        }
    model.eval()
    return ModelBundle(
        family=family,
        model=model,
        processor=processor,
        text_to_message=text_to_message,
        model_config=model_config,
    )


def prepare_inputs(
    bundle: ModelBundle, batch: Dict[str, List[Any]], dataset_name: str
) -> Dict[str, torch.Tensor]:
    processor = bundle.processor
    supports_vision = bool(bundle.model_config.get("supports_vision", True))
    normalized_dataset = normalize_dataset_name(dataset_name)
    if bundle.family == "deepseek_vl":
        packed = []
        for i, text in enumerate(batch["model_input_org_text"]):
            visuals = batch["model_input_visual"][i]
            if not isinstance(visuals, list):
                visuals = [visuals]
            image_prefix = "".join(
                f"This is image_{image_idx + 1}: <image>\n"
                for image_idx in range(len(visuals))
            )
            conversation = [
                {
                    "role": "<|User|>",
                    "content": image_prefix + text,
                    "images": visuals,
                },
                {
                    "role": "<|Assistant|>",
                    "content": batch["model_input_full_answer"][i] + bundle.model_config["eos_token"],
                },
            ]
            prepared = processor(
                conversations=conversation,
                images=visuals,
                force_batchify=False,
                system_prompt="",
            )
            packed.append(prepared)

        batched = processor.batchify(packed)
        if hasattr(batched, "images_seq_mask"):
            batched.images_seq_mask = batched.images_seq_mask.bool()
        return batched

    batched_messages = []
    for i, text in enumerate(batch["model_input_org_text"]):
        if (
            bundle.family == "qwen3"
            and supports_vision
            and normalized_dataset in ("video_mmmu", "m4_instruct", "star")
        ):
            visuals = batch["model_input_visual"][i]
            if not isinstance(visuals, list):
                visuals = [visuals]
            batched_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            *({"type": "image", "image": "path/to/image"} for _ in visuals),
                            {"type": "text", "text": text},
                        ],
                    }
                ]
            )
        else:
            batched_messages.append(bundle.text_to_message(text))
    if supports_vision:
        batched_messages = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        batched_messages = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, tokenize=False
        )
    tmp = []
    for i, _ in enumerate(batched_messages):
        batched_messages[i] = (
            batched_messages[i] + batch["model_input_full_answer"][i]
        )
        batched_messages[i] = batched_messages[i] + bundle.model_config["eos_token"]
        if normalized_dataset in ("video_mmmu", "m4_instruct", "star"):
            tmp.extend(batch["model_input_visual"][i])
            if bundle.family != "qwen3":
                frame_num = batch["model_input_frames"][i]
                media_end_idx = batched_messages[i].find(
                    "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
                )
                batched_messages[i] = (
                    batched_messages[i][:media_end_idx]
                    + "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
                    * (frame_num - 1)
                    + batched_messages[i][media_end_idx:]
                )
    if supports_vision and normalized_dataset in ("video_mmmu", "m4_instruct", "star"):
        batch["model_input_visual"] = tmp
    if supports_vision:
        inputs = processor(
            images=batch["model_input_visual"],
            text=batched_messages,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=False,
        )
        _apply_balanced_multimodal_truncation(bundle, inputs)
    else:
        inputs = processor(
            text=batched_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    return inputs


def _resolve_multimodal_max_length(bundle: ModelBundle) -> Optional[int]:
    processor = bundle.processor
    tokenizer = getattr(processor, "tokenizer", None)
    candidates = [
        getattr(tokenizer, "model_max_length", None),
        getattr(getattr(bundle.model, "config", None), "max_position_embeddings", None),
        getattr(getattr(bundle.model, "config", None), "max_sequence_length", None),
        getattr(getattr(getattr(bundle.model, "config", None), "text_config", None), "max_position_embeddings", None),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < value < 1_000_000:
            return value
    return None


def _resolve_multimodal_media_token_ids(bundle: ModelBundle) -> List[int]:
    config = getattr(bundle.model, "config", None)
    token_ids: List[int] = []
    for attr in ("image_token_id", "video_token_id", "media_placeholder_token_id"):
        value = getattr(config, attr, None)
        if value is not None:
            token_ids.append(int(value))
    return list(dict.fromkeys(token_ids))


def _build_balanced_keep_mask(
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    media_token_ids: List[int],
    max_length: int,
) -> torch.Tensor:
    active_positions = attention_mask_row.to(torch.bool).nonzero(as_tuple=True)[0]
    if active_positions.numel() <= max_length:
        return attention_mask_row.to(torch.bool)

    keep_mask = torch.zeros_like(attention_mask_row, dtype=torch.bool)
    if not media_token_ids:
        keep_mask[active_positions[-max_length:]] = True
        return keep_mask

    media_token_tensor = torch.tensor(
        media_token_ids,
        device=input_ids_row.device,
        dtype=input_ids_row.dtype,
    )
    active_input_ids = input_ids_row.index_select(0, active_positions)
    visual_active_mask = torch.isin(active_input_ids, media_token_tensor)
    visual_positions = active_positions[visual_active_mask]
    text_positions = active_positions[~visual_active_mask]

    visual_budget = max_length // 2
    text_budget = max_length - visual_budget
    keep_visual = min(visual_positions.numel(), visual_budget)
    keep_text = min(text_positions.numel(), text_budget)
    remaining = max_length - keep_visual - keep_text

    if remaining > 0:
        visual_shortfall = visual_positions.numel() - keep_visual
        if visual_shortfall > 0:
            extra = min(remaining, visual_shortfall)
            keep_visual += extra
            remaining -= extra
        text_shortfall = text_positions.numel() - keep_text
        if remaining > 0 and text_shortfall > 0:
            keep_text += min(remaining, text_shortfall)

    if keep_visual > 0:
        keep_mask[visual_positions[:keep_visual]] = True
    if keep_text > 0:
        keep_mask[text_positions[-keep_text:]] = True
    return keep_mask


def _pad_trimmed_sequence(
    tensor: torch.Tensor,
    keep_masks: List[torch.Tensor],
    pad_value: int | float,
) -> torch.Tensor:
    trimmed_rows = []
    max_kept = 0
    for row_idx, keep_mask in enumerate(keep_masks):
        trimmed = tensor[row_idx][keep_mask]
        trimmed_rows.append(trimmed)
        max_kept = max(max_kept, int(trimmed.shape[0]))

    if max_kept == 0:
        target_shape = (tensor.shape[0], 0, *tensor.shape[2:])
        return tensor.new_full(target_shape, pad_value)

    target_shape = (tensor.shape[0], max_kept, *tensor.shape[2:])
    padded = tensor.new_full(target_shape, pad_value)
    for row_idx, trimmed in enumerate(trimmed_rows):
        if trimmed.shape[0] == 0:
            continue
        padded[row_idx, -trimmed.shape[0] :] = trimmed
    return padded


def _apply_balanced_multimodal_truncation(
    bundle: ModelBundle,
    inputs: Dict[str, torch.Tensor],
) -> None:
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    if input_ids is None or attention_mask is None:
        return

    max_length = _resolve_multimodal_max_length(bundle)
    if max_length is None or input_ids.shape[1] <= max_length:
        return

    media_token_ids = _resolve_multimodal_media_token_ids(bundle)
    keep_masks = [
        _build_balanced_keep_mask(
            input_ids_row=input_ids[row_idx],
            attention_mask_row=attention_mask[row_idx],
            media_token_ids=media_token_ids,
            max_length=max_length,
        )
        for row_idx in range(input_ids.shape[0])
    ]

    tokenizer = getattr(bundle.processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    for key, value in list(inputs.items()):
        if not torch.is_tensor(value):
            continue
        if value.ndim < 2:
            continue
        if value.shape[0] != input_ids.shape[0] or value.shape[1] != input_ids.shape[1]:
            continue

        if key == "attention_mask":
            pad_value = 0
        elif key == "input_ids":
            pad_value = pad_token_id
        elif key == "labels":
            pad_value = -100
        else:
            pad_value = 0
        inputs[key] = _pad_trimmed_sequence(value, keep_masks, pad_value)


def move_inputs_to_model_device(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    if hasattr(inputs, "to") and hasattr(inputs, "keys") and not hasattr(inputs, "items"):
        inputs = inputs.to(device=device, dtype=model_dtype)
    moved = {}
    if hasattr(inputs, "items"):
        iterator = inputs.items()
    else:
        iterator = ((key, inputs[key]) for key in inputs.keys())
    for key, value in iterator:
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def filter_model_forward_inputs(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(inputs, "items"):
        items = dict(inputs.items())
    else:
        items = {key: inputs[key] for key in inputs.keys()}
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return items
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in items.items() if key in allowed}


def resolve_activation_fn(obj: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    for attr in ("act_fn", "activation_fn"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn
    return F.silu


def _safe_num_experts(experts: Any) -> int:
    n = getattr(experts, "num_experts", None)
    if n is not None:
        return int(n)
    if hasattr(experts, "__len__"):
        return int(len(experts))
    if hasattr(experts, "gate_up_proj"):
        return int(experts.gate_up_proj.shape[0])
    raise AttributeError(f"Cannot infer number of experts from type: {type(experts)}")


def compute_qwen3_channel_activation(experts, expert_idx: int, hidden_states: torch.Tensor):
    if hasattr(experts, "gate_up_proj"):
        gate_weight = experts.gate_up_proj[expert_idx]
        if gate_weight.shape[-1] == hidden_states.shape[-1]:
            gate_up = F.linear(hidden_states, gate_weight)
        elif gate_weight.shape[0] == hidden_states.shape[-1]:
            gate_up = hidden_states @ gate_weight
        else:
            raise ValueError(
                f"Unsupported Qwen3 fused gate_up_proj shape {tuple(gate_weight.shape)} "
                f"for hidden size {hidden_states.shape[-1]}."
            )
        gate, up = gate_up.chunk(2, dim=-1)
        return experts.act_fn(gate) * up
    if hasattr(experts, "__getitem__"):
        return compute_generic_expert_activation(experts[expert_idx], hidden_states)
    raise NotImplementedError(
        f"Cannot infer Qwen3 expert activation structure for experts type: {type(experts)}"
    )


def _linear_from_module_or_param(module_or_param: Any, hidden_states: torch.Tensor):
    if isinstance(module_or_param, torch.nn.Module):
        return module_or_param(hidden_states)
    return F.linear(hidden_states, module_or_param)


def compute_generic_expert_activation(expert: Any, hidden_states: torch.Tensor):
    if hasattr(expert, "gate_proj") and hasattr(expert, "up_proj"):
        gate = _linear_from_module_or_param(expert.gate_proj, hidden_states)
        up = _linear_from_module_or_param(expert.up_proj, hidden_states)
        return resolve_activation_fn(expert)(gate) * up
    if hasattr(expert, "gate_up_proj"):
        gate_up = _linear_from_module_or_param(expert.gate_up_proj, hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        return resolve_activation_fn(expert)(gate) * up
    if hasattr(expert, "w1") and hasattr(expert, "w3"):
        gate = _linear_from_module_or_param(expert.w1, hidden_states)
        up = _linear_from_module_or_param(expert.w3, hidden_states)
        return resolve_activation_fn(expert)(gate) * up
    raise NotImplementedError(
        f"Cannot infer expert activation structure for expert type: {type(expert)}"
    )


class ObservationAccumulator:
    def __init__(
        self,
        layer_to_num_experts: Dict[int, int],
        layer_to_num_channels: Dict[int, int],
        topk_count_limit: int = 8,
    ):
        self.layer_to_num_experts = layer_to_num_experts
        self.layer_to_num_channels = layer_to_num_channels
        self.topk_count_limit = int(topk_count_limit)
        self.layers = sorted(layer_to_num_experts.keys())
        self.routing_counts = {
            layer: {
                m: torch.zeros(layer_to_num_experts[layer], dtype=torch.float64)
                for m in MODALITIES
            }
            for layer in self.layers
        }
        # 只统计前 min(self.topk_count_limit, 该层 K) 个 top-k 槽位上的 (token, slot) 路由命中
        self.topk_routing_counts = {
            layer: {
                m: torch.zeros(layer_to_num_experts[layer], dtype=torch.float64)
                for m in MODALITIES
            }
            for layer in self.layers
        }
        # 原始 router logits 在 expert 维上按模态对 token 求和（不经过 topk）
        self.router_logits_sum = {
            layer: {
                m: torch.zeros(layer_to_num_experts[layer], dtype=torch.float64)
                for m in MODALITIES
            }
            for layer in self.layers
        }
        self.token_counts = {
            layer: {m: 0.0 for m in MODALITIES} for layer in self.layers
        }
        self.channel_abs_sum = {
            layer: {
                m: torch.zeros(
                    layer_to_num_experts[layer],
                    layer_to_num_channels[layer],
                    dtype=torch.float64,
                )
                for m in MODALITIES
            }
            for layer in self.layers
        }
        self.channel_count = {
            layer: {
                m: torch.zeros(layer_to_num_experts[layer], dtype=torch.float64)
                for m in MODALITIES
            }
            for layer in self.layers
        }

    def record_routing(
        self,
        layer_idx: int,
        router_indices: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> None:
        text_mask = text_mask.bool()
        visual_mask = visual_mask.bool()
        self.token_counts[layer_idx]["text"] += float(text_mask.sum().item())
        self.token_counts[layer_idx]["visual"] += float(visual_mask.sum().item())
        for modality, mask in (("text", text_mask), ("visual", visual_mask)):
            if mask.sum().item() == 0:
                continue
            selected = router_indices[mask].reshape(-1).detach().cpu()
            counts = torch.bincount(
                selected, minlength=self.layer_to_num_experts[layer_idx]
            ).to(torch.float64)
            self.routing_counts[layer_idx][modality] += counts

    def record_topk_routing_and_router_logits(
        self,
        layer_idx: int,
        router_indices: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        """仅将前 `topk_count_limit` 个 top-k 槽位计入 topk 路由；将 router_logits 在 expert 维上按模态对 token 求和。"""
        text_mask = text_mask.bool()
        visual_mask = visual_mask.bool()
        n_tok, k_dim = int(router_indices.shape[0]), int(router_indices.shape[1])
        if int(router_logits.shape[0]) != n_tok:
            raise ValueError(
                f"router_logits 行数 {router_logits.shape[0]} 与 topk 行数 {n_tok} 不一致 (layer {layer_idx})"
            )
        n_exp = int(router_logits.shape[1])
        if n_exp != self.layer_to_num_experts[layer_idx]:
            raise ValueError(
                f"router_logits expert 维 {n_exp} 与该层 n_experts {self.layer_to_num_experts[layer_idx]} 不一致 (layer {layer_idx})"
            )
        k_eff = min(self.topk_count_limit, k_dim)
        if k_eff < 1:
            return
        sliced = router_indices[:, :k_eff]
        log_cpu = router_logits.detach().to(torch.float64)
        for modality, mask in (("text", text_mask), ("visual", visual_mask)):
            if int(mask.sum().item()) == 0:
                continue
            # topk 路由计数（(token,slot) 拉直后 bincount）
            sel = sliced[mask].reshape(-1).detach().cpu()
            counts = torch.bincount(
                sel, minlength=self.layer_to_num_experts[layer_idx]
            ).to(torch.float64)
            self.topk_routing_counts[layer_idx][modality] += counts
            # 原始 logits：每个 expert 一维，对该模态上所有 token 在 expert 维求和
            self.router_logits_sum[layer_idx][modality] += log_cpu[mask].sum(0).cpu()

    def record_channel_response(
        self,
        layer_idx: int,
        expert_idx: int,
        activations: torch.Tensor,
        text_assignment_mask: torch.Tensor,
        visual_assignment_mask: torch.Tensor,
    ) -> None:
        activations = activations.detach().abs().to(torch.float64).cpu()
        for modality, mask in (
            ("text", text_assignment_mask.bool().cpu()),
            ("visual", visual_assignment_mask.bool().cpu()),
        ):
            if mask.sum().item() == 0:
                continue
            self.channel_abs_sum[layer_idx][modality][expert_idx] += activations[
                mask
            ].sum(dim=0)
            self.channel_count[layer_idx][modality][expert_idx] += float(mask.sum().item())

    def to_payload(self) -> Dict[str, Any]:
        return {
            "layers": self.layers,
            "layer_to_num_experts": self.layer_to_num_experts,
            "layer_to_num_channels": self.layer_to_num_channels,
            "routing_counts": self.routing_counts,
            "topk_routing_counts": self.topk_routing_counts,
            "router_logits_sum": self.router_logits_sum,
            "router_topk": self.topk_count_limit,
            "token_counts": self.token_counts,
            "channel_abs_sum": self.channel_abs_sum,
            "channel_count": self.channel_count,
        }


def _build_masks_for_active_tokens(
    text_mask: Optional[torch.Tensor],
    visual_mask: Optional[torch.Tensor],
    padding_mask: Optional[torch.Tensor],
    token_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if text_mask is None:
        text_mask = torch.zeros(token_count, dtype=torch.bool, device=device)
    else:
        text_mask = text_mask.to(device).view(-1)
    if visual_mask is None:
        visual_mask = torch.zeros(token_count, dtype=torch.bool, device=device)
    else:
        visual_mask = visual_mask.to(device).view(-1)
    if padding_mask is not None:
        keep = ~padding_mask.to(device).view(-1)
        text_mask = text_mask[keep]
        visual_mask = visual_mask[keep]
    return text_mask, visual_mask


def _resolve_special_token_tensor_for_model(model) -> Optional[torch.Tensor]:
    for candidate in (
        getattr(model, "special_token_id_tensor", None),
        getattr(getattr(model, "model", None), "special_token_id_tensor", None),
        getattr(getattr(model, "language", None), "special_token_id_tensor", None),
    ):
        if candidate is not None:
            return candidate
    return None


def populate_deepseek_observer_masks(bundle: ModelBundle, inputs: Dict[str, Any]) -> None:
    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        raise KeyError("DeepSeek-VL observation requires `input_ids` in model inputs.")
    flat_input_ids = input_ids.view(-1)
    special_ids = _resolve_special_token_tensor_for_model(bundle.model)
    if special_ids is not None:
        special_ids = special_ids.to(flat_input_ids.device)
        text_mask = ~torch.isin(flat_input_ids, special_ids)
    else:
        text_mask = torch.ones_like(flat_input_ids, dtype=torch.bool)

    image_seq_mask = inputs.get("images_seq_mask", None)
    if image_seq_mask is not None:
        visual_mask = image_seq_mask.to(flat_input_ids.device).view(-1).bool()
    else:
        visual_mask = torch.zeros_like(flat_input_ids, dtype=torch.bool)
    text_mask = text_mask & (~visual_mask)

    attention_mask = inputs.get("attention_mask", None)
    padding_mask = None
    if attention_mask is not None:
        padding_mask = (~attention_mask.to(flat_input_ids.device).bool()).view(-1, 1)

    for layer in _get_deepseek_decoder_layers(bundle.model):
        if not hasattr(layer, "mlp"):
            continue
        if not hasattr(layer.mlp, "experts"):
            continue
        layer.mlp.moe_text_mask = text_mask[:, None]
        layer.mlp.moe_media_mask = visual_mask[:, None]
        layer.mlp.moe_padding_mask = padding_mask


def attach_deepseek_observer(bundle: ModelBundle, accumulator: ObservationAccumulator) -> None:
    model = bundle.model
    config = _get_deepseek_text_config(model)
    for layer_idx, layer in enumerate(_get_deepseek_decoder_layers(model)):
        if not (
            getattr(config, "n_routed_experts", None) is not None
            and layer_idx >= getattr(config, "first_k_dense_replace", 0)
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
            and hasattr(layer.mlp, "experts")
            and hasattr(layer.mlp, "gate")
            and hasattr(layer.mlp, "moe_infer")
        ):
            continue
        if layer_idx not in accumulator.layer_to_num_experts:
            continue

        layer.mlp.layer_idx = layer_idx
        layer.mlp.gate.layer_idx = layer_idx
        original_gate_forward = layer.mlp.gate.forward
        original_moe_infer = layer.mlp.moe_infer

        def observed_gate_forward(
            self,
            hidden_states,
            *args,
            __orig=original_gate_forward,
            __mlp=layer.mlp,
            **kwargs,
        ):
            if hidden_states.dim() != 3:
                raise RuntimeError(
                    f"DeepSeek gate observation expects 3D hidden_states, got {tuple(hidden_states.shape)}"
                )
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            text_mask, visual_mask = _build_masks_for_active_tokens(
                getattr(self, "moe_text_mask", getattr(__mlp, "moe_text_mask", None)),
                getattr(self, "moe_media_mask", getattr(__mlp, "moe_media_mask", None)),
                getattr(self, "moe_padding_mask", getattr(__mlp, "moe_padding_mask", None)),
                flat.shape[0],
                flat.device,
            )
            router_logits = F.linear(
                flat.type(torch.float32), self.weight.type(torch.float32), None
            )
            topk_idx, topk_weight, aux_loss = __orig(hidden_states, *args, **kwargs)
            active_topk_idx = topk_idx
            active_router_logits = router_logits
            padding_mask = getattr(self, "moe_padding_mask", getattr(__mlp, "moe_padding_mask", None))
            if padding_mask is not None:
                keep = ~padding_mask.to(topk_idx.device).view(-1)
                active_topk_idx = topk_idx[keep]
                active_router_logits = router_logits[keep]
            accumulator.record_routing(self.layer_idx, active_topk_idx, text_mask, visual_mask)
            accumulator.record_topk_routing_and_router_logits(
                self.layer_idx, active_topk_idx, text_mask, visual_mask, active_router_logits
            )
            return topk_idx, topk_weight, aux_loss

        def observed_moe_infer(self, x, topk_ids, topk_weight, *args, __orig=original_moe_infer, **kwargs):
            text_mask, visual_mask = _build_masks_for_active_tokens(
                getattr(self, "moe_text_mask", None),
                getattr(self, "moe_media_mask", None),
                getattr(self, "moe_padding_mask", None),
                x.shape[0],
                x.device,
            )
            active_x = x
            active_topk_ids = topk_ids
            if getattr(self, "moe_padding_mask", None) is not None:
                keep = ~self.moe_padding_mask.to(x.device).view(-1)
                active_x = x[keep]
                active_topk_ids = topk_ids[keep]
            num_experts = len(self.experts)
            expert_mask = F.one_hot(
                active_topk_ids.clamp(max=num_experts - 1), num_classes=num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_tensor in expert_hit:
                expert_idx = int(expert_tensor[0].item())
                expert = self.experts[expert_idx]
                if expert is None:
                    continue
                _, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.numel() == 0:
                    continue
                activations = compute_generic_expert_activation(expert, active_x[token_idx])
                accumulator.record_channel_response(
                    self.layer_idx,
                    expert_idx,
                    activations,
                    text_mask[token_idx],
                    visual_mask[token_idx],
                )
            return __orig(x, topk_ids, topk_weight, *args, **kwargs)

        layer.mlp.gate.forward = observed_gate_forward.__get__(layer.mlp.gate)
        layer.mlp.moe_infer = observed_moe_infer.__get__(layer.mlp)


def attach_qwen3_observer(bundle: ModelBundle, accumulator: ObservationAccumulator) -> None:
    model = bundle.model
    config = _get_qwen3_text_config(model)
    layers = _get_qwen3_decoder_layers(model)
    for layer_idx, layer in enumerate(layers):
        # 只在 Qwen3 的稀疏 MoE 层上挂观察逻辑.
        # 条件分别表示:
        # 1. 模型启用了 expert.
        # 2. 当前层命中 decoder_sparse_step 指定的稀疏层周期.
        # 3. 当前层不在只保留 dense MLP 的例外列表里.
        if not (
            getattr(config, "num_experts", 0) > 0
            and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
            and layer_idx not in getattr(config, "mlp_only_layers", [])
        ):
            continue
        if layer_idx not in accumulator.layer_to_num_experts:
            continue
        layer.mlp.experts.layer_idx = layer_idx

        def observation_callback(
            *,
            layer_idx: int,
            router_indices: torch.Tensor,
            router_logits: torch.Tensor,
            active_states: torch.Tensor,
            moe_text_mask: Optional[torch.Tensor],
            moe_media_mask: Optional[torch.Tensor],
            moe_padding_mask: Optional[torch.Tensor],
            experts: Any,
        ) -> None:
            n_tok = int(router_indices.shape[0])
            if moe_text_mask is None and moe_media_mask is None:
                text_mask = torch.ones(n_tok, dtype=torch.bool, device=router_indices.device)
                visual_mask = torch.zeros(n_tok, dtype=torch.bool, device=router_indices.device)
            else:
                text_mask, visual_mask = _build_masks_for_active_tokens(
                    moe_text_mask,
                    moe_media_mask,
                    moe_padding_mask,
                    n_tok,
                    router_indices.device,
                )
                if int(text_mask.numel()) != n_tok or int(visual_mask.numel()) != n_tok:
                    text_mask = torch.ones(n_tok, dtype=torch.bool, device=router_indices.device)
                    visual_mask = torch.zeros(n_tok, dtype=torch.bool, device=router_indices.device)

            accumulator.record_routing(layer_idx, router_indices, text_mask, visual_mask)
            accumulator.record_topk_routing_and_router_logits(
                layer_idx, router_indices, text_mask, visual_mask, router_logits
            )

            num_experts = _safe_num_experts(experts)
            expert_mask = F.one_hot(router_indices, num_classes=num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_tensor in expert_hit:
                expert_idx = int(expert_tensor[0].item())
                _, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.numel() == 0:
                    continue
                current_state = active_states[token_idx]
                activations = compute_qwen3_channel_activation(experts, expert_idx, current_state)
                accumulator.record_channel_response(
                    layer_idx,
                    expert_idx,
                    activations,
                    text_mask[token_idx],
                    visual_mask[token_idx],
                )

        layer.mlp._obs_callback = observation_callback


def attach_kimi_observer(bundle: ModelBundle, accumulator: ObservationAccumulator) -> None:
    model = bundle.model
    config = model.config.text_config
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        # 只给 Kimi 的 MoE 层挂观察逻辑.
        # 条件分别表示:
        # 1. 当前模型确实启用了 routed experts.
        # 2. 当前层已经进入 dense -> MoE 的替换区间.
        # 3. 当前层命中 MoE 层的周期.
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            continue
        if layer_idx not in accumulator.layer_to_num_experts:
            continue
        # Turn on the mask-population path in `models.kimi.model_forward`.
        # 这里写入一个哨兵值, 让上游 forward 路径额外保存文本和多模态 token 的索引或 mask.
        layer.mlp.freq_save_dir = "__observation__"
        # 把层号挂到 gate 上, 后面记录路由结果时需要知道来自哪一层.
        layer.mlp.gate.layer_idx = layer_idx
        original_gate_forward = layer.mlp.gate.forward
        original_moe_infer = layer.mlp.moe_infer

        def observed_gate_forward(self, hidden_states, *args, __orig=original_gate_forward, **kwargs):
            # 与 `gate_forward` 中 gate 权重的线性层一致, 在 topk 前得到 router logits, 供按 expert 维累计.
            hs = hidden_states
            if len(hs.shape) == 3:
                flat = hs.view(-1, hs.shape[-1])
            elif len(hs.shape) == 2:
                flat = hs
            else:
                raise RuntimeError(f"Kimi gate 观察: 不支持的 hidden_states 维数 {hs.shape}")
            router_logits = F.linear(
                flat.type(torch.float32), self.weight.type(torch.float32), None
            )
            # 先执行原始 gate, 拿到每个 token 的 top-k expert 路由结果.
            topk_idx, topk_weight, aux_loss = __orig(hidden_states, *args, **kwargs)
            # 这些索引由上游的 Kimi forward 在打开 freq_save_dir 后提前写入.
            # 如果不存在, 就退化成全 False 的 mask.
            text_index = getattr(self, "moe_text_index", None)
            media_index = getattr(self, "moe_media_index", None)
            if text_index is not None:
                text_mask = torch.zeros(topk_idx.shape[0], dtype=torch.bool, device=topk_idx.device)
                text_mask[text_index.view(-1)] = True
            else:
                text_mask = torch.zeros(topk_idx.shape[0], dtype=torch.bool, device=topk_idx.device)
            if media_index is not None:
                visual_mask = torch.zeros(topk_idx.shape[0], dtype=torch.bool, device=topk_idx.device)
                visual_mask[media_index.view(-1)] = True
            else:
                visual_mask = torch.zeros(topk_idx.shape[0], dtype=torch.bool, device=topk_idx.device)
            # 这里记录的是 token -> expert 的路由选择, 还没有真正跑 expert 计算.
            accumulator.record_routing(self.layer_idx, topk_idx, text_mask, visual_mask)
            accumulator.record_topk_routing_and_router_logits(
                self.layer_idx, topk_idx, text_mask, visual_mask, router_logits
            )
            return topk_idx, topk_weight, aux_loss

        def observed_moe_infer(self, x, topk_ids, topk_weight, *args, __orig=original_moe_infer, **kwargs):
            # moe_infer 阶段已经拿到了 token 的输入表示 x, 以及 gate 给出的 expert 选择.
            # 这里优先复用上游保存好的 text/media mask, 没有就补零.
            text_mask = getattr(self, "moe_text_mask", None)
            visual_mask = getattr(self, "moe_media_mask", None)
            if text_mask is None:
                text_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            else:
                text_mask = text_mask.to(x.device).view(-1)
            if visual_mask is None:
                visual_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            else:
                visual_mask = visual_mask.to(x.device).view(-1)
            num_experts = len(self.experts)
            # one_hot 后的形状大致是 [num_tokens, topk, num_experts].
            # permute 成 [num_experts, topk, num_tokens] 后, 更方便按 expert 遍历.
            expert_mask = F.one_hot(
                topk_ids.clamp(max=num_experts - 1), num_classes=num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            # 先找出本 batch 中至少命中过一次的 expert, 避免遍历所有 expert.
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_tensor in expert_hit:
                expert_idx = int(expert_tensor[0].item())
                # token_idx 是被当前 expert 选中的 token 下标.
                # topk_pos 表示它命中的是第几个 top-k 位置, 这里后续不需要直接使用.
                topk_pos, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.numel() == 0:
                    continue
                # 不走完整的 moe_infer 分发逻辑, 只对命中的 token 单独调用 expert,
                # 提取这个 expert 的中间激活用于观察统计.
                activations = compute_generic_expert_activation(
                    self.experts[expert_idx], x[token_idx]
                )
                accumulator.record_channel_response(
                    self.layer_idx,
                    expert_idx,
                    activations,
                    text_mask[token_idx],
                    visual_mask[token_idx],
                )
            saved_freq_flag = getattr(self, "freq_save_dir", None)
            # 真正执行原始 moe_infer 前临时关闭 freq_save_dir.
            # 否则原始实现可能再次触发额外的保存逻辑, 造成重复副作用.
            self.freq_save_dir = None
            try:
                return __orig(x, topk_ids, topk_weight, *args, **kwargs)
            finally:
                # 无论原始 forward 是否报错, 都恢复现场.
                self.freq_save_dir = saved_freq_flag

        layer.mlp.gate.forward = observed_gate_forward.__get__(layer.mlp.gate)
        layer.mlp.moe_infer = observed_moe_infer.__get__(layer.mlp)


def discover_layer_structure(bundle: ModelBundle) -> Tuple[Dict[int, int], Dict[int, int]]:
    layer_to_num_experts = {}
    layer_to_num_channels = {}
    if bundle.family == "qwen3":
        config = _get_qwen3_text_config(bundle.model)
        layers = _get_qwen3_decoder_layers(bundle.model)
        for layer_idx, layer in enumerate(layers):
            if (
                getattr(config, "num_experts", 0) > 0
                and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
                and layer_idx not in getattr(config, "mlp_only_layers", [])
            ):
                experts = layer.mlp.experts
                num_experts = _safe_num_experts(experts)
                layer_to_num_experts[layer_idx] = int(num_experts)
                intermediate_size = getattr(
                    experts, "intermediate_dim", None
                )
                if intermediate_size is None:
                    intermediate_size = getattr(
                        experts, "intermediate_size", None
                    )
                if intermediate_size is None:
                    sample_expert = experts[0]
                    if hasattr(sample_expert, "up_proj"):
                        intermediate_size = sample_expert.up_proj.out_features
                    elif hasattr(sample_expert, "w3"):
                        w3 = sample_expert.w3
                        intermediate_size = (
                            w3.out_features if isinstance(w3, torch.nn.Module) else w3.shape[0]
                        )
                if intermediate_size is None:
                    raise AttributeError("Cannot infer Qwen3 expert width for observation hooks.")
                layer_to_num_channels[layer_idx] = int(intermediate_size)
        return layer_to_num_experts, layer_to_num_channels
    if bundle.family == "internvl":
        config = _get_internvl_text_config(bundle.model)
        for layer_idx, layer in enumerate(_get_internvl_decoder_layers(bundle.model)):
            if hasattr(layer.mlp, "experts") and getattr(config, "num_experts", 0) > 0:
                experts = layer.mlp.experts
                num_experts = _safe_num_experts(experts)
                layer_to_num_experts[layer_idx] = int(num_experts)
                intermediate_size = getattr(experts, "intermediate_dim", None)
                if intermediate_size is None:
                    intermediate_size = getattr(experts, "intermediate_size", None)
                if intermediate_size is None:
                    intermediate_size = getattr(experts, "expert_dim", None)
                if intermediate_size is None:
                    sample_expert = experts[0]
                    if hasattr(sample_expert, "up_proj"):
                        intermediate_size = sample_expert.up_proj.out_features
                    elif hasattr(sample_expert, "w3"):
                        w3 = sample_expert.w3
                        intermediate_size = (
                            w3.out_features if isinstance(w3, torch.nn.Module) else w3.shape[0]
                        )
                if intermediate_size is None:
                    raise AttributeError(
                        "Cannot infer InternVL/Qwen3-MoE expert width for observation hooks."
                    )
                layer_to_num_channels[layer_idx] = int(intermediate_size)
                continue

            if getattr(config, "num_local_experts", 0) <= 0:
                continue
            if not (hasattr(layer.mlp, "router") and hasattr(layer.mlp, "experts")):
                continue
            experts = layer.mlp.experts
            layer_to_num_experts[layer_idx] = int(getattr(experts, "num_experts"))
            intermediate_size = getattr(experts, "expert_dim", None)
            if intermediate_size is None:
                intermediate_size = getattr(experts, "intermediate_size", None)
            if intermediate_size is None:
                raise AttributeError(
                    "Cannot infer InternVL/GPT-OSS expert width: expected "
                    "`expert_dim` or `intermediate_size` on fused experts."
                )
            layer_to_num_channels[layer_idx] = int(intermediate_size)
        return layer_to_num_experts, layer_to_num_channels
    if bundle.family == "deepseek_vl":
        config = _get_deepseek_text_config(bundle.model)
        for layer_idx, layer in enumerate(_get_deepseek_decoder_layers(bundle.model)):
            if layer_idx < getattr(config, "first_k_dense_replace", 0):
                continue
            if not hasattr(layer.mlp, "experts"):
                continue
            layer_to_num_experts[layer_idx] = len(layer.mlp.experts)
            sample_expert = layer.mlp.experts[0]
            if hasattr(sample_expert, "up_proj"):
                layer_to_num_channels[layer_idx] = int(sample_expert.up_proj.out_features)
            elif hasattr(sample_expert, "gate_up_proj"):
                gate_up = sample_expert.gate_up_proj
                layer_to_num_channels[layer_idx] = int(gate_up.shape[0] // 2)
            else:
                raise NotImplementedError(
                    f"Cannot infer DeepSeek-VL expert width from type {type(sample_expert)}"
                )
        return layer_to_num_experts, layer_to_num_channels
    config = bundle.model.config.text_config
    for layer_idx, layer in enumerate(bundle.model.language_model.model.layers):
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            layer_to_num_experts[layer_idx] = len(layer.mlp.experts)
            sample_expert = layer.mlp.experts[0]
            if hasattr(sample_expert, "up_proj"):
                up_proj = sample_expert.up_proj
                layer_to_num_channels[layer_idx] = up_proj.out_features
            elif hasattr(sample_expert, "gate_up_proj"):
                gate_up = sample_expert.gate_up_proj
                layer_to_num_channels[layer_idx] = gate_up.shape[0] // 2
            elif hasattr(sample_expert, "w3"):
                w3 = sample_expert.w3
                layer_to_num_channels[layer_idx] = (
                    w3.out_features if isinstance(w3, torch.nn.Module) else w3.shape[0]
                )
            else:
                raise NotImplementedError(
                    f"Cannot infer Kimi expert width from type {type(sample_expert)}"
                )
    return layer_to_num_experts, layer_to_num_channels


def collect_observation_stats(args) -> Dict[str, Any]:
    normalized_dataset = normalize_dataset_name(args.dataset)
    print(
        f"[Observation] Starting collection for dataset={args.dataset}, requested_samples={args.num_samples}",
        flush=True,
    )
    ensure_writable_datasets_cache()
    resolved_name_or_path = resolve_model_name_or_path(args.model_name_or_path)
    print(f"[Observation] Resolved model path: {resolved_name_or_path}", flush=True)
    print("[Observation] Loading model and processor...", flush=True)
    bundle = load_model_bundle(args.model_name_or_path)
    print(f"[Observation] Loaded model family: {bundle.family}", flush=True)
    layer_to_num_experts, layer_to_num_channels = discover_layer_structure(bundle)
    observe_layers_raw = (getattr(args, "observe_layers", "") or "").strip()
    if observe_layers_raw:
        wanted = set()
        for item in observe_layers_raw.split(","):
            item = item.strip()
            if not item:
                continue
            wanted.add(int(item))
        layer_to_num_experts = {
            layer_idx: n_exp
            for layer_idx, n_exp in layer_to_num_experts.items()
            if layer_idx in wanted
        }
        layer_to_num_channels = {
            layer_idx: n_ch
            for layer_idx, n_ch in layer_to_num_channels.items()
            if layer_idx in wanted
        }
        if not layer_to_num_experts:
            raise ValueError(
                f"--observe_layers={observe_layers_raw!r} 过滤后没有可观测 MoE 层。"
            )
        print(
            f"[Observation] Layer filter active: keep layers={sorted(layer_to_num_experts.keys())}",
            flush=True,
        )
    print(
        f"[Observation] Discovered {len(layer_to_num_experts)} MoE layers "
        f"covering {sum(layer_to_num_experts.values())} experts.",
        flush=True,
    )
    router_topk = int(getattr(args, "router_topk", 8))
    accumulator = ObservationAccumulator(
        layer_to_num_experts, layer_to_num_channels, topk_count_limit=router_topk
    )
    if bundle.family == "qwen3":
        attach_qwen3_observer(bundle, accumulator)
    elif bundle.family == "deepseek_vl":
        attach_deepseek_observer(bundle, accumulator)
    else:
        attach_kimi_observer(bundle, accumulator)
    print("[Observation] Observation hooks attached.", flush=True)

    print(f"[Observation] Building dataset: {args.dataset}", flush=True)
    if normalized_dataset == "m4_instruct":
        required_rows = max(0, args.start_idx) + max(0, args.num_samples)
        data = build_dataset(args.dataset, bundle.family, max_rows=required_rows)
        if args.subset_seed is not None:
            print(
                "[Observation] m4_instruct 使用流式按需加载，忽略 --subset_seed，改为顺序区间采样。",
                flush=True,
            )
        subset_end = min(args.start_idx + args.num_samples, len(data))
        subset_indices = list(range(args.start_idx, subset_end))
        print(
            f"[Observation] Dataset size={len(data)} (stream loaded), processing range=[{args.start_idx}, {subset_end}), "
            f"actual_samples={len(subset_indices)}, batch_size={args.batch_size}",
            flush=True,
        )
    else:
        data = build_dataset(args.dataset, bundle.family)
        if args.subset_seed is not None:
            pool = list(range(args.start_idx, len(data)))
            n_pick = min(args.num_samples, len(pool))
            rng = random.Random(args.subset_seed)
            subset_indices = rng.sample(pool, n_pick) if n_pick > 0 else []
            print(
                f"[Observation] Dataset size={len(data)}, random subset: seed={args.subset_seed}, "
                f"pool=[{args.start_idx}, {len(data)}), picked={len(subset_indices)}, batch_size={args.batch_size}",
                flush=True,
            )
        else:
            subset_end = min(args.start_idx + args.num_samples, len(data))
            subset_indices = list(range(args.start_idx, subset_end))
            print(
                f"[Observation] Dataset size={len(data)}, processing range=[{args.start_idx}, {subset_end}), "
                f"actual_samples={len(subset_indices)}, batch_size={args.batch_size}",
                flush=True,
            )
    print_probe_snapshot(accumulator, stage="开始前（尚未处理任何 batch）")
    subset = Subset(data, subset_indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    total_batches = len(dataloader)
    print(f"[Observation] Starting forward passes over {total_batches} batches...", flush=True)
    with torch.no_grad():
        progress = tqdm(
            dataloader,
            total=total_batches,
            desc="Collecting observation stats",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(progress, start=1):
            inputs = prepare_inputs(bundle, batch, args.dataset)
            inputs = move_inputs_to_model_device(bundle.model, inputs)
            if bundle.family == "deepseek_vl":
                populate_deepseek_observer_masks(bundle, inputs)
            bundle.model(**filter_model_forward_inputs(bundle.model, inputs), use_cache=False, return_dict=True)
            progress.set_postfix_str(
                f"samples={min(batch_idx * args.batch_size, len(subset_indices))}/{len(subset_indices)}"
            )
            if batch_idx == 1:
                print_probe_snapshot(accumulator, stage="第 1 个 batch 后")
    print("[Observation] Forward pass collection complete.", flush=True)
    print_probe_snapshot(accumulator, stage="全部 batch 结束后")

    payload = accumulator.to_payload()
    payload["model_name_or_path"] = args.model_name_or_path
    payload["resolved_model_name_or_path"] = resolved_name_or_path
    payload["dataset"] = args.dataset
    payload["num_samples"] = args.num_samples
    payload["start_idx"] = args.start_idx
    payload["subset_seed"] = args.subset_seed
    payload["batch_size"] = args.batch_size
    payload["model_family"] = bundle.family
    print("[Observation] Aggregated observation statistics are ready.", flush=True)
    return payload


def compute_routing_freq(
    raw_stats: Dict[str, Any], counts_key: str = "routing_counts"
) -> Dict[int, Dict[str, torch.Tensor]]:
    if counts_key not in raw_stats:
        counts_key = "routing_counts"
    output: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in raw_stats["layers"]:
        output[layer] = {}
        for modality in MODALITIES:
            denom = raw_stats["token_counts"][layer][modality]
            counts = raw_stats[counts_key][layer][modality].clone()
            if denom <= 0:
                output[layer][modality] = torch.zeros_like(counts)
            else:
                output[layer][modality] = counts / float(denom)
    return output


def compute_ema(routing_freq: Dict[int, Dict[str, torch.Tensor]], eps: float = 1e-8):
    ema = {}
    for layer, per_modality in routing_freq.items():
        text = per_modality["text"]
        visual = per_modality["visual"]
        ema[layer] = (visual - text) / (visual + text + eps)
    return ema


def compute_channel_response(raw_stats: Dict[str, Any]) -> Dict[int, Dict[str, torch.Tensor]]:
    output = {}
    for layer in raw_stats["layers"]:
        output[layer] = {}
        for modality in MODALITIES:
            count = raw_stats["channel_count"][layer][modality].clone().unsqueeze(-1)
            sums = raw_stats["channel_abs_sum"][layer][modality].clone()
            output[layer][modality] = sums / (count + 1e-8)
    return output


def compute_modal_bias(channel_response: Dict[int, Dict[str, torch.Tensor]], eps: float = 1e-8):
    output = {}
    for layer, per_modality in channel_response.items():
        text = per_modality["text"]
        visual = per_modality["visual"]
        output[layer] = (visual - text).abs() / (visual + text + eps)
    return output


def compute_conflict_scores(
    raw_stats: Dict[str, Any],
    routing_freq: Dict[int, Dict[str, torch.Tensor]],
    modal_bias: Dict[int, torch.Tensor],
    channel_response: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-8,
):
    conflict = {}
    mean_modal_bias = {}
    for layer in raw_stats["layers"]:
        conflict[layer] = {}
        mean_modal_bias[layer] = {}
        for modality in MODALITIES:
            freq = routing_freq[layer][modality]
            if freq.max().item() > 0:
                norm_freq = freq / (freq.max() + eps)
            else:
                norm_freq = torch.zeros_like(freq)
            dominance = channel_response[layer][modality] / (
                channel_response[layer]["text"] + channel_response[layer]["visual"] + eps
            )
            modality_specific_bias = (modal_bias[layer] * dominance).mean(dim=-1)
            mean_modal_bias[layer][modality] = modality_specific_bias
            conflict[layer][modality] = (1.0 - norm_freq) * modality_specific_bias
    return conflict, mean_modal_bias


def stack_layer_tensors(mapping: Dict[int, torch.Tensor], layers: List[int]) -> torch.Tensor:
    return torch.stack([mapping[layer] for layer in layers], dim=0)


def pick_plot_layer(raw_stats: Dict[str, Any]) -> int:
    best_layer = raw_stats["layers"][0]
    best_score = -1.0
    for layer in raw_stats["layers"]:
        score = raw_stats["token_counts"][layer]["text"] + raw_stats["token_counts"][layer]["visual"]
        if score > best_score:
            best_layer = layer
            best_score = score
    return best_layer


def plot_heatmap(matrix: torch.Tensor, title: str, output_path: str, cmap: str = "coolwarm"):
    plt.figure(figsize=(14, 6))
    plt.imshow(matrix.cpu().numpy(), aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Expert ID")
    plt.ylabel("Layer Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"[结果输出] 已保存 {title} 热力图: {output_path}", flush=True)
    plt.close()


def plot_expert_channel_bars(
    text_resp: torch.Tensor,
    visual_resp: torch.Tensor,
    layer_idx: int,
    expert_indices: List[int],
    output_path: str,
):
    rows = len(expert_indices)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 3 * rows), squeeze=False)
    for axis, expert_idx in zip(axes[:, 0], expert_indices):
        axis.plot(text_resp[expert_idx].cpu().numpy(), label="text", alpha=0.9)
        axis.plot(visual_resp[expert_idx].cpu().numpy(), label="visual", alpha=0.9)
        axis.set_title(f"Layer {layer_idx} Expert {expert_idx}")
        axis.set_xlabel("Channel Index")
        axis.set_ylabel("Mean |activation|")
        axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"[结果输出] 已保存第 {layer_idx} 层通道响应折线图: {output_path}", flush=True)
    plt.close()


def pick_top_experts_global(
    modal_bias: Dict[int, torch.Tensor],
    raw_stats: Dict[str, Any],
    top_k: int,
    min_count: int = 5,
    ema: Optional[Dict[int, torch.Tensor]] = None,
    sort_by: str = "modal_bias",
) -> List[Tuple[int, int, float]]:
    """全模型遍历，返回 top_k 个 (layer, expert, sort_score) 列表。

    sort_by:
      "modal_bias" — 按通道级平均 ModalBias 降序（通道差异最大）
      "ema_abs"    — 按 |EMA| 降序（模态偏好最强，需传入 ema）
    min_count 过滤掉任一模态 token 数 < min_count 的 expert。
    """
    candidates = []
    for layer, bias in modal_bias.items():
        text_counts = raw_stats["channel_count"][layer]["text"]
        visual_counts = raw_stats["channel_count"][layer]["visual"]
        for expert_idx in range(bias.shape[0]):
            if (
                text_counts[expert_idx].item() < min_count
                or visual_counts[expert_idx].item() < min_count
            ):
                continue
            if sort_by == "ema_abs" and ema is not None and layer in ema:
                score = float(ema[layer][expert_idx].abs().item())
            else:
                score = float(bias[expert_idx].mean().item())
            candidates.append((layer, expert_idx, score))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_k]


def plot_expert_channel_lines_per_expert(
    channel_response: Dict[int, Dict[str, torch.Tensor]],
    candidates: List[Tuple[int, int, float]],
    output_dir: str,
    ema: Optional[Dict[int, torch.Tensor]] = None,
) -> List[str]:
    """每个 (layer, expert) 单独出一张折线图，返回已保存的路径列表。

    title 优先显示 EMA（expert modality affinity，+1 纯视觉 / -1 纯文本 / 0 中性）；
    若未传入 ema 则回退到显示 mean ModalBias。
    """
    saved = []
    for layer_idx, expert_idx, bias_val in candidates:
        t = channel_response[layer_idx]["text"][expert_idx].cpu().numpy()
        v = channel_response[layer_idx]["visual"][expert_idx].cpu().numpy()
        fig, axis = plt.subplots(1, 1, figsize=(14, 3))
        axis.plot(t, label="text", alpha=0.9, color="steelblue")
        axis.plot(v, label="visual", alpha=0.9, color="darkorange")
        if ema is not None and layer_idx in ema:
            ema_val = float(ema[layer_idx][expert_idx].item())
            title_suffix = f"EMA = {ema_val:+.3f}  (+1 visual / -1 text)"
        else:
            title_suffix = f"mean ModalBias = {bias_val:.3f}"
        axis.set_title(f"Layer {layer_idx} Expert {expert_idx}  ({title_suffix})")
        axis.set_xlabel("Channel Index")
        axis.set_ylabel("Mean |activation|")
        axis.legend()
        plt.tight_layout()
        fname = os.path.join(output_dir, f"channel_resp_L{layer_idx}_E{expert_idx}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(
            f"[结果输出] 已保存 Layer {layer_idx} Expert {expert_idx} 通道响应折线图: {fname}",
            flush=True,
        )
        saved.append(fname)
    return saved


def build_base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="gqa",
        choices=["gqa", "coco", "video_mmmu", "m4_instruct", "m4-instruct", "m4"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="若设置，则在 [start_idx, len(dataset)) 内无放回随机抽 num_samples 条；"
        "子集顺序即遍历顺序，DataLoader 无需 shuffle。不设则沿用连续区间。",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--raw_stats_path", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--router-topk",
        type=int,
        default=8,
        help="O1/路由：top-k 路由直方图只计前 k 个槽位；超过模型 top_k 时以模型 top_k 为准。",
    )
    parser.add_argument(
        "--observe_layers",
        type=str,
        default="",
        help="仅观测指定 MoE 层，逗号分隔，如 '44,45,46,47'。为空时观测全部可观测层。",
    )
    return parser
