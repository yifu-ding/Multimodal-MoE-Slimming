import os
from functools import partial

from datasets import concatenate_datasets, load_dataset
from loguru import logger
from transformers import AutoConfig

from tasks.coco import coco_transform
from tasks.dataset_paths import require_dataset_dir
from tasks.gqa import gqa_transform
from tasks.video_mmmu import videommmu_transform


def resolve_model_family_from_path(model_name_or_path: str) -> str:
    name = os.path.basename(model_name_or_path).lower()
    if "kimi-vl" in name:
        return "kimi"
    if "qwen3-vl" in name:
        return "qwen3"
    raise ValueError(f"Unsupported model for MoDES integration: {model_name_or_path}")


def build_default_layer_gate_dict(
    pretrained: str,
    *,
    trust_remote_code: bool = True,
    topk=None,
    topk_text=None,
    topk_visual=None,
    expert_num=None,
    text_expert_num=None,
    visual_expert_num=None,
    start: int = 0,
    end: int = -1,
):
    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
    family = resolve_model_family_from_path(pretrained)
    layer_gate_dict = {}

    if family == "kimi":
        text_cfg = config.text_config
        for i in range(text_cfg.num_hidden_layers):
            if (
                i >= text_cfg.first_k_dense_replace
                and i % text_cfg.moe_layer_freq == 0
                and text_cfg.n_routed_experts is not None
            ):
                layer_gate_dict[i] = {
                    "text": text_cfg.num_experts_per_tok,
                    "visual": text_cfg.num_experts_per_tok,
                    "text_expert_num": text_cfg.n_routed_experts,
                    "visual_expert_num": text_cfg.n_routed_experts,
                }
        if topk is not None:
            topk_text = text_cfg.num_experts_per_tok if topk_text is None else topk_text
            topk_visual = text_cfg.num_experts_per_tok if topk_visual is None else topk_visual
            for i in range(start, end + 1):
                if i in layer_gate_dict:
                    layer_gate_dict[i]["text"] = topk_text
                    layer_gate_dict[i]["visual"] = topk_visual
        if expert_num is not None:
            text_expert_num = (
                text_cfg.n_routed_experts if text_expert_num is None else text_expert_num
            )
            visual_expert_num = (
                text_cfg.n_routed_experts
                if visual_expert_num is None
                else visual_expert_num
            )
            for i in range(start, end + 1):
                if i in layer_gate_dict:
                    layer_gate_dict[i]["text_expert_num"] = text_expert_num
                    layer_gate_dict[i]["visual_expert_num"] = visual_expert_num
        return layer_gate_dict, config

    text_cfg = config.text_config
    for i in range(text_cfg.num_hidden_layers):
        if (
            i not in text_cfg.mlp_only_layers
            and text_cfg.num_experts > 0
            and (i + 1) % text_cfg.decoder_sparse_step == 0
        ):
            layer_gate_dict[i] = {
                "text": text_cfg.num_experts_per_tok,
                "visual": text_cfg.num_experts_per_tok,
            }
    if topk is not None:
        topk_text = text_cfg.num_experts_per_tok if topk_text is None else topk_text
        topk_visual = text_cfg.num_experts_per_tok if topk_visual is None else topk_visual
        for i in range(start, end + 1):
            if i in layer_gate_dict:
                layer_gate_dict[i]["text"] = topk_text
                layer_gate_dict[i]["visual"] = topk_visual
    return layer_gate_dict, config


def apply_optional_structural_pruning(model, kwargs, log_prefix: str) -> None:
    scores_path = kwargs.get("scores_path", None)
    prune_ratio = float(kwargs.get("prune_ratio", 0) or 0)
    if not scores_path or prune_ratio <= 0:
        return

    from src.generate_mask import generate_masks
    from src.prune import apply_structural_pruning

    inter_method = kwargs.get("inter_method", "uniform")
    intra_method = kwargs.get("intra_method", "uniform")
    intra_expert_metric = kwargs.get("intra_expert_metric", "activation")
    modality_aware = bool(int(kwargs.get("modality_aware", 0)))
    shared_protect = bool(int(kwargs.get("shared_protect", 0)))
    text_only = bool(int(kwargs.get("text_only", 0)))
    visual_only = bool(int(kwargs.get("visual_only", 0)))
    normalize = bool(int(kwargs.get("normalize", 0)))
    expertwise_budget_normalize = bool(int(kwargs.get("expertwise_budget_normalize", 0)))
    smooth_fn = kwargs.get("smooth_fn", "sqrt")
    ema_source_key = kwargs.get("ema_source_key", "ema_matrix")
    layerwise_loss_key = kwargs.get("layerwise_loss_key", "layerwise_second_order_sum")
    align_inter = int(kwargs.get("align_inter", 0))
    min_per_expert = int(kwargs.get("min_per_expert", 0))
    thresholds_path = kwargs.get("thresholds_path", None)

    logger.info(
        f"[{log_prefix}] Generating masks: ratio={prune_ratio}, "
        f"inter={inter_method}, intra={intra_method}, metric={intra_expert_metric}"
    )
    mask_result = generate_masks(
        scores_dir=scores_path,
        prune_kwargs={
            "prune_ratio": prune_ratio,
            "thresholds_path": thresholds_path,
            "mask_method_kwargs": {
                "inter_layer_method": inter_method,
                "intra_layer_method": intra_method,
                "intra_expert_metric": intra_expert_metric,
                "layerwise_loss_key": layerwise_loss_key,
            },
            "adjust_masks_kwargs": {
                "align_inter": align_inter,
                "min_per_expert": min_per_expert,
            },
            "modality_aware": modality_aware,
            "shared_protect": shared_protect,
            "text_only": text_only,
            "visual_only": visual_only,
            "normalize": normalize,
            "expertwise_budget_normalize": expertwise_budget_normalize,
            "ema_source_key": ema_source_key,
            "prune_hidden": False,
            "prune_gqa": False,
            "smooth_fn": smooth_fn,
        },
        device="cpu",
        verbose=True,
    )
    mask_tensor = mask_result["intermediate_masks"]
    layers = [
        int(layer)
        for layer in mask_result.get("layers", list(range(mask_tensor.shape[0])))
    ]
    masks = {
        layer_idx: mask_tensor[pos].detach().cpu().bool()
        for pos, layer_idx in enumerate(layers)
    }
    apply_structural_pruning(model, masks, model.config.text_config)
    logger.info(f"[{log_prefix}] Structural pruning applied: ratio={prune_ratio}")


def build_text_to_message(text: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def get_family_calibration_config(model_name_or_path: str):
    family = resolve_model_family_from_path(model_name_or_path)
    if family == "kimi":
        return {
            "family": family,
            "model_name": "Kimi-VL",
            "modalities": ["text", "visual"],
            "get_lm": lambda m: m.language_model,
            "is_moe_layer": lambda cfg, idx: idx >= cfg.first_k_dense_replace
            and idx % cfg.moe_layer_freq == 0,
            "create_mask": partial(build_mask_after_token, special_token_id=163588, offset=3),
            "eos_token": "<|im_end|>[EOS]",
            "special_token_id": 163588,
            "answer_offset": 3,
        }
    return {
        "family": family,
        "model_name": "Qwen3-VL",
        "modalities": ["text", "visual"],
        "get_lm": lambda m: m.model.language_model,
        "is_moe_layer": lambda cfg, idx: (
            hasattr(cfg, "num_experts")
            and cfg.num_experts > 0
            and (idx + 1) % cfg.decoder_sparse_step == 0
            and idx not in getattr(cfg, "mlp_only_layers", [])
        ),
        "create_mask": partial(
            build_mask_after_last_token, special_token_id=151644, offset=3
        ),
        "eos_token": "<|im_end|>",
        "special_token_id": 151644,
        "answer_offset": 3,
    }


def load_modes_dataset(dataset: str, family: str):
    if dataset == "gqa":
        data = load_dataset(
            require_dataset_dir("GQA", "testdev_balanced_instructions"), token=True
        )["train"]
        data.set_transform(gqa_transform)
        return data
    if dataset == "coco":
        data = load_dataset(
            require_dataset_dir("COCO-Caption2017", "data"), token=True
        )["validation"]
        data.set_transform(coco_transform)
        return data
    if dataset == "video_mmmu":
        if family != "kimi":
            raise AssertionError("VideoMMMU only supports Kimi-VL")
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
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_mask_after_token(*args, **kwargs):
    from src.base.masking import create_mask_after_token

    return create_mask_after_token(*args, **kwargs)


def build_mask_after_last_token(*args, **kwargs):
    from src.base.masking import create_mask_after_last_token

    return create_mask_after_last_token(*args, **kwargs)
