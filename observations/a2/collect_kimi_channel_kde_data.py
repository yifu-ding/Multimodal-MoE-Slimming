import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from observations.common import (
    MODALITIES,
    build_base_arg_parser,
    build_dataset,
    compute_generic_expert_activation,
    custom_collate_fn,
    discover_layer_structure,
    ensure_dir,
    filter_model_forward_inputs,
    infer_model_family,
    load_model_bundle,
    move_inputs_to_model_device,
    normalize_dataset_name,
    prepare_inputs,
    print_saved_artifact_message,
)


DEFAULT_LAYERS = (5, 15, 25)


def resolve_device_map(device_map_mode: str):
    if device_map_mode == "auto":
        return "auto"
    if device_map_mode == "cpu":
        return "cpu"
    if device_map_mode == "single_gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device_map single_gpu requires CUDA, but torch.cuda.is_available() is False."
            )
        return {"": torch.cuda.current_device()}
    raise ValueError(f"Unsupported --device_map mode: {device_map_mode}")


class SampleLevelChannelAccumulator:
    def __init__(
        self,
        layer_to_num_experts: Dict[int, int],
        layer_to_num_channels: Dict[int, int],
        selected_layers: List[int],
    ):
        self.selected_layers = sorted(selected_layers)
        self.layer_to_num_experts = {
            layer: int(layer_to_num_experts[layer]) for layer in self.selected_layers
        }
        self.layer_to_num_channels = {
            layer: int(layer_to_num_channels[layer]) for layer in self.selected_layers
        }
        self.channel_abs_samples = {
            layer: {modality: [] for modality in MODALITIES}
            for layer in self.selected_layers
        }
        self.channel_num_tokens = {
            layer: {modality: [] for modality in MODALITIES}
            for layer in self.selected_layers
        }
        self._current_values: Dict[int, Dict[str, torch.Tensor]] = {}
        self._current_counts: Dict[int, Dict[str, torch.Tensor]] = {}

    def start_sample(self) -> None:
        self._current_values = {}
        self._current_counts = {}
        for layer in self.selected_layers:
            self._current_values[layer] = {}
            self._current_counts[layer] = {}
            for modality in MODALITIES:
                self._current_values[layer][modality] = torch.full(
                    (
                        self.layer_to_num_experts[layer],
                        self.layer_to_num_channels[layer],
                    ),
                    float("nan"),
                    dtype=torch.float32,
                )
                self._current_counts[layer][modality] = torch.zeros(
                    self.layer_to_num_experts[layer], dtype=torch.int32
                )

    def record_channel_response(
        self,
        layer_idx: int,
        expert_idx: int,
        activations: torch.Tensor,
        text_assignment_mask: torch.Tensor,
        visual_assignment_mask: torch.Tensor,
    ) -> None:
        if layer_idx not in self._current_values:
            return
        activations = activations.detach().abs().to(torch.float32).cpu()
        for modality, mask in (
            ("text", text_assignment_mask.bool().cpu()),
            ("visual", visual_assignment_mask.bool().cpu()),
        ):
            token_count = int(mask.sum().item())
            if token_count == 0:
                continue
            self._current_values[layer_idx][modality][expert_idx] = activations[mask].mean(dim=0)
            self._current_counts[layer_idx][modality][expert_idx] = token_count

    def finish_sample(self) -> None:
        for layer in self.selected_layers:
            for modality in MODALITIES:
                self.channel_abs_samples[layer][modality].append(
                    self._current_values[layer][modality].clone()
                )
                self.channel_num_tokens[layer][modality].append(
                    self._current_counts[layer][modality].clone()
                )

    def to_payload(self, model_name_or_path: str, dataset: str, num_samples: int) -> Dict[str, Any]:
        return {
            "model_name_or_path": model_name_or_path,
            "dataset": dataset,
            "num_samples": num_samples,
            "layers": self.selected_layers,
            "layer_to_num_experts": self.layer_to_num_experts,
            "layer_to_num_channels": self.layer_to_num_channels,
            "channel_abs_samples": {
                layer: {
                    modality: torch.stack(self.channel_abs_samples[layer][modality], dim=0)
                    if self.channel_abs_samples[layer][modality]
                    else torch.empty(
                        0,
                        self.layer_to_num_experts[layer],
                        self.layer_to_num_channels[layer],
                        dtype=torch.float32,
                    )
                    for modality in MODALITIES
                }
                for layer in self.selected_layers
            },
            "channel_num_tokens": {
                layer: {
                    modality: torch.stack(self.channel_num_tokens[layer][modality], dim=0)
                    if self.channel_num_tokens[layer][modality]
                    else torch.empty(
                        0,
                        self.layer_to_num_experts[layer],
                        dtype=torch.int32,
                    )
                    for modality in MODALITIES
                }
                for layer in self.selected_layers
            },
        }


def attach_kimi_sample_observer(bundle, accumulator: SampleLevelChannelAccumulator) -> None:
    model = bundle.model
    config = model.config.text_config
    mask_bootstrap_layer = min(
        max(int(getattr(config, "first_k_dense_replace", 0)), 0),
        len(model.language_model.model.layers) - 1,
    )
    target_layers = set(accumulator.selected_layers)
    target_layers.add(mask_bootstrap_layer)

    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if layer_idx not in target_layers:
            continue
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            raise ValueError(f"Layer {layer_idx} is not a Kimi MoE layer.")
        layer.mlp.freq_save_dir = "__observation__"
        original_gate_forward = layer.mlp.gate.forward
        original_moe_infer = layer.mlp.moe_infer

        def _make_gate_forward(orig_gate_forward):
            def observed_gate_forward(self, hidden_states, *args, **kwargs):
                return orig_gate_forward(hidden_states, *args, **kwargs)

            return observed_gate_forward

        def _make_moe_infer(bound_layer_idx: int, orig_moe_infer):
            def observed_moe_infer(self, x, topk_ids, topk_weight, *args, **kwargs):
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
                expert_mask = F.one_hot(
                    topk_ids.clamp(max=num_experts - 1), num_classes=num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                for expert_tensor in expert_hit:
                    expert_idx = int(expert_tensor[0].item())
                    _, token_idx = torch.where(expert_mask[expert_idx])
                    if token_idx.numel() == 0:
                        continue
                    if bound_layer_idx in accumulator.selected_layers:
                        activations = compute_generic_expert_activation(
                            self.experts[expert_idx], x[token_idx]
                        )
                        accumulator.record_channel_response(
                            bound_layer_idx,
                            expert_idx,
                            activations,
                            text_mask[token_idx],
                            visual_mask[token_idx],
                        )

                saved_freq_flag = getattr(self, "freq_save_dir", None)
                self.freq_save_dir = None
                try:
                    return orig_moe_infer(x, topk_ids, topk_weight, *args, **kwargs)
                finally:
                    self.freq_save_dir = saved_freq_flag

            return observed_moe_infer

        layer.mlp.gate.forward = _make_gate_forward(original_gate_forward).__get__(layer.mlp.gate)
        layer.mlp.moe_infer = _make_moe_infer(layer_idx, original_moe_infer).__get__(layer.mlp)


def parse_selected_layers(observe_layers: str) -> List[int]:
    if not observe_layers.strip():
        return list(DEFAULT_LAYERS)
    parsed = []
    for item in observe_layers.split(","):
        item = item.strip()
        if item:
            parsed.append(int(item))
    if not parsed:
        raise ValueError("--observe_layers was provided but no valid layer ids were parsed.")
    return parsed


def main() -> None:
    parser = build_base_arg_parser("A2: Collect Kimi sample-level channel activations for KDE.")
    for action in parser._actions:
        if action.dest in {"model_name_or_path", "output_dir"}:
            action.required = False
    parser.set_defaults(
        model_name_or_path="moonshotai/Kimi-VL-A3B-Instruct",
        dataset="gqa",
        batch_size=1,
        num_samples=64,
        subset_seed=None,
        output_dir=os.path.join("observations", "a2", "results", "kimi"),
        observe_layers="5,15,25",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="",
        help="输出文件名；为空时自动生成 kimi_kde_samples_<dataset>_<timestamp>.pt。",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="模型加载时使用的 attention 实现；默认用 eager，兼容当前 Kimi 架构。",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="single_gpu",
        choices=["single_gpu", "auto", "cpu"],
        help="模型加载方式；A2 默认用 single_gpu，避免 auto 分片/懒加载下手动取 expert 激活触发 meta device 错误。",
    )
    args = parser.parse_args()

    family = infer_model_family(args.model_name_or_path)
    if family != "kimi":
        raise ValueError(
            f"This script only supports Kimi models, but got family={family} from {args.model_name_or_path}."
        )
    if args.batch_size != 1:
        raise ValueError("This script stores sample-level tensors and currently requires --batch_size 1.")

    ensure_dir(args.output_dir)
    selected_layers = parse_selected_layers(args.observe_layers)

    bundle = load_model_bundle(
        args.model_name_or_path,
        device_map=resolve_device_map(args.device_map),
        attn_implementation=args.attn_implementation,
    )
    layer_to_num_experts, layer_to_num_channels = discover_layer_structure(bundle)

    missing_layers = [layer for layer in selected_layers if layer not in layer_to_num_experts]
    if missing_layers:
        raise ValueError(f"Selected layers are not observable MoE layers for this model: {missing_layers}")

    accumulator = SampleLevelChannelAccumulator(
        layer_to_num_experts=layer_to_num_experts,
        layer_to_num_channels=layer_to_num_channels,
        selected_layers=selected_layers,
    )
    attach_kimi_sample_observer(bundle, accumulator)

    normalized_dataset = normalize_dataset_name(args.dataset)
    data = build_dataset(normalized_dataset, bundle.family)
    subset_end = min(args.start_idx + args.num_samples, len(data))
    subset_indices = list(range(args.start_idx, subset_end))
    subset = Subset(data, subset_indices)
    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    with torch.no_grad():
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Collecting Kimi KDE samples",
            unit="sample",
            dynamic_ncols=True,
        )
        for batch in progress:
            accumulator.start_sample()
            inputs = prepare_inputs(bundle, batch, normalized_dataset)
            inputs = move_inputs_to_model_device(bundle.model, inputs)
            bundle.model(
                **filter_model_forward_inputs(bundle.model, inputs),
                use_cache=False,
                return_dict=True,
            )
            accumulator.finish_sample()

    payload = accumulator.to_payload(
        model_name_or_path=args.model_name_or_path,
        dataset=normalized_dataset,
        num_samples=len(subset_indices),
    )
    payload["start_idx"] = args.start_idx
    payload["source_indices"] = subset_indices

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = args.save_name or f"kimi_kde_samples_{normalized_dataset}_{timestamp}.pt"
    output_path = os.path.join(args.output_dir, filename)
    torch.save(payload, output_path)
    print_saved_artifact_message(output_path, "A2 Kimi sample-level KDE 数据")


if __name__ == "__main__":
    main()
