import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, Optional

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from observations.common import (
    build_dataset,
    compute_generic_expert_activation,
    custom_collate_fn,
    discover_layer_structure,
    ensure_dir,
    load_model_bundle,
    resolve_model_name_or_path,
)
from src.calibration.collector.utils import (
    safe_add_with_ema,
    _is_fused_expert_container,
    _normalize_per_layer_counts,
    _to_nested_expert_dict,
    _tensor_map_to_nested_dict,
    _scalar_map_to_nested_dict,
)
from src.calibration.collector.naive_metric_fn import channel_rms, weight_rms
from src.calibration.block_forward import block_forward
# from src.calibration.threshold_calibration import (
#     run_threshold_calibration as _run_threshold_calibration,
#     build_arg_parser as _threshold_arg_parser,
# )


CHANNEL_METRICS = (
    "activation",
    "gateup_act",
    "activation_text",
    "activation_visual",
    "channel_second_order_text",
    "channel_second_order_visual",
    "saliency",
    "wa",
    "grad",
    "token_contrib",
    "wg",
    "weight",
)
EXPERT_METRICS = (
    "first_attr_usage",
    "usage",
    "usage_text",
    "usage_visual",
    "second_exact_attr",
    "true_ablate",
    "expert_modality_affinity",
)

class ModalityActivationAccumulator:
    def __init__(self, layer_to_num_experts: Dict[int, int], layer_to_num_channels: Dict[int, int]) -> None:
        self.layers = sorted(layer_to_num_experts.keys())
        self.activation_text = {
            layer_idx: torch.zeros(layer_to_num_experts[layer_idx], layer_to_num_channels[layer_idx], dtype=torch.float32)
            for layer_idx in self.layers
        }
        self.activation_visual = {
            layer_idx: torch.zeros(layer_to_num_experts[layer_idx], layer_to_num_channels[layer_idx], dtype=torch.float32)
            for layer_idx in self.layers
        }
        self.usage_text = {
            layer_idx: torch.zeros(layer_to_num_experts[layer_idx], dtype=torch.float32)
            for layer_idx in self.layers
        }
        self.usage_visual = {
            layer_idx: torch.zeros(layer_to_num_experts[layer_idx], dtype=torch.float32)
            for layer_idx in self.layers
        }

    def update(
        self,
        layer_idx: int,
        expert_idx: int,
        activations: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        visual_mask: Optional[torch.Tensor],
        ema: float,
    ) -> None:
        if text_mask is not None and bool(text_mask.any()):
            text_score = channel_rms(activations[text_mask]).detach().cpu().float()
            current = self.activation_text[layer_idx][expert_idx]
            if float(self.usage_text[layer_idx][expert_idx].item()) == 0.0:
                self.activation_text[layer_idx][expert_idx] = text_score
            else:
                self.activation_text[layer_idx][expert_idx] = safe_add_with_ema(current, ema, text_score)
            self.usage_text[layer_idx][expert_idx] += float(text_mask.sum().item())

        if visual_mask is not None and bool(visual_mask.any()):
            visual_score = channel_rms(activations[visual_mask]).detach().cpu().float()
            current = self.activation_visual[layer_idx][expert_idx]
            if float(self.usage_visual[layer_idx][expert_idx].item()) == 0.0:
                self.activation_visual[layer_idx][expert_idx] = visual_score
            else:
                self.activation_visual[layer_idx][expert_idx] = safe_add_with_ema(current, ema, visual_score)
            self.usage_visual[layer_idx][expert_idx] += float(visual_mask.sum().item())

    def finalize(self):
        self.usage_text = _normalize_per_layer_counts(self.usage_text)
        self.usage_visual = _normalize_per_layer_counts(self.usage_visual)


class RichScoreAccumulator:
    def __init__(
        self,
        layer_to_num_experts: Dict[int, int],
        layer_to_num_channels: Dict[int, int],
        modality_aware: bool,
    ) -> None:
        self.layer_to_num_experts = layer_to_num_experts
        self.layer_to_num_channels = layer_to_num_channels
        self.layers = sorted(layer_to_num_experts.keys())
        self.modality_aware = modality_aware

        self.expert_scores: Dict[str, Dict[int, torch.Tensor]] = {}
        for metric in CHANNEL_METRICS:
            self.expert_scores[metric] = {}
        for metric in EXPERT_METRICS:
            self.expert_scores[metric] = {}

        self.gate_scores: Dict[str, Dict[int, torch.Tensor]] = {
            "usage": {},
            "router": {},
            "usage_text": {},
            "usage_visual": {},
        }
        self.hit_counts: Dict[int, torch.Tensor] = {}
        self.layerwise_loss: Dict[int, float] = {}

        for layer_idx in self.layers:
            e = layer_to_num_experts[layer_idx]
            i = layer_to_num_channels[layer_idx]
            for metric in CHANNEL_METRICS:
                self.expert_scores[metric][layer_idx] = torch.zeros(e, i, dtype=torch.float32)
            for metric in EXPERT_METRICS:
                self.expert_scores[metric][layer_idx] = torch.zeros(e, dtype=torch.float32)
            self.gate_scores["usage"][layer_idx] = torch.zeros(e, dtype=torch.float32)
            self.gate_scores["router"][layer_idx] = torch.zeros(e, dtype=torch.float32)
            self.gate_scores["usage_text"][layer_idx] = torch.zeros(e, dtype=torch.float32)
            self.gate_scores["usage_visual"][layer_idx] = torch.zeros(e, dtype=torch.float32)
            self.hit_counts[layer_idx] = torch.zeros(e, dtype=torch.int64)

        self.modality_scores = (
            ModalityActivationAccumulator(layer_to_num_experts, layer_to_num_channels)
            if modality_aware else None
        )

    def absorb_layer_scores(self, layer_idx: int, copied_block) -> None:
        expert_container = copied_block.mlp.experts
        if _is_fused_expert_container(expert_container):
            num_experts = self.layer_to_num_experts[layer_idx]
            for metric in CHANNEL_METRICS:
                value = getattr(expert_container, metric, None)
                if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == num_experts:
                    self.expert_scores[metric][layer_idx] = value.detach().cpu().float()
            for metric in EXPERT_METRICS:
                value = getattr(expert_container, metric, None)
                if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == num_experts:
                    self.expert_scores[metric][layer_idx] = value.detach().cpu().float().view(-1)
            for metric in ("usage", "router", "usage_text", "usage_visual"):
                value = getattr(expert_container, metric, None)
                if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == num_experts:
                    self.gate_scores[metric][layer_idx] = value.detach().cpu().float().view(-1)
            activation = getattr(expert_container, "activation", None)
            if isinstance(activation, torch.Tensor) and activation.ndim >= 2 and activation.shape[0] == num_experts:
                self.hit_counts[layer_idx] = (activation.detach().cpu().float().abs().sum(dim=1) > 0).to(torch.int64)
            return

        experts = list(expert_container)
        for eid, expert in enumerate(experts):
            for metric in CHANNEL_METRICS:
                value = getattr(expert, metric, None)
                if value is None:
                    continue
                self.expert_scores[metric][layer_idx][eid] = value.detach().cpu().float()
            for metric in EXPERT_METRICS:
                value = getattr(expert, metric, None)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    self.expert_scores[metric][layer_idx][eid] = value.detach().cpu().float().reshape(()).item()
                else:
                    self.expert_scores[metric][layer_idx][eid] = float(value)
            usage = getattr(expert, "usage", None)
            if usage is not None:
                self.gate_scores["usage"][layer_idx][eid] = float(usage)
            router = getattr(expert, "router", None)
            if router is not None:
                self.gate_scores["router"][layer_idx][eid] = float(router)
            usage_text = getattr(expert, "usage_text", None)
            if usage_text is not None:
                self.gate_scores["usage_text"][layer_idx][eid] = float(usage_text)
            usage_visual = getattr(expert, "usage_visual", None)
            if usage_visual is not None:
                self.gate_scores["usage_visual"][layer_idx][eid] = float(usage_visual)
            if getattr(expert, "activation", None) is not None:
                self.hit_counts[layer_idx][eid] = 1

    def finalize(self) -> None:
        self.gate_scores["usage"] = _normalize_per_layer_counts(self.gate_scores["usage"])
        self.gate_scores["router"] = _normalize_per_layer_counts(self.gate_scores["router"])
        self.gate_scores["usage_text"] = _normalize_per_layer_counts(self.gate_scores["usage_text"])
        self.gate_scores["usage_visual"] = _normalize_per_layer_counts(self.gate_scores["usage_visual"])
        if self.modality_scores is not None:
            self.modality_scores.finalize()

    def build_scores_payload(self, args) -> dict:
        expert_scores = {
            metric: (
                _to_nested_expert_dict(self.expert_scores[metric], scalar=False)
                if metric in CHANNEL_METRICS
                else _to_nested_expert_dict(self.expert_scores[metric], scalar=True)
            )
            for metric in tuple(CHANNEL_METRICS) + tuple(EXPERT_METRICS)
        }
        ema_matrix = {}
        for layer_idx in self.layers:
            text_freq = self.gate_scores["usage_text"][layer_idx]
            visual_freq = self.gate_scores["usage_visual"][layer_idx]
            ema_matrix[layer_idx] = (visual_freq - text_freq) / (visual_freq + text_freq + 1e-8)

        payload = {
            "model_name_or_path": args.model_name_or_path,
            "resolved_model_name_or_path": resolve_model_name_or_path(args.model_name_or_path),
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "start_idx": args.start_idx,
            "subset_seed": args.subset_seed,
            "ema": args.ema,
            "modality_aware": self.modality_aware,
            "layers": self.layers,
            "layer_to_num_experts": self.layer_to_num_experts,
            "layer_to_num_channels": self.layer_to_num_channels,
            "available_channel_metrics": list(CHANNEL_METRICS),
            "available_expert_metrics": list(EXPERT_METRICS),
            "expert_scores": expert_scores,
            "gate_scores": {
                "usage": _to_nested_expert_dict(self.gate_scores["usage"], scalar=True),
                "router": _to_nested_expert_dict(self.gate_scores["router"], scalar=True),
                "usage_text": _to_nested_expert_dict(self.gate_scores["usage_text"], scalar=True),
                "usage_visual": _to_nested_expert_dict(self.gate_scores["usage_visual"], scalar=True),
            },
            "ema_matrix": _to_nested_expert_dict(ema_matrix, scalar=True),
            "layerwise_loss": dict(self.layerwise_loss),
            # Backward-compatible aliases
            "scores": _tensor_map_to_nested_dict(self.expert_scores["activation"]),
            "first_attr_usage": _scalar_map_to_nested_dict(
                self.expert_scores["first_attr_usage"]
            ),
            "expert_usage": _scalar_map_to_nested_dict(self.gate_scores["usage"]),
            "expert_router": _scalar_map_to_nested_dict(self.gate_scores["router"]),
        }
        if self.modality_scores is not None:
            payload["modality_channel_scores"] = {
                "text": _tensor_map_to_nested_dict(self.modality_scores.activation_text),
                "visual": _tensor_map_to_nested_dict(self.modality_scores.activation_visual),
            }
            payload["gate_scores"]["usage_text"] = _scalar_map_to_nested_dict(
                self.modality_scores.usage_text
            )
            payload["gate_scores"]["usage_visual"] = _scalar_map_to_nested_dict(
                self.modality_scores.usage_visual
            )
        return payload


def attach_kimi_modality_hooks(model, config, accumulator: ModalityActivationAccumulator, ema: float):
    states = []
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            continue

        states.append(
            (
                layer.mlp,
                getattr(layer.mlp, "freq_save_dir", None),
                getattr(layer.mlp.gate, "layer_idx", None),
                layer.mlp.moe_infer,
            )
        )
        layer.mlp.freq_save_dir = "__observation__"
        layer.mlp.gate.layer_idx = layer_idx
        original_moe_infer = layer.mlp.moe_infer

        def observed_moe_infer(
            self,
            x,
            topk_ids,
            topk_weight,
            *args,
            __orig=original_moe_infer,
            __layer_idx=layer_idx,
            __acc=accumulator,
            __ema=ema,
            **kwargs,
        ):
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
                with torch.no_grad():
                    activations = compute_generic_expert_activation(
                        self.experts[expert_idx], x[token_idx]
                    )
                __acc.update(
                    __layer_idx,
                    expert_idx,
                    activations,
                    text_mask[token_idx],
                    visual_mask[token_idx],
                    __ema,
                )

            saved = getattr(self, "freq_save_dir", None)
            self.freq_save_dir = None
            try:
                return __orig(x, topk_ids, topk_weight, *args, **kwargs)
            finally:
                self.freq_save_dir = saved

        layer.mlp.moe_infer = observed_moe_infer.__get__(layer.mlp)
    return states


def restore_kimi_modality_hooks(states) -> None:
    for mlp, old_freq_save_dir, old_layer_idx, old_moe_infer in states:
        mlp.freq_save_dir = old_freq_save_dir
        mlp.moe_infer = old_moe_infer
        mlp.gate.layer_idx = old_layer_idx


def collect_weight_scores(model, config, accumulator: RichScoreAccumulator) -> None:
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            continue
        for eid, expert in enumerate(layer.mlp.experts):
            g = expert.gate_proj.weight
            u = expert.up_proj.weight
            d = expert.down_proj.weight
            score = (weight_rms(d, channel_dim=1) + weight_rms(u, channel_dim=0) + weight_rms(g, channel_dim=0)) / 3.0
            accumulator.expert_scores["weight"][layer_idx][eid] = score.detach().cpu().float()
            accumulator.hit_counts[layer_idx][eid] = 1


def save_score_artifacts(output_dir: str, accumulator: RichScoreAccumulator, args) -> None:
    snapshot = copy.deepcopy(accumulator)
    snapshot.finalize()
    scores_path = os.path.join(output_dir, "scores.pt")
    torch.save(snapshot.build_scores_payload(args), scores_path)
    print(f"[calibration] Saved scores: {scores_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect channel scores for Kimi-VL on multimodal calibration data."
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="gqa", choices=["gqa", "coco"])
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--ema", type=float, default=0.9)
    p.add_argument("--modality_aware", action="store_true")
    p.add_argument("--force", "-f", action="store_true")
    return p


def run_collection(args) -> None:
    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, "scores.pt")
    if os.path.exists(out_path) and not args.force:
        print(
            f"[calibration] Found existing scores at {out_path}. "
            "Pass --force to overwrite."
        )
        return

    bundle = load_model_bundle(args.model_name_or_path)
    if bundle.family != "kimi":
        raise NotImplementedError("The current calibration adapter only supports Kimi-VL.")

    model = bundle.model
    config = model.config.text_config
    layer_to_num_experts, layer_to_num_channels = discover_layer_structure(bundle)
    accumulator = RichScoreAccumulator(
        layer_to_num_experts,
        layer_to_num_channels,
        modality_aware=args.modality_aware,
    )

    print(
        f"[calibration] Discovered {len(layer_to_num_experts)} MoE layers, "
        f"{sum(layer_to_num_experts.values())} experts total."
    )

    collect_weight_scores(model, config, accumulator)
    save_score_artifacts(args.output_dir, accumulator, args)

    dataset = build_dataset(args.dataset, bundle.family)
    pool = list(range(args.start_idx, len(dataset)))
    if args.subset_seed is not None and args.subset_seed >= 0:
        rng = random.Random(args.subset_seed)
        indices = rng.sample(pool, min(args.num_samples, len(pool)))
    else:
        indices = pool[: args.num_samples]

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # if args.modality_aware and accumulator.modality_scores is not None:
    #     print("[calibration] Collecting modality-split activation scores...")
    #     hook_states = attach_kimi_modality_hooks(model, config, accumulator.modality_scores, args.ema)
    #     try:
    #         model.eval()
    #         with torch.no_grad():
    #             for batch in tqdm(loader, desc="Collecting modality activations", unit="batch"):
    #                 inputs = prepare_inputs(bundle, batch, args.dataset)
    #                 inputs = move_inputs_to_model_device(model, inputs)
    #                 model(**inputs, use_cache=False, return_dict=True)
    #     finally:
    #         restore_kimi_modality_hooks(hook_states)

    print("[calibration] Collecting block-reconstruction scores with attn_mlp collector...")
    for layer_idx in accumulator.layers:
        teacher_block = model.language_model.model.layers[layer_idx]
        copied_block = copy.deepcopy(teacher_block)
        block_dtype = next(teacher_block.parameters()).dtype
        layer_loss = block_forward(
            bundle=bundle,
            cnt_block=copied_block,
            layer_idx=layer_idx,
            dataloader=loader,
            dataset_name=args.dataset,
            saliency_ema=args.ema,
            loss_fn="l2",
            second_order_mode="exact",  # default second-order mode is exact
            dtype=block_dtype,
            verbose=True,
        )
        accumulator.layerwise_loss[layer_idx] = float(layer_loss)
        accumulator.absorb_layer_scores(layer_idx, copied_block)
        save_score_artifacts(args.output_dir, accumulator, args)
        print(f"[calibration] Layer {layer_idx}: layer loss={layer_loss:.6f}. "
              f"Have saved to {args.output_dir}/scores.pt")

    save_score_artifacts(args.output_dir, accumulator, args)


def main() -> None:
    # import sys as _sys
    # if len(_sys.argv) > 1 and _sys.argv[1] == "threshold":
    #     _sys.argv.pop(1)
    #     args = _threshold_arg_parser().parse_args()
    #     _run_threshold_calibration(args)
    # else:
    args = build_arg_parser().parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
