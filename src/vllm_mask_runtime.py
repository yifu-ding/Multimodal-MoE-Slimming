"""Apply arbitrary MAES channel masks to vLLM's unchanged MoE layout."""

from __future__ import annotations

import inspect
import os
import re

import torch


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")


def load_mask_plan(path):
    plan = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("invalid MAES mask plan")
    masks = plan.get("intermediate_masks")
    layers = plan.get("model_layer_ids")
    if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool or masks.ndim != 3:
        raise ValueError("intermediate_masks must be a bool [layers, experts, channels] tensor")
    if not isinstance(layers, (list, tuple)) or len(layers) != masks.shape[0]:
        raise ValueError("model_layer_ids must match the mask layer count")
    if len(set(map(int, layers))) != len(layers):
        raise ValueError("model_layer_ids must be unique")
    return plan


def pad_pruned_weight(weight, shard_id, mask):
    if shard_id not in ("w1", "w2", "w3") or weight.ndim != 2:
        raise ValueError(f"unsupported expert weight: shard={shard_id}, shape={tuple(weight.shape)}")
    dim = 1 if shard_id == "w2" else 0
    if weight.shape[dim] != mask.numel():
        raise ValueError(f"expert width {weight.shape[dim]} != mask width {mask.numel()}")
    shape = (1, -1) if dim == 1 else (-1, 1)
    return weight * mask.to(device=weight.device, dtype=weight.dtype).reshape(shape)


def install_into(module):
    if getattr(module, "_maes_mask_installed", False):
        return
    plan_path = os.environ["MAES_MASK_PLAN"]
    plan = load_mask_plan(plan_path)
    masks = plan["intermediate_masks"]
    positions = {int(layer): pos for pos, layer in enumerate(plan["model_layer_ids"])}
    cls = module.FusedMoE
    original_init = cls.__init__
    original_loader = cls.weight_loader
    signature = inspect.signature(original_init)

    def patched_init(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        match = _LAYER.search(str(bound.arguments.get("prefix", "")))
        if match and int(match.group(1)) in positions:
            expected = (masks.shape[1], masks.shape[2])
            actual = (bound.arguments.get("num_experts"), bound.arguments.get("intermediate_size"))
            if actual != expected:
                raise ValueError(f"MAES mask shape {expected} disagrees with FusedMoE {actual}")
        original_init(self, *args, **kwargs)
        if match and int(match.group(1)) in positions:
            layer_id = int(match.group(1))
            self._maes_channel_masks = masks[positions[layer_id]]
            print(f"[MAES MASK] layer={layer_id} original expert order, padded width={masks.shape[2]}", flush=True)

    def patched_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                       return_success=False):
        channel_masks = getattr(self, "_maes_channel_masks", None)
        if channel_masks is not None and "weight" in weight_name:
            if not 0 <= expert_id < channel_masks.shape[0]:
                raise ValueError(f"expert ID {expert_id} outside mask")
            loaded_weight = pad_pruned_weight(loaded_weight, shard_id, channel_masks[expert_id])
        return original_loader(self, param, loaded_weight, weight_name, shard_id,
                               expert_id, return_success)

    cls.__init__ = patched_init
    cls.weight_loader = patched_loader
    module._maes_mask_installed = True
    print(f"[MAES MASK] installed plan={plan_path} model={plan['model']}", flush=True)
