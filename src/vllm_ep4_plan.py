"""Validation and model presets for MAES EP4 pruning artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1
MODEL_WIDTH_PRESETS = {
    "moonshotai/Kimi-VL-A3B-Instruct": (0, 704, 960, 1152, 1408),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": (0, 384, 512, 640, 768),
    "OpenGVLab/InternVL3_5-30B-A3B-HF": (0, 384, 512, 640, 768),
    # Speed-only presets (no accuracy plan exists for these two): a fine
    # width-quantum near the top of each model's real moe_intermediate_size,
    # chosen empirically so a forced-balanced tier budget stays reachable
    # near both p=0.30 and p=0.50 (see build_ep4_balanced_plan.py).
    "mistralai/Mistral-Small-4-119B-2603": (0, 1280, 1536, 1792, 2048),
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8": (0, 960, 1152, 1344, 1536),
}


def validate_ep4_plan(plan: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "model",
        "model_layer_ids",
        "expert_widths",
        "rank_widths",
        "expert_to_rank",
        "expert_to_local_id",
        "local_to_global",
        "intermediate_masks",
        "active_widths",
    }
    missing = sorted(required.difference(plan))
    if missing:
        raise ValueError(f"EP4 plan is missing fields: {missing}")
    if int(plan["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported EP4 plan schema {plan['schema_version']}; expected {SCHEMA_VERSION}"
        )

    widths = torch.as_tensor(plan["expert_widths"], dtype=torch.int64, device="cpu")
    rank_widths = torch.as_tensor(plan["rank_widths"], dtype=torch.int64, device="cpu")
    expert_to_rank = torch.as_tensor(plan["expert_to_rank"], dtype=torch.int64, device="cpu")
    expert_to_local = torch.as_tensor(
        plan["expert_to_local_id"], dtype=torch.int64, device="cpu"
    )
    masks = torch.as_tensor(plan["intermediate_masks"], dtype=torch.bool, device="cpu")
    layer_ids = [int(value) for value in plan["model_layer_ids"]]
    active_widths = tuple(int(value) for value in plan["active_widths"])

    if widths.ndim != 2:
        raise ValueError(f"expert_widths must be 2D, got {tuple(widths.shape)}")
    num_layers, num_experts = widths.shape
    if len(layer_ids) != num_layers or len(set(layer_ids)) != num_layers:
        raise ValueError("model_layer_ids must contain one unique ID per planned layer")
    if expert_to_rank.shape != widths.shape or expert_to_local.shape != widths.shape:
        raise ValueError("expert rank/local mappings must match expert_widths")
    if masks.ndim != 3 or masks.shape[:2] != widths.shape:
        raise ValueError("intermediate_masks must have shape [layers, experts, channels]")
    if not masks.sum(dim=-1, dtype=torch.int64).equal(widths):
        raise ValueError("mask channel counts do not match expert_widths")
    if rank_widths.shape != (num_layers, 4):
        raise ValueError(f"rank_widths must have shape {(num_layers, 4)}")
    if len(active_widths) != 4 or any(value <= 0 for value in active_widths):
        raise ValueError("EP4 plan must contain exactly four positive active widths")
    active_widths_tensor = torch.tensor(active_widths, dtype=torch.int64, device="cpu")
    if not torch.isin(rank_widths, active_widths_tensor).all():
        raise ValueError("rank_widths contains a width outside active_widths")

    removed = widths == 0
    if bool((expert_to_rank[removed] != -1).any()) or bool(
        (expert_to_local[removed] != -1).any()
    ):
        raise ValueError("removed experts must have rank/local ID -1")
    active = ~removed
    if bool(((expert_to_rank[active] < 0) | (expert_to_rank[active] >= 4)).any()):
        raise ValueError("active expert ranks must be in [0, 3]")
    if bool((expert_to_local[active] < 0).any()):
        raise ValueError("active experts must have non-negative local IDs")

    local_to_global = plan["local_to_global"]
    if len(local_to_global) != num_layers:
        raise ValueError("local_to_global must contain one entry per planned layer")
    for layer in range(num_layers):
        present_widths = set(int(value) for value in widths[layer].unique().tolist() if value > 0)
        placed_widths = set(int(value) for value in rank_widths[layer].tolist())
        if not present_widths.issubset(placed_widths):
            raise ValueError(f"layer {layer} has an active tier without an EP rank")
        if len(local_to_global[layer]) != 4:
            raise ValueError(f"layer {layer} must contain four rank mappings")
        seen: set[int] = set()
        for rank, global_ids in enumerate(local_to_global[layer]):
            expected_width = int(rank_widths[layer, rank])
            for local_id, expert_id_value in enumerate(global_ids):
                expert_id = int(expert_id_value)
                if not 0 <= expert_id < num_experts or expert_id in seen:
                    raise ValueError(f"invalid/duplicate expert {expert_id} in layer {layer}")
                seen.add(expert_id)
                if int(expert_to_rank[layer, expert_id]) != rank:
                    raise ValueError("local_to_global disagrees with expert_to_rank")
                if int(expert_to_local[layer, expert_id]) != local_id:
                    raise ValueError("local_to_global disagrees with expert_to_local_id")
                if int(widths[layer, expert_id]) != expected_width:
                    raise ValueError("expert width disagrees with rank width")
        expected_active = set(torch.where(active[layer])[0].tolist())
        if seen != expected_active:
            raise ValueError(f"layer {layer} rank mappings do not cover every active expert")

    plan["expert_widths"] = widths
    plan["rank_widths"] = rank_widths
    plan["expert_to_rank"] = expert_to_rank
    plan["expert_to_local_id"] = expert_to_local
    plan["intermediate_masks"] = masks
    plan["model_layer_ids"] = layer_ids
    plan["active_widths"] = active_widths
    return plan


def load_ep4_plan(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"EP4 plan must be a dict, got {type(payload).__name__}")
    payload["plan_path"] = str(resolved)
    return validate_ep4_plan(payload)
