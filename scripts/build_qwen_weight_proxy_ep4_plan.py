#!/usr/bin/env python3
"""Build a performance-only EP4 plan from real Qwen expert weight magnitudes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_mask.ep4_intplan import plan_ep4_intplan
from src.vllm_ep4_plan import SCHEMA_VERSION, validate_ep4_plan

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prune-ratio", type=float, default=0.30)
    parser.add_argument("--placement-tolerance", type=float, default=0.01)
    return parser.parse_args()


def load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = model_path / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing plan: {output_path}")

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    num_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["num_experts"])
    width = int(text_config["moe_intermediate_size"])
    if (num_layers, num_experts, width) != (48, 128, 768):
        raise SystemExit(
            f"unexpected Qwen3-VL MoE shape {(num_layers, num_experts, width)}"
        )

    index_path = model_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    channel_scores = torch.empty((num_layers, num_experts, width), dtype=torch.float32)
    started = time.monotonic()
    for layer in range(num_layers):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        gate_up = load_tensor(model_path, weight_map, f"{prefix}.gate_up_proj")
        down = load_tensor(model_path, weight_map, f"{prefix}.down_proj")
        if tuple(gate_up.shape) != (num_experts, int(text_config["hidden_size"]), 2 * width):
            raise ValueError(f"unexpected layer {layer} gate_up shape {tuple(gate_up.shape)}")
        if tuple(down.shape) != (num_experts, width, int(text_config["hidden_size"])):
            raise ValueError(f"unexpected layer {layer} down shape {tuple(down.shape)}")

        gate_up_score = gate_up.abs().mean(dim=1, dtype=torch.float32)
        down_score = down.abs().mean(dim=2, dtype=torch.float32)
        channel_scores[layer] = (
            gate_up_score[:, :width]
            + gate_up_score[:, width:]
            + down_score
        )
        del gate_up, down, gate_up_score, down_score
        print(
            f"[weight proxy] layer={layer + 1}/{num_layers} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    expert_sensitivity = channel_scores.mean(dim=-1)
    layer_sensitivity = expert_sensitivity.mean(dim=-1)
    plan = plan_ep4_intplan(
        layer_sensitivity=layer_sensitivity,
        expert_sensitivity=expert_sensitivity,
        scores=channel_scores,
        prune_ratio=args.prune_ratio,
        widths=(0, 384, 512, 640, 768),
        placement_tolerance=args.placement_tolerance,
        strict_placement_tolerance=False,
        verbose=True,
    )
    plan.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model": MODEL_ID,
            "model_layer_ids": list(range(num_layers)),
            "source_model_path": str(model_path),
            "source_model_revision": model_path.name,
            "plan_purpose": "performance_only",
            "score_proxy": "mean_abs_gate_plus_up_plus_down_weights",
        }
    )
    validate_ep4_plan(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(plan, handle)
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
