"""Print MoE decoder layer indices without loading checkpoint weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observations.common import resolve_model_name_or_path


def discover_moe_layers(config) -> list[int]:
    def get(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    text_config = get(config, "text_config", config)
    num_layers = int(get(text_config, "num_hidden_layers"))
    num_experts = int(
        get(text_config, "num_experts", get(text_config, "n_routed_experts", 0))
    )
    if num_experts <= 0:
        raise ValueError("The model config does not describe an MoE model.")

    mlp_only_layers = {int(layer) for layer in get(text_config, "mlp_only_layers", [])}
    if get(text_config, "decoder_sparse_step") is not None:
        sparse_step = int(get(text_config, "decoder_sparse_step"))
        return [
            layer
            for layer in range(num_layers)
            if (layer + 1) % sparse_step == 0 and layer not in mlp_only_layers
        ]

    first_moe_layer = int(get(text_config, "first_k_dense_replace", 0))
    moe_frequency = int(get(text_config, "moe_layer_freq", 1))
    return [
        layer
        for layer in range(num_layers)
        if layer >= first_moe_layer and layer % moe_frequency == 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    model_path = Path(resolve_model_name_or_path(args.model))
    with (model_path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    for layer in discover_moe_layers(config):
        print(layer)


if __name__ == "__main__":
    main()
