"""Strictly merge method-validation B layer shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != "method_validation_b_shard":
        raise ValueError(f"Not a method-validation B shard: {path}")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported shard schema: {path}")
    return payload


def merge(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("At least one shard is required.")
    payloads = [load(path) for path in paths]
    reference = dict(payloads[0]["metadata"])
    reference.pop("sweep_layer", None)
    layers = {}
    sweep_layers = []
    for path, payload in zip(paths, payloads):
        metadata = dict(payload["metadata"])
        sweep_layer = metadata.pop("sweep_layer", None)
        if metadata != reference:
            raise ValueError(f"Shard metadata mismatch: {path}")
        if sweep_layer is not None:
            sweep_layers.append(int(sweep_layer))
        for raw_layer, result in payload["layers"].items():
            layer_idx = int(raw_layer)
            if layer_idx in layers:
                raise ValueError(f"Duplicate layer {layer_idx} in {path}.")
            layers[layer_idx] = result
    actual_sweeps = [layer for layer, result in layers.items() if result.get("beta_sweep") is not None]
    if len(actual_sweeps) != 1:
        raise ValueError(f"Expected exactly one beta-sweep layer, got {sorted(actual_sweeps)}.")
    if sweep_layers and set(sweep_layers) != set(actual_sweeps):
        raise ValueError(
            f"Declared sweep layers {sorted(set(sweep_layers))} do not match data {actual_sweeps}."
        )
    return {
        "schema_version": 1,
        "kind": "method_validation_b",
        "layers": dict(sorted(layers.items())),
        "metadata": {**reference, "sweep_layer": actual_sweeps[0], "num_layers": len(layers)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {output}; pass --force to overwrite.")
    payload = merge([path.expanduser().resolve() for path in args.shards])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"Merged {len(args.shards)} shards / {len(payload['layers'])} layers to {output}")


if __name__ == "__main__":
    main()
