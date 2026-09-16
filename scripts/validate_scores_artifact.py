#!/usr/bin/env python3
"""Validate a mixed-calibration scores artifact before compiling EP4 plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from validate_modality_scores import validate_modality_scores


def parse_layer_spec(value: str) -> list[int]:
    layers: list[int] = []
    for token in value.replace(",", " ").split():
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid descending layer range: {token}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(token))
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("expected layers must be non-empty and unique")
    return sorted(layers)


def int_keys(value: Any, label: str) -> set[int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dictionary")
    return {int(key) for key in value}


def require_layer_map(payload: dict[str, Any], key: str, expected: set[int]) -> dict:
    value = payload.get(key)
    actual = int_keys(value, key)
    if actual != expected:
        raise ValueError(
            f"{key} layer coverage mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def tensor_values_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(tensor_values_finite(item) for item in value.values())
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    return bool(torch.isfinite(tensor).all())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--expected-layers", required=True)
    parser.add_argument("--expected-token-budget", type=int, default=262144)
    parser.add_argument("--expected-samples", type=int, default=512)
    args = parser.parse_args()

    scores_path = args.scores.resolve()
    payload = torch.load(scores_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid scores payload type: {type(payload).__name__}")

    expected_list = parse_layer_spec(args.expected_layers)
    expected = set(expected_list)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit("scores metadata is missing")
    metadata_layers = {int(layer) for layer in metadata.get("layers", [])}
    if metadata_layers != expected:
        raise SystemExit(
            f"metadata.layers mismatch: missing={sorted(expected - metadata_layers)}, "
            f"extra={sorted(metadata_layers - expected)}"
        )

    required_nested = {
        "channel_scores": "gateup_act",
        "expert_scores": "second_attr",
    }
    try:
        if "internvl" in str(metadata.get("model_name_or_path", "")).lower():
            validate_modality_scores(payload, expected_list)
        for outer_key, metric in required_nested.items():
            outer = payload.get(outer_key)
            if not isinstance(outer, dict) or metric not in outer:
                raise ValueError(f"missing {outer_key}.{metric}")
            layer_map = require_layer_map(outer, metric, expected)
            for layer, value in layer_map.items():
                if not tensor_values_finite(value):
                    raise ValueError(f"{outer_key}.{metric}.{layer} contains non-finite values")

        for key in (
            "layerwise_loss",
            "layerwise_second_order_sum",
            "ema_matrix",
        ):
            layer_map = require_layer_map(payload, key, expected)
            for layer, value in layer_map.items():
                if not tensor_values_finite(value):
                    raise ValueError(f"{key}.{layer} contains non-finite values")

        layerwise_loss = require_layer_map(payload, "layerwise_loss", expected)
        invalid_loss_layers = sorted(
            int(layer)
            for layer, value in layerwise_loss.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        )
        if invalid_loss_layers:
            raise ValueError(f"layerwise_loss must be positive for every layer: {invalid_loss_layers}")

        if metadata.get("score_token_budget") != args.expected_token_budget:
            raise ValueError(
                f"score_token_budget={metadata.get('score_token_budget')!r}, "
                f"expected {args.expected_token_budget}"
            )
        if metadata.get("score_tokens_per_sample") is not None:
            raise ValueError("score_tokens_per_sample must be null for variable token quotas")
        if metadata.get("score_token_counts_variable") is not True:
            raise ValueError("score_token_counts_variable must be true")
        if metadata.get("selected_num_samples") != args.expected_samples:
            raise ValueError(
                f"selected_num_samples={metadata.get('selected_num_samples')!r}, "
                f"expected {args.expected_samples}"
            )
        if metadata.get("score_aggregation") != "mean":
            raise ValueError(f"score_aggregation={metadata.get('score_aggregation')!r}, expected 'mean'")

        manifest_value = metadata.get("selection_manifest")
        manifest_hash = metadata.get("selection_manifest_sha256")
        if not manifest_value or not manifest_hash:
            raise ValueError("selection manifest path/hash metadata is missing")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if not manifest_path.is_file():
            raise ValueError(f"selection manifest does not exist: {manifest_path}")
        if sha256(manifest_path) != manifest_hash:
            raise ValueError("selection manifest SHA256 does not match scores metadata")
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid scores artifact {scores_path}: {exc}") from exc

    print(
        json.dumps(
            {
                "scores": str(scores_path),
                "status": "valid",
                "layers": expected_list,
                "score_token_budget": metadata["score_token_budget"],
                "selected_num_samples": metadata["selected_num_samples"],
                "manifest_sha256": metadata["selection_manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
