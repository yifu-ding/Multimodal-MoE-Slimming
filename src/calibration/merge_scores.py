import argparse
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


def _resolve_scores_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "scores.pt")
    return path


def _to_int_keyed_map(maybe_nested: Dict[Any, Any]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for k, v in maybe_nested.items():
        out[int(k)] = v
    return out


def _extract_layers(payload: dict) -> List[int]:
    meta = payload.get("metadata", {})
    layers = meta.get("layers")
    if isinstance(layers, list) and layers:
        return [int(x) for x in layers]

    channel_scores = payload.get("channel_scores", {})
    if not channel_scores:
        return []
    first_metric = next(iter(channel_scores.keys()))
    return sorted(int(k) for k in channel_scores[first_metric].keys())


def _source_timestamp(path: str, payload: dict) -> float:
    meta = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    ts = meta.get("created_at_unix")
    if isinstance(ts, (int, float)):
        return float(ts)
    return float(os.path.getmtime(path))


@dataclass
class LoadedScores:
    path: str
    payload: dict
    layers: List[int]
    ts: float


def _pick_winners(loaded: List[LoadedScores]) -> Tuple[Dict[int, int], List[str]]:
    winner_idx_by_layer: Dict[int, int] = {}
    warnings: List[str] = []
    for idx, item in enumerate(loaded):
        for layer in item.layers:
            if layer not in winner_idx_by_layer:
                winner_idx_by_layer[layer] = idx
                continue
            prev_idx = winner_idx_by_layer[layer]
            prev = loaded[prev_idx]
            if item.ts > prev.ts:
                warnings.append(
                    f"[merge warning] layer={layer} found in multiple files: "
                    f"use newer {item.path} (ts={item.ts:.3f}) over {prev.path} (ts={prev.ts:.3f})"
                )
                winner_idx_by_layer[layer] = idx
            else:
                warnings.append(
                    f"[merge warning] layer={layer} found in multiple files: "
                    f"keep newer {prev.path} (ts={prev.ts:.3f}), ignore {item.path} (ts={item.ts:.3f})"
                )
    return winner_idx_by_layer, warnings


def _get_layer_value(layer_map: Dict[Any, Any], layer: int):
    if layer in layer_map:
        return layer_map[layer]
    key_str = str(layer)
    if key_str in layer_map:
        return layer_map[key_str]
    return None


def _merge_payloads(loaded: List[LoadedScores], winner_idx_by_layer: Dict[int, int]) -> dict:
    if not loaded:
        raise ValueError("No scores payloads provided.")

    newest = max(loaded, key=lambda x: x.ts)
    merged = {
        "channel_scores": {},
        "expert_scores": {},
        "ema_matrix": {},
        "layerwise_loss": {},
        "metadata": copy.deepcopy(newest.payload.get("metadata", {})),
    }

    channel_metrics = set()
    expert_metrics = set()
    for item in loaded:
        channel_metrics.update(item.payload.get("channel_scores", {}).keys())
        expert_metrics.update(item.payload.get("expert_scores", {}).keys())

    for metric in sorted(channel_metrics):
        merged["channel_scores"][metric] = {}
        for layer, winner_idx in winner_idx_by_layer.items():
            src = loaded[winner_idx].payload.get("channel_scores", {}).get(metric, {})
            val = _get_layer_value(src, layer)
            if val is not None:
                merged["channel_scores"][metric][layer] = val

    for metric in sorted(expert_metrics):
        merged["expert_scores"][metric] = {}
        for layer, winner_idx in winner_idx_by_layer.items():
            src = loaded[winner_idx].payload.get("expert_scores", {}).get(metric, {})
            val = _get_layer_value(src, layer)
            if val is not None:
                merged["expert_scores"][metric][layer] = val

    for layer, winner_idx in winner_idx_by_layer.items():
        src_ema = loaded[winner_idx].payload.get("ema_matrix", {})
        val = _get_layer_value(src_ema, layer)
        if val is not None:
            merged["ema_matrix"][layer] = val

        src_loss = loaded[winner_idx].payload.get("layerwise_loss", {})
        val = _get_layer_value(src_loss, layer)
        if val is not None:
            merged["layerwise_loss"][layer] = val

    merged_layers = sorted(winner_idx_by_layer.keys())
    if not isinstance(merged["metadata"], dict):
        merged["metadata"] = {}
    merged["metadata"]["layers"] = merged_layers

    layer_to_num_experts: Dict[int, Any] = {}
    layer_to_num_channels: Dict[int, Any] = {}
    for layer, winner_idx in winner_idx_by_layer.items():
        src_meta = loaded[winner_idx].payload.get("metadata", {})
        if not isinstance(src_meta, dict):
            continue
        src_e = _to_int_keyed_map(src_meta.get("layer_to_num_experts", {})) if src_meta.get("layer_to_num_experts") else {}
        src_c = _to_int_keyed_map(src_meta.get("layer_to_num_channels", {})) if src_meta.get("layer_to_num_channels") else {}
        if layer in src_e:
            layer_to_num_experts[layer] = src_e[layer]
        if layer in src_c:
            layer_to_num_channels[layer] = src_c[layer]

    if layer_to_num_experts:
        merged["metadata"]["layer_to_num_experts"] = layer_to_num_experts
    if layer_to_num_channels:
        merged["metadata"]["layer_to_num_channels"] = layer_to_num_channels

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple calibration scores.pt by layer index.")
    parser.add_argument("scores", nargs="+", help="Input scores.pt files or directories containing scores.pt")
    parser.add_argument("--output", type=str, required=True, help="Output merged scores.pt path")
    args = parser.parse_args()

    loaded: List[LoadedScores] = []
    for raw in args.scores:
        path = _resolve_scores_path(raw)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"scores.pt not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        layers = _extract_layers(payload)
        ts = _source_timestamp(path, payload)
        loaded.append(LoadedScores(path=path, payload=payload, layers=layers, ts=ts))
        print(f"[merge] loaded {path} layers={layers} ts={ts:.3f}")

    winner_idx_by_layer, warnings = _pick_winners(loaded)
    for w in warnings:
        print(w)

    merged = _merge_payloads(loaded, winner_idx_by_layer)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(merged, args.output)
    print(f"[merge] saved merged scores to: {args.output}")
    print(f"[merge] merged layers: {sorted(winner_idx_by_layer.keys())}")


if __name__ == "__main__":
    main()
