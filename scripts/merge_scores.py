import argparse
import copy
import numbers
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def _get_layer_value(layer_map: Dict[Any, Any], layer: int):
    if layer in layer_map:
        return layer_map[layer]
    key_str = str(layer)
    if key_str in layer_map:
        return layer_map[key_str]
    return None


def _value_has_nonzero(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, torch.Tensor):
        if v.numel() == 0:
            return False
        return bool(torch.any(v != 0).item())
    if isinstance(v, numbers.Number):
        return v != 0
    if isinstance(v, dict):
        return any(_value_has_nonzero(x) for x in v.values())
    if isinstance(v, (list, tuple, set)):
        return any(_value_has_nonzero(x) for x in v)
    try:
        t = torch.as_tensor(v)
        if t.numel() == 0:
            return False
        return bool(torch.any(t != 0).item())
    except Exception:
        return False


def _merge_nonzero_candidates(candidates: List[Tuple[float, Any]]) -> Optional[Any]:
    if not candidates:
        return None

    if any(isinstance(v, dict) for _, v in candidates):
        keys = set()
        for _, v in candidates:
            if isinstance(v, dict):
                keys.update(v.keys())
        merged_dict = {}
        for k in sorted(keys, key=lambda x: str(x)):
            child_candidates = []
            for ts, v in candidates:
                if isinstance(v, dict) and k in v and v[k] is not None:
                    child_candidates.append((ts, v[k]))
            child = _merge_nonzero_candidates(child_candidates)
            if child is not None:
                merged_dict[k] = child
        return merged_dict if merged_dict else None

    for _, v in sorted(candidates, key=lambda x: x[0], reverse=True):
        if _value_has_nonzero(v):
            return v
    return None


def _merge_latest_nonzero_value(loaded: List[LoadedScores], value_getter) -> Tuple[Optional[Any], bool]:
    """Return (merged_value, has_any_candidate_value)."""
    candidates: List[Tuple[float, Any]] = []
    has_any_value = False
    for item in loaded:
        val = value_getter(item.payload)
        if val is None:
            continue
        has_any_value = True
        candidates.append((item.ts, val))
    return _merge_nonzero_candidates(candidates), has_any_value


def _pick_latest_value(loaded: List[LoadedScores], value_getter) -> Optional[Any]:
    candidates: List[Tuple[float, Any]] = []
    for item in loaded:
        val = value_getter(item.payload)
        if val is not None:
            candidates.append((item.ts, val))
    if not candidates:
        return None
    _, winner_val = max(candidates, key=lambda x: x[0])
    return winner_val


def _merge_payloads(loaded: List[LoadedScores]) -> Tuple[dict, List[str]]:
    if not loaded:
        raise ValueError("No scores payloads provided.")

    newest = max(loaded, key=lambda x: x.ts)
    warnings: List[str] = []
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

    all_layers = sorted({layer for item in loaded for layer in item.layers})

    for metric in sorted(channel_metrics):
        merged_metric: Dict[int, Any] = {}
        for layer in all_layers:
            val, has_any_value = _merge_latest_nonzero_value(
                loaded,
                lambda payload, m=metric, l=layer: _get_layer_value(
                    payload.get("channel_scores", {}).get(m, {}), l
                ),
            )
            if val is None:
                if has_any_value:
                    warnings.append(
                        f"[merge warning] channel_scores.{metric}.{layer} all candidates are zero; skipped"
                    )
                continue
            merged_metric[layer] = val
        if merged_metric:
            merged["channel_scores"][metric] = merged_metric

    for metric in sorted(expert_metrics):
        merged_metric = {}
        for layer in all_layers:
            val, has_any_value = _merge_latest_nonzero_value(
                loaded,
                lambda payload, m=metric, l=layer: _get_layer_value(
                    payload.get("expert_scores", {}).get(m, {}), l
                ),
            )
            if val is None:
                if has_any_value:
                    warnings.append(
                        f"[merge warning] expert_scores.{metric}.{layer} all candidates are zero; skipped"
                    )
                continue
            merged_metric[layer] = val
        if merged_metric:
            merged["expert_scores"][metric] = merged_metric

    for layer in all_layers:
        val, has_any_value = _merge_latest_nonzero_value(
            loaded, lambda payload, l=layer: _get_layer_value(payload.get("ema_matrix", {}), l)
        )
        if val is not None:
            merged["ema_matrix"][layer] = val
        elif has_any_value:
            warnings.append(f"[merge warning] ema_matrix.{layer} all candidates are zero; skipped")

        val, has_any_value = _merge_latest_nonzero_value(
            loaded, lambda payload, l=layer: _get_layer_value(payload.get("layerwise_loss", {}), l)
        )
        if val is not None:
            merged["layerwise_loss"][layer] = val
        elif has_any_value:
            warnings.append(f"[merge warning] layerwise_loss.{layer} all candidates are zero; skipped")

    merged_layers = sorted(
        {
            *{int(k) for metric_map in merged["channel_scores"].values() for k in metric_map.keys()},
            *{int(k) for metric_map in merged["expert_scores"].values() for k in metric_map.keys()},
            *{int(k) for k in merged["ema_matrix"].keys()},
            *{int(k) for k in merged["layerwise_loss"].keys()},
        }
    )
    if not isinstance(merged["metadata"], dict):
        merged["metadata"] = {}
    merged["metadata"]["layers"] = merged_layers

    layer_to_num_experts: Dict[int, Any] = {}
    layer_to_num_channels: Dict[int, Any] = {}
    for layer in merged_layers:
        src_e_val = _pick_latest_value(
            loaded,
            lambda payload, l=layer: _to_int_keyed_map(
                payload.get("metadata", {}).get("layer_to_num_experts", {})
            ).get(l)
            if isinstance(payload.get("metadata", {}), dict)
            else None,
        )
        src_c_val = _pick_latest_value(
            loaded,
            lambda payload, l=layer: _to_int_keyed_map(
                payload.get("metadata", {}).get("layer_to_num_channels", {})
            ).get(l)
            if isinstance(payload.get("metadata", {}), dict)
            else None,
        )
        if src_e_val is not None:
            layer_to_num_experts[layer] = src_e_val
        if src_c_val is not None:
            layer_to_num_channels[layer] = src_c_val

    if layer_to_num_experts:
        merged["metadata"]["layer_to_num_experts"] = layer_to_num_experts
    if layer_to_num_channels:
        merged["metadata"]["layer_to_num_channels"] = layer_to_num_channels

    return merged, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple calibration scores.pt by layer index.")
    parser.add_argument("scores", nargs="+", help="Input scores.pt files or directories containing scores.pt")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output merged scores.pt path")
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

    merged, warnings = _merge_payloads(loaded)
    for w in warnings:
        print(w)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(merged, args.output)
    print(f"[merge] saved merged scores to: {args.output}")
    print(f"[merge] merged layers: {merged.get('metadata', {}).get('layers', [])}")


if __name__ == "__main__":
    main()


############# take notes ###############
# /home/dyf/code/distill/MoDES/storage/prune/scores/kimi-vl-a3b_coco-rell2-04142126/scores.pt  
# /home/dyf/code/distill/MoDES/storage/prune/scores/kimi-vl-a3b_coco-rell2-04142125/scores.pt
# merged -> /home/dyf/code/distill/MoDES/storage/prune/scores/kimi-vl-a3b_coco-rell2-041421.pt
