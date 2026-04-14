import os
from typing import Any, Dict, Tuple

import torch

from src.base.shared_utils import _print
from src.base.shared_utils.dict_to_tensor import dict_to_tensor
from src.calibration.helpers.score_namespace import CHANNEL_METRICS, EXPERT_METRICS


__all__ = [
    "prepare_scores",
    "load_modality_channel_scores",
]


def _nested_to_layer_tensors(nested: dict) -> Dict[int, torch.Tensor]:
    out = {}
    for lid in sorted(nested.keys()):
        eids = sorted(nested[lid].keys())
        first = nested[lid][eids[0]]
        if isinstance(first, torch.Tensor):
            out[lid] = torch.stack([nested[lid][e].detach().cpu().float() for e in eids], dim=0)
        else:
            out[lid] = torch.tensor([float(nested[lid][e]) for e in eids], dtype=torch.float32)
    return out


def _load_payload(scores_dir: str, device: str) -> dict:
    path = scores_dir if os.path.isfile(scores_dir) else os.path.join(scores_dir, "scores.pt")
    return torch.load(path, map_location=device, weights_only=False)


def _payload_get(payload: dict, key: str, default=None):
    """Read a field from legacy top-level scores.pt or from ``metadata`` (new format).

    Prefer top-level when both exist so older files behave unchanged.
    """
    if key in payload:
        return payload[key]
    meta = payload.get("metadata")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return default

def load_modality_channel_scores(
    payload, 
    device: str = "cpu",
    intra_expert_metric: str = "activation",
) -> dict:
    channel_scores = {
        m: _nested_to_layer_tensors(v) for m, v in payload["channel_scores"].items()
    }
    text_scores = channel_scores[f"{intra_expert_metric}_text"]
    visual_scores = channel_scores[f"{intra_expert_metric}_visual"]

    ema_tensor = dict_to_tensor(
        _nested_to_layer_tensors(payload["ema_matrix"])
    ).to(device=device, dtype=torch.float32)

    return {
        "text": dict_to_tensor(text_scores).to(device=device, dtype=torch.float32),
        "visual": dict_to_tensor(visual_scores).to(device=device, dtype=torch.float32),
        "ema_matrix": ema_tensor,
    }

def prepare_scores(
    scores_dir: str,
    mask_method_kwargs: Dict[str, Any],
    smooth_fn: str = "sqrt",
    modality_aware: bool = False,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, Dict[str, Any], list]:
    payload = _load_payload(scores_dir, device)
    if verbose:
        _print(f"[Score Loading] Loaded scores payload from {scores_dir}")
    
    layers = _payload_get(payload, "layers")
    if layers is None:
        first_metric = next(iter(payload["channel_scores"].keys()))
        nested = payload["channel_scores"][first_metric]
        layers = sorted(int(k) for k in nested.keys())
        
    intra_expert_metric = mask_method_kwargs.get("intra_expert_metric", "activation")
    if modality_aware:
        modality_scores = load_modality_channel_scores(payload, device, intra_expert_metric)
        intermediate_scores = (modality_scores["text"] + modality_scores["visual"]) / 2.0
        _print("[prepare_scores] intermediate_scores is mean of text and visual scores")
        L, E, I = intermediate_scores.shape
        intermediate_scores = (intermediate_scores, modality_scores)
    else:
        channel_scores = {
            m: _nested_to_layer_tensors(v) for m, v in payload["channel_scores"].items()
        }
        intermediate_scores = dict_to_tensor(channel_scores[intra_expert_metric]).to(
            device=device, dtype=torch.float32
        )
        L, E, I = intermediate_scores.shape
    
    intra_layer_method = mask_method_kwargs.get("intra_layer_method", "uniform")
    if intra_layer_method == "uniform":
        expertwise_scores = torch.ones((L, E), dtype=torch.float32, device=device)
    else:
        expert_scores = {
            m: _nested_to_layer_tensors(v) for m, v in payload["expert_scores"].items()
        }
        metric_name = intra_layer_method.removesuffix("_coverage")
        if metric_name not in EXPERT_METRICS:
            raise ValueError(
                f"Invalid intra_layer_method: {intra_layer_method} is not in EXPERT_METRICS. "
            )
        if metric_name not in expert_scores:
            raise KeyError(
                f"Metric '{metric_name}' not found in payload['expert_scores']. "
                f"Available keys: {sorted(expert_scores.keys())}. Re-run score collection to get all the scores. "
            )
        expertwise_scores = dict_to_tensor(expert_scores[metric_name]).to(
            device=device, dtype=torch.float32
        )

    # ---- inter-layer loss ----
    inter_layer_method = mask_method_kwargs.get("inter_layer_method", "uniform")
    loss_based_kwargs: Dict[str, Any] = {
        "inter_layer_method": inter_layer_method,
        "smooth_fn": smooth_fn,
        "smooth_times": 0,
        "layerwise_loss": None,
    }
    if "loss" in inter_layer_method:
        raw = _payload_get(payload, "layerwise_loss")
        if raw is None:
            raw = {}
        loss_based_kwargs["layerwise_loss"] = torch.tensor(
            [raw[l] for l in sorted(raw.keys())], dtype=torch.float32, device=device
        )
        if inter_layer_method.startswith("loss_smooth_"):
            loss_based_kwargs["smooth_times"] = int(inter_layer_method.split("_")[-1])

    return (
        intermediate_scores,
        expertwise_scores,
        L,
        E,
        I,
        loss_based_kwargs,
        layers,
    )
