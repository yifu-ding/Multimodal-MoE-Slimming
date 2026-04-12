import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from src.base.shared_utils import _print
from src.base.shared_utils.dict_to_tensor import dict_to_tensor
from src.generate_mask.planners import inter_layer_planner

__all__ = [
    "load_channel_scores",
    "load_layerwise_loss",
    "load_attention_head_scores",
    "load_modality_channel_scores",
    "prepare_scores",
]


def _load_legacy_payload(path: str, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    def _nested_scores_to_layer_tensors(nested):
        output = {}
        for lid in sorted(nested.keys()):
            expert_ids = sorted(nested[lid].keys())
            output[lid] = torch.stack(
                [nested[lid][eid].detach().cpu().float() for eid in expert_ids],
                dim=0,
            )
        return output
    expert_scores = {
        "activation": _nested_scores_to_layer_tensors(payload["scores"])
    }
    if payload.get("modality_channel_scores") is not None:
        expert_scores["activation_text"] = _nested_scores_to_layer_tensors(
            payload["modality_channel_scores"]["text"]
        )
        expert_scores["activation_visual"] = _nested_scores_to_layer_tensors(
            payload["modality_channel_scores"]["visual"]
        )
    gate_scores = {}
    if payload.get("expert_usage") is not None:
        gate_scores["usage"] = {
            lid: torch.tensor(
                [payload["expert_usage"][lid][eid] for eid in sorted(payload["expert_usage"][lid].keys())],
                dtype=torch.float32,
            )
            for lid in sorted(payload["expert_usage"].keys())
        }
    if payload.get("expert_router") is not None:
        gate_scores["router"] = {
            lid: torch.tensor(
                [payload["expert_router"][lid][eid] for eid in sorted(payload["expert_router"][lid].keys())],
                dtype=torch.float32,
            )
            for lid in sorted(payload["expert_router"].keys())
        }
    metadata = {
        "modality_aware": bool(payload.get("modality_aware", False)),
        "layers": sorted(payload["scores"].keys()),
    }
    return expert_scores, gate_scores, metadata, payload


def _nested_scores_to_layer_tensors(nested):
    output = {}
    for lid in sorted(nested.keys()):
        expert_ids = sorted(nested[lid].keys())
        first_val = nested[lid][expert_ids[0]] if expert_ids else None
        if first_val is None:
            output[lid] = torch.empty(0)
            continue
        if isinstance(first_val, torch.Tensor):
            output[lid] = torch.stack(
                [nested[lid][eid].detach().cpu().float() for eid in expert_ids],
                dim=0,
            )
        else:
            output[lid] = torch.tensor(
                [float(nested[lid][eid]) for eid in expert_ids],
                dtype=torch.float32,
            )
    return output


def _load_scores_payload(path: str, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    if "expert_scores" not in payload:
        return _load_legacy_payload(path, device)
    expert_scores = {
        metric: _nested_scores_to_layer_tensors(nested)
        for metric, nested in payload["expert_scores"].items()
    }
    gate_scores = {
        metric: _nested_scores_to_layer_tensors(nested)
        for metric, nested in payload.get("gate_scores", {}).items()
    }
    metadata = {
        "modality_aware": bool(payload.get("modality_aware", False)),
        "layers": payload.get("layers", sorted(next(iter(payload["expert_scores"].values())).keys())),
        "layerwise_loss": payload.get("layerwise_loss", {}),
    }
    return expert_scores, gate_scores, metadata, payload


def load_channel_scores(
    scores_dir: str,
    device: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Optional[Dict[str, Dict[int, torch.Tensor]]], Dict[str, Any], Optional[dict]]:
    if os.path.isfile(scores_dir):
        if verbose:
            _print(f"[Score Loading] Loading scores payload from {scores_dir}")
        expert_scores, gate_scores, metadata, payload = _load_scores_payload(scores_dir, device)
        return expert_scores, None, {"gate_scores": gate_scores, "metadata": metadata}, payload

    modern_fp = os.path.join(scores_dir, "scores.pt")
    legacy_fp = os.path.join(scores_dir, "channel_scores.pt")
    if os.path.exists(modern_fp):
        if verbose:
            _print(f"[Score Loading] Loading unified scores payload from {modern_fp}")
        expert_scores, gate_scores, metadata, payload = _load_scores_payload(modern_fp, device)
        return expert_scores, None, {"gate_scores": gate_scores, "metadata": metadata}, payload
    if os.path.exists(legacy_fp):
        if verbose:
            _print(f"[Score Loading] Falling back to legacy payload at {legacy_fp}")
        expert_scores, gate_scores, metadata, payload = _load_legacy_payload(legacy_fp, device)
        return expert_scores, None, {"gate_scores": gate_scores, "metadata": metadata}, payload

    raise FileNotFoundError(f"No scores.pt found under {scores_dir}")


def load_layerwise_loss(
    scores_dir: str,
    inter_layer_method: str,
    smooth_fn: str,
    device: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    m = {"layerwise_loss": None, "smooth_times": 0, "smooth_fn": smooth_fn}
    if "loss" not in inter_layer_method:
        return m

    scores_fp = scores_dir if os.path.isfile(scores_dir) else os.path.join(scores_dir, "scores.pt")
    if not os.path.exists(scores_fp):
        raise FileNotFoundError(f"scores.pt not found under {scores_dir}")
    payload = torch.load(scores_fp, map_location=device, weights_only=False)
    layerwise_loss_dict = payload.get("layerwise_loss", None)
    if layerwise_loss_dict is None:
        raise FileNotFoundError(f"layerwise_loss not found in {scores_fp}")
    layerwise_loss = torch.tensor(
        [layerwise_loss_dict[layer] for layer in sorted(layerwise_loss_dict.keys())],
        dtype=torch.float32,
        device=device,
    )
    if verbose:
        _print(f"[Score Loading] Loading layerwise_loss from {scores_fp}")
    m["layerwise_loss"] = layerwise_loss
    if inter_layer_method.startswith("loss_smooth_"):
        m["smooth_times"] = int(inter_layer_method.split("_")[-1])
    return m


def load_attention_head_scores(scores_dir: str, verbose: bool = True):
    if os.path.isfile(scores_dir):
        maybe_dir = os.path.dirname(scores_dir)
    else:
        maybe_dir = scores_dir
    fp = os.path.join(maybe_dir, "attn_head_scores.pth")
    if os.path.exists(fp):
        if verbose:
            _print(f"[Score Loading] Loading attention head scores from {fp}")
        return torch.load(fp, map_location="cpu")
    return None


def load_modality_channel_scores(
    scores_dir: str,
    device: str = "cpu",
    intra_expert_metric: str = "activation",
):
    expert_scores, _, aux, payload = load_channel_scores(scores_dir, device=device, verbose=False)
    gate_scores = aux.get("gate_scores", {})
    if intra_expert_metric == "second_order":
        text_scores = expert_scores.get("channel_second_order_text", None)
        visual_scores = expert_scores.get("channel_second_order_visual", None)
        if text_scores is None or visual_scores is None:
            raise ValueError(
                "Requested modality-aware second_order masks, but "
                "`channel_second_order_text/visual` were not found in scores payload."
            )
    else:
        text_scores = expert_scores.get("activation_text", None)
        visual_scores = expert_scores.get("activation_visual", None)
        if text_scores is None or visual_scores is None:
            if payload is not None and payload.get("modality_channel_scores") is not None:
                text_scores = _nested_scores_to_layer_tensors(payload["modality_channel_scores"]["text"])
                visual_scores = _nested_scores_to_layer_tensors(payload["modality_channel_scores"]["visual"])
            else:
                raise ValueError(f"modality-split scores are required, but not found in {scores_dir}")

    ema_nested = payload.get("ema_matrix") if payload is not None else None
    if ema_nested is None:
        usage_text = gate_scores.get("usage_text")
        usage_visual = gate_scores.get("usage_visual")
        if usage_text is not None and usage_visual is not None:
            ema_nested = {}
            for lid in sorted(usage_text.keys()):
                t = usage_text[lid].detach().cpu().float()
                v = usage_visual[lid].detach().cpu().float()
                ema_nested[lid] = (v - t) / (v + t + 1e-8)

    ema_tensor = None
    if ema_nested is not None:
        if isinstance(next(iter(ema_nested.values())), torch.Tensor):
            ema_tensor = dict_to_tensor(ema_nested).to(device=device, dtype=torch.float32)
        else:
            ema_tensor = dict_to_tensor(_nested_scores_to_layer_tensors(ema_nested)).to(
                device=device, dtype=torch.float32
            )

    return {
        "text": dict_to_tensor(text_scores).to(device=device, dtype=torch.float32),
        "visual": dict_to_tensor(visual_scores).to(device=device, dtype=torch.float32),
        "ema_matrix": ema_tensor,
    }


def prepare_scores(
    scores_dir: str,
    mask_method_kwargs: Dict[str, Any],
    prune_ratio: float,
    smooth_fn: str = "sqrt",
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, Dict[str, Any], list]:
    expert_scores, _, aux, _ = load_channel_scores(scores_dir, device, verbose)
    gate_scores = aux["gate_scores"]

    ####################################
    # source of channel ranking score  # 
    ####################################
    
    intra_expert_metric = mask_method_kwargs.get("intra_expert_metric", "activation")
    if intra_expert_metric == "second_order":
        text_scores = expert_scores.get("channel_second_order_text")
        visual_scores = expert_scores.get("channel_second_order_visual")
        if text_scores is None or visual_scores is None:
            raise KeyError(
                "Requested intra_expert_metric=second_order, but "
                "`channel_second_order_text/visual` are missing from scores payload."
            )
        intermediate_scores = (
            dict_to_tensor(text_scores).to(device=device, dtype=torch.float32)
            + dict_to_tensor(visual_scores).to(device=device, dtype=torch.float32)
        ) / 2.0
        
        _print("[prepare_scores] intermediate_scores is mean of text and visual scores")
    else:
        if intra_expert_metric not in expert_scores:
            raise KeyError(
                f"Requested intra_expert_metric={intra_expert_metric}, "
                f"available={sorted(expert_scores.keys())}"
            )
        intermediate_scores = dict_to_tensor(expert_scores[intra_expert_metric]).to(device=device, dtype=torch.float32)
    L, E, I = intermediate_scores.shape

    ####################################
    # source of expert-level scores    # 
    ####################################
    
    intra_layer_method = mask_method_kwargs.get("intra_layer_method", "uniform")
    # A. 根据 expert 输出来分配 expert budget
    expert_metric_by_method = {
        "attr_coverage": "first_attr_usage",
        "second_attr_coverage": "second_exact_attr",
        "true_ablate": "true_ablate",
        "true_ablate_coverage": "true_ablate",
    }
    # B. 根据 gate 输出来分配 expert budget
    gate_metric_by_method = {
        "usage": "usage",
        "usage_coverage": "usage",
        "router": "router",
        "router_coverage": "router",
    }
    if intra_layer_method in expert_metric_by_method:
        metric_name = expert_metric_by_method[intra_layer_method]
        if metric_name not in expert_scores:
            raise KeyError(
                f"{metric_name} is required for intra_layer_method={intra_layer_method}."
            )
        expertwise_scores = dict_to_tensor(expert_scores[metric_name]).to(
            device=device, dtype=torch.float32
        )
    elif intra_layer_method in gate_metric_by_method:
        metric_name = gate_metric_by_method[intra_layer_method]
        if metric_name not in gate_scores:
            raise KeyError(
                f"gate_scores['{metric_name}'] is required for intra_layer_method={intra_layer_method}."
            )
        expertwise_scores = dict_to_tensor(gate_scores[metric_name]).to(device=device, dtype=torch.float32)
    elif intra_layer_method == "uniform":
        expertwise_scores = torch.ones((L, E), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Invalid intra_layer_method: {intra_layer_method}")

    inter_layer_method = mask_method_kwargs.get("inter_layer_method", "uniform")
    loss_based_kwargs = load_layerwise_loss(scores_dir, inter_layer_method, smooth_fn, device, verbose)
    loss_based_kwargs["inter_layer_method"] = inter_layer_method
    layers = aux["metadata"].get("layers")
    if layers is None: layers = list(range(L))
    
    return (
        intermediate_scores,
        expertwise_scores,
        L,
        E,
        I,
        loss_based_kwargs,
        layers,
    )
