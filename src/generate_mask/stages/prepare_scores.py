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
    metadata = {
        "modality_aware": bool(payload.get("modality_aware", False)),
        "layers": sorted(payload["scores"].keys()),
    }
    return expert_scores, gate_scores, metadata, payload


def _load_modern_scores(scores_dir: str, device: str):
    expert_scores = torch.load(os.path.join(scores_dir, "expert_scores.pth"), map_location=device)
    gate_scores_path = os.path.join(scores_dir, "gate_scores.pth")
    gate_scores = torch.load(gate_scores_path, map_location=device) if os.path.exists(gate_scores_path) else {}
    metadata_path = os.path.join(scores_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return expert_scores, gate_scores, metadata, None


def load_channel_scores(
    scores_dir: str,
    prune_hidden: bool,
    device: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Optional[Dict[str, Dict[int, torch.Tensor]]], Dict[str, Any], Optional[dict]]:
    if os.path.isfile(scores_dir):
        if verbose:
            _print(f"[Score Loading] Loading legacy payload from {scores_dir}")
        expert_scores, gate_scores, metadata, payload = _load_legacy_payload(scores_dir, device)
        return expert_scores, None, {"gate_scores": gate_scores, "metadata": metadata}, payload

    modern_fp = os.path.join(scores_dir, "expert_scores.pth")
    legacy_fp = os.path.join(scores_dir, "channel_scores.pt")
    if not os.path.exists(modern_fp) and os.path.exists(legacy_fp):
        if verbose:
            _print(f"[Score Loading] Falling back to legacy payload at {legacy_fp}")
        expert_scores, gate_scores, metadata, payload = _load_legacy_payload(legacy_fp, device)
        return expert_scores, None, {"gate_scores": gate_scores, "metadata": metadata}, payload

    if verbose:
        _print(f"[Score Loading] Loading score directory from {scores_dir}")
    expert_scores, gate_scores, metadata, payload = _load_modern_scores(scores_dir, device)
    return expert_scores, None, {"gate_scores": gate_scores, "metadata": metadata}, payload


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

    if os.path.isfile(scores_dir):
        maybe_dir = os.path.dirname(scores_dir)
    else:
        maybe_dir = scores_dir
    fp = os.path.join(maybe_dir, "layerwise_loss.pth")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"layerwise_loss.pth not found under {maybe_dir}")
    layerwise_loss = torch.load(fp, map_location=device)
    if verbose:
        _print(f"[Score Loading] Loading layerwise_loss from {fp}")
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


def load_modality_channel_scores(scores_dir: str, device: str = "cpu"):
    expert_scores, _, aux, payload = load_channel_scores(scores_dir, prune_hidden=False, device=device, verbose=False)
    metadata = aux["metadata"]
    modality_aware = bool(metadata.get("modality_aware", False))
    if not modality_aware:
        return None
    text_scores = expert_scores.get("activation_text")
    visual_scores = expert_scores.get("activation_visual")
    if text_scores is None or visual_scores is None:
        if payload is not None and payload.get("modality_channel_scores") is not None:
            text_scores = {
                lid: dict_to_tensor(payload["modality_channel_scores"]["text"])[idx]
                for idx, lid in enumerate(sorted(payload["modality_channel_scores"]["text"].keys()))
            }
            visual_scores = {
                lid: dict_to_tensor(payload["modality_channel_scores"]["visual"])[idx]
                for idx, lid in enumerate(sorted(payload["modality_channel_scores"]["visual"].keys()))
            }
        else:
            return None
    return {
        "text": dict_to_tensor(text_scores).to(device=device, dtype=torch.float32),
        "visual": dict_to_tensor(visual_scores).to(device=device, dtype=torch.float32),
    }


def prepare_scores(
    scores_dir: str,
    mask_method_kwargs: Dict[str, Any],
    HI_ratio_kwargs: Dict[str, Any],
    prune_ratio: float,
    prune_hidden: bool,
    prune_gqa: bool,
    smooth_fn: str = "sqrt",
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, Dict[str, Any]]:
    expert_scores, _, aux, _ = load_channel_scores(scores_dir, prune_hidden, device, verbose)
    gate_scores = aux["gate_scores"]

    intra_expert_metric = mask_method_kwargs.get("intra_expert_metric", "activation")
    if intra_expert_metric not in expert_scores:
        raise KeyError(
            f"Requested intra_expert_metric={intra_expert_metric}, "
            f"available={sorted(expert_scores.keys())}"
        )

    intermediate_scores = dict_to_tensor(expert_scores[intra_expert_metric]).to(device=device, dtype=torch.float32)
    L, E, I = intermediate_scores.shape
    hidden_scores = None
    H = None

    intra_layer_method = mask_method_kwargs.get("intra_layer_method", "uniform")
    if intra_layer_method in ("attr_coverage", "loss_coverage"):
        if "expert_out_token_contrib" not in expert_scores:
            raise KeyError("expert_out_token_contrib is required for attr_coverage.")
        expertwise_scores = dict_to_tensor(expert_scores["expert_out_token_contrib"]).to(device=device, dtype=torch.float32)
    elif "usage" in intra_layer_method:
        if "usage" not in gate_scores:
            raise KeyError("gate_scores['usage'] is required for usage-based planning.")
        expertwise_scores = dict_to_tensor(gate_scores["usage"]).to(device=device, dtype=torch.float32)
    else:
        expertwise_scores = torch.ones((L, E), dtype=torch.float32, device=device)

    inter_layer_method = mask_method_kwargs.get("inter_layer_method", "uniform")
    loss_based_kwargs = load_layerwise_loss(scores_dir, inter_layer_method, smooth_fn, device, verbose)
    layers = aux["metadata"].get("layers")
    if layers is None:
        layers = list(range(L))
    layerwise_keep_plan = inter_layer_planner(
        intermediate_scores,
        p_target=prune_ratio,
        method=inter_layer_method,
        L=L,
        loss_based_importance_kwargs=loss_based_kwargs,
        tol=0.1,
        verbose=verbose,
    )
    loss_based_kwargs["layerwise_keep_plan"] = layerwise_keep_plan
    loss_based_kwargs["layers"] = layers

    return (
        intermediate_scores,
        hidden_scores,
        expertwise_scores,
        L,
        E,
        I,
        H,
        loss_based_kwargs,
    )
