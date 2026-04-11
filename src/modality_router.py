"""Modality-aware MoE routing for Kimi-VL inference.

When a model was scored with --modality_aware in collect_scores.py, per-expert
affinity values are saved to ``affinity.pt`` alongside the channel scores.
This module provides helpers to load those values and attach them to the model
so that at inference time the gate blocks mismatched-modality tokens from
activating strongly-specialised experts.

Mechanism
---------
For expert e in MoE layer l, with affinity a = (visual_freq - text_freq) /
(visual_freq + text_freq + eps) ∈ [-1, +1]:

  a > +threshold  (visual-preferring) → text   tokens get score → 0 for e
  a < -threshold  (text-preferring)   → visual tokens get score → 0 for e

The score is zeroed in ``models/kimi.py:gate_forward`` on ``tmp_scores``
(noaux_tc method) or ``scores`` (greedy method), which are the selection
tensors used to pick top-k experts.  This prevents the expert from being
selected for mismatched tokens, routing them to other experts instead.

Usage
-----
    from src.modality_router import load_affinity, attach_modality_aware_router

    affinity = load_affinity("storage/prune/scores/kimi_gqa/affinity.pt")
    attach_modality_aware_router(model, affinity, threshold=0.9)
    # run inference normally
"""

import torch
from typing import Dict, Optional


def load_affinity(path: str) -> Dict[int, Dict[int, float]]:
    """Load affinity dict from a file saved by collect_scores.py.

    Returns Dict[layer_idx, Dict[expert_idx, float]] with values in [-1, +1].
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["affinity"]


def attach_modality_aware_router(
    model,
    affinity: Dict[int, Dict[int, float]],
    threshold: float = 0.9,
) -> None:
    """Attach affinity-aware routing masks to every applicable MoE gate.

    Sets ``gate.affinity_mask`` on each layer whose gate has at least one
    expert with |affinity| > threshold.  The gate_forward in models/kimi.py
    reads this attribute and zeroes out selection scores for blocked
    (token, expert) pairs before top-k routing.

    Also triggers the model_forward block that populates
    ``gate.moe_text_index`` / ``gate.moe_media_index`` per forward pass
    (the same block used by the observation / skip-expert paths).

    Parameters
    ----------
    model : loaded Kimi-VL model (AutoModelForCausalLM or similar)
    affinity : Dict[layer_idx, Dict[expert_idx, float]]
        Per-expert affinity values, typically from load_affinity().
    threshold : float
        Magnitude cutoff; default 0.9.
    """
    patched = 0
    total_vis, total_txt = 0, 0

    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if layer_idx not in affinity:
            continue
        aff_layer = affinity[layer_idx]

        vis_only = [e for e, a in aff_layer.items() if a > threshold]
        txt_only = [e for e, a in aff_layer.items() if a < -threshold]

        if not vis_only and not txt_only:
            continue

        gate = layer.mlp.gate
        gate.affinity_mask = {
            "visual_only": torch.tensor(vis_only, dtype=torch.long) if vis_only else None,
            "text_only":   torch.tensor(txt_only, dtype=torch.long) if txt_only else None,
        }
        patched += 1
        total_vis += len(vis_only)
        total_txt += len(txt_only)

    print(
        f"[modality_router] Attached affinity routing to {patched} MoE layers "
        f"(threshold={threshold}): "
        f"{total_vis} visual-only experts, {total_txt} text-only experts."
    )


def detach_modality_aware_router(model) -> None:
    """Remove affinity masks from all gates (restore normal routing)."""
    for layer in model.language_model.model.layers:
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is not None and hasattr(gate, "affinity_mask"):
            gate.affinity_mask = None
