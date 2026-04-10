"""Phase 2: Generate keep-masks and apply structural channel pruning to Kimi-VL.

Loads pre-computed channel scores (from collect_scores.py), generates per-expert
top-k keep-masks, applies structural pruning in-place (resizing gate_proj / up_proj /
down_proj), updates the model config, and saves with save_pretrained.

The pruned model is a smaller, fully valid Kimi-VL model that can be loaded with
AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True).

Usage
-----
    python src/prune.py \\
        --model_path moonshotai/Kimi-VL-A3B-Instruct \\
        --scores_path storage/prune/scores/kimi_gqa/channel_scores.pt \\
        --prune_ratio 0.30 \\
        --output_dir storage/prune/pruned_models/kimi_gqa_p30

Pruning scope
-------------
- Only routed experts (layer.mlp.experts[eid]) are pruned.
- Shared experts (layer.mlp.shared_experts) are left untouched.
- Layer 0 is dense (no MoE); only layers 1–26 are pruned.
- Hidden-dimension (H) pruning is NOT applied; only intermediate (I) is pruned.
"""

import argparse
import math
import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from observations.common import resolve_model_name_or_path


# ---------------------------------------------------------------------------
# MoE layer predicate
# ---------------------------------------------------------------------------

def _is_moe_layer(layer_idx: int, config) -> bool:
    return (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_masks(
    scores: Dict[int, Dict[int, torch.Tensor]],
    prune_ratio: float,
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
) -> Dict[int, torch.Tensor]:
    """Generate per-expert top-k keep-masks.

    For each (layer, expert) independently, keep the top-k channels by score
    where k = max(1, round(I * (1 - prune_ratio))).

    Returns
    -------
    masks : Dict[int, Tensor[E, I]] bool
        True  = keep this channel
        False = prune this channel
    """
    masks: Dict[int, torch.Tensor] = {}
    for layer_idx in sorted(scores.keys()):
        E = layer_to_num_experts[layer_idx]
        I = layer_to_num_channels[layer_idx]
        k = max(1, round(I * (1.0 - prune_ratio)))
        layer_mask = torch.zeros(E, I, dtype=torch.bool)
        for eid in range(E):
            s = scores[layer_idx].get(eid, None)
            if s is None or s.numel() == 0:
                # No score available — keep first k channels (no-op for this expert)
                layer_mask[eid, :k] = True
            else:
                s = s.float().cpu()
                if s.numel() != I:
                    raise ValueError(
                        f"Layer {layer_idx} expert {eid}: score length {s.numel()} != I={I}"
                    )
                topk_idx = torch.topk(s, k, largest=True).indices
                layer_mask[eid][topk_idx] = True
        masks[layer_idx] = layer_mask
    return masks


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_structural_pruning(
    model: nn.Module,
    masks: Dict[int, torch.Tensor],
    config,
) -> None:
    """Structurally prune routed expert intermediate dimensions in-place.

    For each routed expert at every MoE layer:
        gate_proj : [I, H] -> [I', H]   (keep rows where mask=True)
        up_proj   : [I, H] -> [I', H]   (same rows)
        down_proj : [H, I] -> [H, I']   (keep cols where mask=True)

    Shared experts (layer.mlp.shared_experts) are never touched.

    Ported and simplified from:
        LLM-Distillation/src/prune/apply/slimming/expert_slim.py
        slim_moe_inter_and_optional_hidden_keepmask_inplace (lines 252-370)
    """
    layers = model.language_model.model.layers
    pbar = tqdm(total=len(layers), desc="Pruning experts", unit="layer")

    params_removed = 0
    params_kept = 0

    for layer_idx, layer in enumerate(layers):
        pbar.update(1)
        if not _is_moe_layer(layer_idx, config):
            continue
        if layer_idx not in masks:
            continue

        layer_mask = masks[layer_idx]  # [E, I]

        for eid, expert in enumerate(layer.mlp.experts):
            m_inter = layer_mask[eid].to(
                device=expert.gate_proj.weight.device, dtype=torch.bool
            )  # [I]
            I_prime = int(m_inter.sum().item())
            if I_prime == 0:
                # Degenerate: keep at least one channel to avoid zero-dim layers
                m_inter[0] = True
                I_prime = 1

            dtype = expert.gate_proj.weight.dtype
            device = expert.gate_proj.weight.device
            H = expert.gate_proj.in_features

            W_gate = expert.gate_proj.weight.data[m_inter, :]  # [I', H]
            W_up   = expert.up_proj.weight.data[m_inter, :]    # [I', H]
            W_down = expert.down_proj.weight.data[:, m_inter]  # [H, I']

            I_old = expert.gate_proj.out_features
            params_removed += int((I_old - I_prime) * H * 2 + H * (I_old - I_prime))
            params_kept    += int(I_prime * H * 2 + H * I_prime)

            new_gate = nn.Linear(H, I_prime, bias=False, device=device, dtype=dtype)
            new_up   = nn.Linear(H, I_prime, bias=False, device=device, dtype=dtype)
            new_down = nn.Linear(I_prime, H, bias=False, device=device, dtype=dtype)

            new_gate.weight = nn.Parameter(W_gate.contiguous())
            new_up.weight   = nn.Parameter(W_up.contiguous())
            new_down.weight = nn.Parameter(W_down.contiguous())

            expert.gate_proj = new_gate
            expert.up_proj   = new_up
            expert.down_proj = new_down

    pbar.close()
    total = params_removed + params_kept
    pct = 100.0 * params_removed / total if total > 0 else 0.0
    print(
        f"[prune] Expert params removed: {params_removed:,}  "
        f"kept: {params_kept:,}  "
        f"({pct:.1f}% removed)"
    )


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def update_config(model: nn.Module, masks: Dict[int, torch.Tensor]) -> int:
    """Update moe_intermediate_size in the text config to reflect the new I'.

    Assumes uniform pruning (all experts have the same I' across all layers).
    Returns I'.
    """
    # Sample I' from the first MoE layer's first expert
    first_layer = sorted(masks.keys())[0]
    I_prime = int(masks[first_layer][0].sum().item())
    model.config.text_config.moe_intermediate_size = I_prime
    print(f"[prune] Updated config: moe_intermediate_size = {I_prime}")
    return I_prime


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structural channel pruning of Kimi-VL MoE experts."
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HF model id or local path to the original (un-pruned) model.",
    )
    p.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to channel_scores.pt produced by collect_scores.py.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--prune_ratio",
        type=float,
        default=0.30,
        help="Fraction of intermediate channels to remove (0.0 = no pruning, 0.5 = half).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load scores ----
    print(f"[prune] Loading scores from {args.scores_path}")
    score_payload = torch.load(args.scores_path, weights_only=False)
    scores           = score_payload["scores"]
    layer_to_num_experts  = score_payload["layer_to_num_experts"]
    layer_to_num_channels = score_payload["layer_to_num_channels"]
    print(
        f"[prune] Scores: {len(scores)} MoE layers, "
        f"score_type={score_payload.get('score_type', 'unknown')}"
    )

    # ---- Generate masks ----
    print(f"[prune] Generating masks with prune_ratio={args.prune_ratio}")
    masks = generate_masks(scores, args.prune_ratio, layer_to_num_experts, layer_to_num_channels)
    first_layer = sorted(masks.keys())[0]
    I_orig = layer_to_num_channels[first_layer]
    I_prime = int(masks[first_layer][0].sum().item())
    print(f"[prune] I: {I_orig} -> {I_prime}  (keeping {I_prime}/{I_orig} channels)")

    # ---- Load clean model (no monkey-patches) ----
    model_path = resolve_model_name_or_path(args.model_path)
    print(f"[prune] Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    config = model.config.text_config

    # ---- Apply structural pruning ----
    apply_structural_pruning(model, masks, config)

    # ---- Update config ----
    update_config(model, masks)

    # ---- Save ----
    print(f"[prune] Saving pruned model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[prune] Done.  Pruned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
