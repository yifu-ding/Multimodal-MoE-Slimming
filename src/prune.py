"""Structural channel pruning for Kimi-VL MoE experts."""

from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# MoE layer predicate
# ---------------------------------------------------------------------------

def _is_moe_layer(layer_idx: int, config) -> bool:
    return (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


def _shrink_kimi_router_for_active_experts(module: nn.Module, keep_mask: torch.Tensor) -> int:
    keep_mask = keep_mask.to(dtype=torch.bool)
    gate = module.gate
    old_num_experts = int(keep_mask.numel())
    n_active = int(keep_mask.sum().item())
    if n_active == 0:
        raise RuntimeError("All experts in this layer were fully pruned.")

    keep_idx = torch.nonzero(keep_mask.to(gate.weight.device), as_tuple=False).view(-1)
    gate.weight = nn.Parameter(gate.weight.data.index_select(0, keep_idx).contiguous())

    if hasattr(gate, "e_score_correction_bias") and gate.e_score_correction_bias is not None:
        gate.e_score_correction_bias = nn.Parameter(
            gate.e_score_correction_bias.data.index_select(0, keep_idx).contiguous()
        )

    gate.n_routed_experts = n_active
    if hasattr(module, "num_experts_per_tok"):
        module.num_experts_per_tok = min(int(module.num_experts_per_tok), n_active)
    if hasattr(gate, "top_k"):
        gate.top_k = min(int(gate.top_k), n_active)
    if hasattr(module, "experts_per_rank"):
        module.experts_per_rank = len(module.experts)
    if hasattr(gate, "experts_len"):
        gate.experts_len = len(module.experts)

    # noaux_tc requires a valid grouped layout; if the pruned expert count no longer
    # fits the original grouping assumptions, fall back to greedy top-k.
    if getattr(gate, "topk_method", None) == "noaux_tc":
        n_group = int(getattr(gate, "n_group", 1))
        per_group = n_active // max(n_group, 1) if n_group > 0 else 0
        topk_group = int(getattr(gate, "topk_group", 1))
        if (
            n_group <= 0
            or n_active % n_group != 0
            or per_group < 2
            or topk_group > n_group
        ):
            gate.topk_method = "greedy"

    return old_num_experts - n_active


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
    """
    layers = model.language_model.model.layers
    pbar = tqdm(total=len(layers), desc="Pruning experts", unit="layer")

    params_removed = 0
    params_kept = 0
    inactive_experts = 0
    shrink_gate_cnt = 0

    for layer_idx, layer in enumerate(layers):
        pbar.update(1)
        if not _is_moe_layer(layer_idx, config):
            continue
        if layer_idx not in masks:
            continue

        layer_mask = masks[layer_idx]  # [E, I]
        old_num_experts = len(layer.mlp.experts)
        if layer_mask.shape[0] != old_num_experts:
            raise RuntimeError(
                f"Layer {layer_idx}: mask expert dim={int(layer_mask.shape[0])} "
                f"but model has {old_num_experts} experts."
            )
        layer_active_expert = torch.ones(old_num_experts, dtype=torch.bool)

        for eid, expert in enumerate(layer.mlp.experts):
            m_inter = layer_mask[eid].to(
                device=expert.gate_proj.weight.device, dtype=torch.bool
            )  # [I]
            I_prime = int(m_inter.sum().item())
            if I_prime == 0:
                layer_active_expert[eid] = False
                I_old = expert.gate_proj.out_features
                H = expert.gate_proj.in_features
                params_removed += int(I_old * H * 3)
                continue

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

        n_active = int(layer_active_expert.sum().item())
        if n_active == 0:
            raise RuntimeError(
                f"All experts in layer {layer_idx} were fully pruned. "
                "Adjust masks to keep at least one expert."
            )
        inactive_experts += old_num_experts - n_active
        if n_active != old_num_experts:
            keep_eids = torch.nonzero(layer_active_expert, as_tuple=False).view(-1).tolist()
            layer.mlp.experts = nn.ModuleList([layer.mlp.experts[eid] for eid in keep_eids])
            shrink_gate_cnt += _shrink_kimi_router_for_active_experts(layer.mlp, layer_active_expert)

    pbar.close()
    total = params_removed + params_kept
    pct = 100.0 * params_removed / total if total > 0 else 0.0
    print(
        f"[prune] Expert params removed: {params_removed:,}  "
        f"kept: {params_kept:,}  "
        f"({pct:.1f}% removed)"
    )
    if shrink_gate_cnt > 0:
        print(
            f"[prune] Removed {inactive_experts} fully pruned experts and shrank "
            f"{shrink_gate_cnt} gate entries."
        )
