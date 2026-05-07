"""Structural channel pruning for Kimi-VL, Qwen3-VL-MoE, and InternVL/GPT-OSS MoE experts."""

from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# MoE layer predicate / helpers
# ---------------------------------------------------------------------------

def _is_kimi_moe_layer(layer_idx: int, config) -> bool:
    return (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


def _is_gpt_oss_moe_layer(layer: nn.Module, config) -> bool:
    return (
        getattr(config, "num_local_experts", 0) > 0
        and hasattr(layer, "mlp")
        and hasattr(layer.mlp, "router")
        and hasattr(layer.mlp, "experts")
    )


def _is_qwen3_moe_layer(layer_idx: int, layer: nn.Module, config) -> bool:
    return (
        layer_idx not in getattr(config, "mlp_only_layers", [])
        and getattr(config, "num_experts", 0) > 0
        and (layer_idx + 1) % int(getattr(config, "decoder_sparse_step", 1)) == 0
        and hasattr(layer, "mlp")
        and hasattr(layer.mlp, "gate")
        and hasattr(layer.mlp, "experts")
    )


def _num_experts_of(experts: nn.Module) -> int:
    num_experts = getattr(experts, "num_experts", None)
    if num_experts is not None:
        return int(num_experts)
    try:
        return int(len(experts))
    except TypeError as exc:
        raise TypeError(
            f"Cannot infer number of experts from container type {type(experts).__name__}."
        ) from exc


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


def _shrink_gpt_oss_router_for_active_experts(module: nn.Module, keep_mask: torch.Tensor) -> int:
    keep_mask = keep_mask.to(dtype=torch.bool)
    router = module.router
    old_num_experts = int(keep_mask.numel())
    n_active = int(keep_mask.sum().item())
    if n_active == 0:
        raise RuntimeError("All experts in this layer were fully pruned.")

    keep_idx = torch.nonzero(keep_mask.to(router.weight.device), as_tuple=False).view(-1)
    router.weight = nn.Parameter(
        router.weight.data.index_select(0, keep_idx).contiguous()
    )
    router.bias = nn.Parameter(
        router.bias.data.index_select(0, keep_idx).contiguous()
    )
    router.num_experts = n_active
    router.top_k = min(int(router.top_k), n_active)
    return old_num_experts - n_active


def _shrink_qwen3_router_for_active_experts(module: nn.Module, keep_mask: torch.Tensor) -> int:
    keep_mask = keep_mask.to(dtype=torch.bool)
    gate = module.gate
    old_num_experts = int(keep_mask.numel())
    n_active = int(keep_mask.sum().item())
    if n_active == 0:
        raise RuntimeError("All experts in this layer were fully pruned.")

    keep_idx = torch.nonzero(keep_mask.to(gate.weight.device), as_tuple=False).view(-1)
    gate.weight = nn.Parameter(gate.weight.data.index_select(0, keep_idx).contiguous())
    module.num_experts = n_active
    gate.out_features = n_active
    if hasattr(gate, "num_experts"):
        gate.num_experts = n_active
    module.top_k = min(int(module.top_k), n_active)
    return old_num_experts - n_active


class PrunedGptOssExpert(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        alpha: float,
        limit: float,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.alpha = float(alpha)
        self.limit = float(limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return self.down_proj((up + 1) * glu)


class PrunedGptOssExperts(nn.Module):
    def __init__(self, experts: list[PrunedGptOssExpert], hidden_size: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.hidden_size = int(hidden_size)

    def __len__(self) -> int:
        return len(self.experts)

    def __iter__(self):
        return iter(self.experts)

    def __getitem__(self, idx: int) -> PrunedGptOssExpert:
        return self.experts[idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices=None,
        routing_weights=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        next_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=self.num_experts + 1
        ).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_tensor in expert_hit:
            expert_idx = int(expert_tensor[0].item())
            if expert_idx == self.num_experts:
                continue
            _, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            out = self.experts[expert_idx](current_state)
            weighted = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(0, token_idx, weighted.to(next_states.dtype))
        return next_states.view(batch_size, -1, self.hidden_size)


class PrunedQwen3Expert(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        act_fn,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PrunedQwen3Experts(nn.Module):
    def __init__(self, experts: list[PrunedQwen3Expert], hidden_size: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.hidden_size = int(hidden_size)

    def __len__(self) -> int:
        return len(self.experts)

    def __iter__(self):
        return iter(self.experts)

    def __getitem__(self, idx: int) -> PrunedQwen3Expert:
        return self.experts[idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        next_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=self.num_experts
        ).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_tensor in expert_hit:
            expert_idx = int(expert_tensor[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            out = self.experts[expert_idx](hidden_states[token_idx])
            weighted = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted.to(next_states.dtype))
        return next_states


def _make_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    in_features: int,
    out_features: int,
) -> nn.Linear:
    layer = nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    layer.weight = nn.Parameter(weight.contiguous())
    if bias is not None:
        layer.bias = nn.Parameter(bias.contiguous())
    return layer


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------

def _resolve_model_layout(model: nn.Module, config):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return "kimi", model.language_model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        if getattr(config, "num_experts", 0) > 0:
            return "qwen3", model.language_model.layers
        if getattr(config, "num_local_experts", 0) > 0:
            return "gpt_oss", model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if getattr(config, "num_experts", 0) > 0:
            return "qwen3", model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if getattr(config, "num_local_experts", 0) > 0:
            return "gpt_oss", model.model.language_model.layers
    raise NotImplementedError(
        f"Unsupported model layout for structural pruning: {type(model)}"
    )

@torch.no_grad()
def apply_structural_pruning(
    model: nn.Module,
    masks: Dict[int, torch.Tensor] | None,
    config,
) -> None:
    """Structurally prune routed expert intermediate dimensions in-place.

    For each routed expert at every MoE layer:
        gate_proj : [I, H] -> [I', H]   (keep rows where mask=True)
        up_proj   : [I, H] -> [I', H]   (same rows)
        down_proj : [H, I] -> [H, I']   (keep cols where mask=True)

    Shared experts (layer.mlp.shared_experts) are never touched.
    """
    if masks is None:
        return
    model_layout, layers = _resolve_model_layout(model, config)
    pbar = tqdm(total=len(layers), desc="Pruning experts", unit="layer")

    params_removed = 0
    params_kept = 0
    inactive_experts = 0
    shrink_gate_cnt = 0

    for layer_idx, layer in enumerate(layers):
        pbar.update(1)
        if model_layout == "kimi":
            is_moe_layer = _is_kimi_moe_layer(layer_idx, config)
        elif model_layout == "qwen3":
            is_moe_layer = _is_qwen3_moe_layer(layer_idx, layer, config)
        else:
            is_moe_layer = _is_gpt_oss_moe_layer(layer, config)
        if not is_moe_layer:
            continue
        if layer_idx not in masks:
            continue

        layer_mask = masks[layer_idx]  # [E, I]
        if model_layout == "kimi":
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
                )
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

                W_gate = expert.gate_proj.weight.data[m_inter, :]
                W_up = expert.up_proj.weight.data[m_inter, :]
                W_down = expert.down_proj.weight.data[:, m_inter]

                I_old = expert.gate_proj.out_features
                params_removed += int((I_old - I_prime) * H * 3)
                params_kept += int(I_prime * H * 3)

                new_gate = nn.Linear(H, I_prime, bias=False, device=device, dtype=dtype)
                new_up = nn.Linear(H, I_prime, bias=False, device=device, dtype=dtype)
                new_down = nn.Linear(I_prime, H, bias=False, device=device, dtype=dtype)

                new_gate.weight = nn.Parameter(W_gate.contiguous())
                new_up.weight = nn.Parameter(W_up.contiguous())
                new_down.weight = nn.Parameter(W_down.contiguous())

                expert.gate_proj = new_gate
                expert.up_proj = new_up
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
                shrink_gate_cnt += _shrink_kimi_router_for_active_experts(
                    layer.mlp, layer_active_expert
                )
            continue

        if model_layout == "qwen3":
            experts = layer.mlp.experts
            old_num_experts = _num_experts_of(experts)
            if layer_mask.shape[0] != old_num_experts:
                raise RuntimeError(
                    f"Layer {layer_idx}: mask expert dim={int(layer_mask.shape[0])} "
                    f"but model has {old_num_experts} experts."
                )

            new_experts = []
            layer_active_expert = torch.zeros(old_num_experts, dtype=torch.bool)
            if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
                for eid in range(old_num_experts):
                    m_inter = layer_mask[eid].to(device=experts.gate_up_proj.device, dtype=torch.bool)
                    I_old = int(experts.down_proj.shape[1])
                    H = int(experts.gate_up_proj.shape[1])
                    I_prime = int(m_inter.sum().item())
                    if I_prime == 0:
                        params_removed += int(I_old * H * 3)
                        continue

                    layer_active_expert[eid] = True
                    keep_idx = torch.nonzero(m_inter, as_tuple=False).view(-1)
                    gate_up_w = experts.gate_up_proj.data[eid]
                    down_w = experts.down_proj.data[eid][m_inter, :]

                    gate_w = gate_up_w[:, keep_idx].transpose(0, 1).contiguous()
                    up_w = gate_up_w[:, keep_idx + I_old].transpose(0, 1).contiguous()
                    down_w = down_w.transpose(0, 1).contiguous()

                    params_removed += int((I_old - I_prime) * H * 3)
                    params_kept += int(I_prime * H * 3)

                    new_experts.append(
                        PrunedQwen3Expert(
                            gate_proj=_make_linear(gate_w, None, H, I_prime),
                            up_proj=_make_linear(up_w, None, H, I_prime),
                            down_proj=_make_linear(down_w, None, I_prime, H),
                            act_fn=experts.act_fn,
                        )
                    )
            else:
                for eid, expert in enumerate(experts):
                    m_inter = layer_mask[eid].to(
                        device=expert.gate_proj.weight.device, dtype=torch.bool
                    )
                    I_prime = int(m_inter.sum().item())
                    if I_prime == 0:
                        layer_active_expert[eid] = False
                        I_old = expert.gate_proj.out_features
                        H = expert.gate_proj.in_features
                        params_removed += int(I_old * H * 3)
                        continue

                    layer_active_expert[eid] = True
                    dtype = expert.gate_proj.weight.dtype
                    device = expert.gate_proj.weight.device
                    H = expert.gate_proj.in_features
                    I_old = expert.gate_proj.out_features

                    gate_w = expert.gate_proj.weight.data[m_inter, :]
                    up_w = expert.up_proj.weight.data[m_inter, :]
                    down_w = expert.down_proj.weight.data[:, m_inter]

                    params_removed += int((I_old - I_prime) * H * 3)
                    params_kept += int(I_prime * H * 3)

                    new_experts.append(
                        PrunedQwen3Expert(
                            gate_proj=_make_linear(gate_w.contiguous(), None, H, I_prime),
                            up_proj=_make_linear(up_w.contiguous(), None, H, I_prime),
                            down_proj=_make_linear(down_w.contiguous(), None, I_prime, H),
                            act_fn=expert.act_fn,
                        )
                    )

            n_active = int(layer_active_expert.sum().item())
            if n_active == 0:
                raise RuntimeError(
                    f"All experts in layer {layer_idx} were fully pruned. "
                    "Adjust masks to keep at least one expert."
                )
            inactive_experts += old_num_experts - n_active
            layer.mlp.experts = PrunedQwen3Experts(new_experts, hidden_size=H)
            if n_active != old_num_experts:
                shrink_gate_cnt += _shrink_qwen3_router_for_active_experts(
                    layer.mlp, layer_active_expert
                )
            continue

        experts = layer.mlp.experts
        old_num_experts = int(experts.num_experts)
        if layer_mask.shape[0] != old_num_experts:
            raise RuntimeError(
                f"Layer {layer_idx}: mask expert dim={int(layer_mask.shape[0])} "
                f"but model has {old_num_experts} experts."
            )

        new_experts = []
        layer_active_expert = torch.zeros(old_num_experts, dtype=torch.bool)
        for eid in range(old_num_experts):
            m_inter = layer_mask[eid].to(device=experts.gate_up_proj.device, dtype=torch.bool)
            I_old = int(experts.down_proj.shape[1])
            H = int(experts.gate_up_proj.shape[1])
            I_prime = int(m_inter.sum().item())
            if I_prime == 0:
                params_removed += int(I_old * H * 3 + 2 * I_old + H)
                continue

            layer_active_expert[eid] = True
            keep_idx = torch.nonzero(m_inter, as_tuple=False).view(-1)
            pair_idx = torch.stack((keep_idx * 2, keep_idx * 2 + 1), dim=1).reshape(-1)

            gate_up_w = experts.gate_up_proj.data[eid][:, pair_idx]
            gate_up_b = experts.gate_up_proj_bias.data[eid][pair_idx]
            down_w = experts.down_proj.data[eid][m_inter, :]
            down_b = experts.down_proj_bias.data[eid]

            gate_w = gate_up_w[:, ::2].transpose(0, 1).contiguous()
            up_w = gate_up_w[:, 1::2].transpose(0, 1).contiguous()
            gate_b = gate_up_b[::2].contiguous()
            up_b = gate_up_b[1::2].contiguous()
            down_w = down_w.transpose(0, 1).contiguous()

            params_removed += int((I_old - I_prime) * H * 3 + 2 * (I_old - I_prime))
            params_kept += int(I_prime * H * 3 + 2 * I_prime + H)

            new_experts.append(
                PrunedGptOssExpert(
                    gate_proj=_make_linear(gate_w, gate_b, H, I_prime),
                    up_proj=_make_linear(up_w, up_b, H, I_prime),
                    down_proj=_make_linear(down_w, down_b, I_prime, H),
                    alpha=float(experts.alpha),
                    limit=float(experts.limit),
                )
            )

        n_active = int(layer_active_expert.sum().item())
        if n_active == 0:
            raise RuntimeError(
                f"All experts in layer {layer_idx} were fully pruned. "
                "Adjust masks to keep at least one expert."
            )
        inactive_experts += old_num_experts - n_active
        layer.mlp.experts = PrunedGptOssExperts(new_experts, hidden_size=H)
        if n_active != old_num_experts:
            shrink_gate_cnt += _shrink_gpt_oss_router_for_active_experts(
                layer.mlp, layer_active_expert
            )

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
