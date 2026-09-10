import numbers

import torch
import torch.nn as nn
from src.calibration.helpers.utils import (
    clear_fused_saved_tensors,
    is_fused_expert_container,
)

def weight_rms(weight: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
    # weight shape [..., I, ...], take L2 norm over all dims except channel_dim.
    x = weight.detach()
    reduce_dims = [d for d in range(x.ndim) if d != channel_dim]
    x2 = x.pow(2).sum(dim=reduce_dims)
    return x2.sqrt()


def channel_rms(act: torch.Tensor) -> torch.Tensor:
    # act shape [..., I], aggregate over sample dims and keep channel dim.
    x = act.detach()
    dims = tuple(range(x.dim() - 1))
    x2 = x.pow(2).sum(dim=dims)
    return x2.sqrt()


def get_fused_saved_tensor(experts: nn.Module, name: str, expert_idx: int):
    value = getattr(experts, name, None)
    if isinstance(value, (list, tuple)):
        if expert_idx >= len(value):
            return None
        return value[expert_idx]
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim == 0 or value.shape[0] <= expert_idx:
        return None
    return value[expert_idx]


_RUNNING_MEAN_COUNT_SUFFIX = "_running_mean_count"


def safe_update_running_stat(target, value, *, key, aggregation="mean", ema=0.9):
    """Update one module metric with an order-independent mean or legacy EMA."""
    if aggregation not in {"mean", "ema"}:
        raise ValueError(f"Unsupported score aggregation: {aggregation}")
    if isinstance(value, torch.Tensor):
        value = value.detach()
    assert isinstance(target, nn.Module), f"target must be nn.Module, got {type(target)}"
    old = getattr(target, key, None)
    count_key = f"_{key}{_RUNNING_MEAN_COUNT_SUFFIX}"
    count = int(getattr(target, count_key, 0))

    if old is None or count == 0:
        updated = value.clone() if isinstance(value, torch.Tensor) else value
    elif aggregation == "ema":
        if isinstance(value, torch.Tensor):
            if isinstance(old, torch.Tensor):
                updated = old.mul(ema).add(value, alpha=1.0 - ema)
            elif isinstance(old, numbers.Number):
                updated = torch.as_tensor(old, dtype=value.dtype, device=value.device) * ema
                updated = updated + value * (1.0 - ema)
            else:
                raise TypeError(
                    f"EMA previous value must be Tensor or scalar number, got {type(old)}"
                )
        else:
            updated = old * ema + value * (1.0 - ema)
    else:
        # Stable online mean. The count belongs to this exact expert/metric pair,
        # so sparse routing does not dilute a routed-only metric with missing data.
        if isinstance(value, torch.Tensor):
            if not isinstance(old, torch.Tensor):
                old = torch.as_tensor(old, dtype=value.dtype, device=value.device)
            updated = old + (value - old) / float(count + 1)
        else:
            updated = old + (value - old) / float(count + 1)

    setattr(target, key, updated)
    setattr(target, count_key, count + 1)


def safe_add_with_ema(target, ema, value, key=None):
    """Backward-compatible wrapper for callers outside the score collector."""
    if key is None:
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if target is None:
                return value.clone()
            if isinstance(target, torch.Tensor):
                return target.mul(ema).add(value, alpha=1.0 - ema)
            old = torch.as_tensor(target, dtype=value.dtype, device=value.device)
            return old * ema + value * (1.0 - ema)
        return value if target is None else target * ema + value * (1.0 - ema)
    safe_update_running_stat(target, value, key=key, aggregation="ema", ema=ema)


def update_fused_running_stats(
    target,
    key,
    per_expert,
    *,
    num_experts,
    device,
    aggregation="mean",
    ema=0.9,
    accumulate_sum=False,
):
    """Commit sparse per-expert values to a fused expert container."""
    if not per_expert:
        return
    if aggregation not in {"mean", "ema"}:
        raise ValueError(f"Unsupported score aggregation: {aggregation}")

    template = next(iter(per_expert.values())).detach().to(device=device, dtype=torch.float32)
    current = getattr(target, key, None)
    if current is None:
        current = torch.zeros((num_experts, *template.shape), dtype=torch.float32, device=device)
    else:
        current = current.detach().to(device=device, dtype=torch.float32)

    count_key = f"_{key}{_RUNNING_MEAN_COUNT_SUFFIX}"
    counts = getattr(target, count_key, None)
    if counts is None:
        counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    else:
        counts = counts.detach().to(device=device, dtype=torch.int64)

    for expert_idx, raw_value in per_expert.items():
        value = raw_value.detach().to(device=device, dtype=torch.float32)
        count = int(counts[expert_idx].item())
        if accumulate_sum:
            current[expert_idx].add_(value)
        elif count == 0:
            current[expert_idx] = value
        elif aggregation == "ema":
            current[expert_idx].mul_(ema).add_(value, alpha=1.0 - ema)
        else:
            current[expert_idx].add_((value - current[expert_idx]) / float(count + 1))
        counts[expert_idx] += 1

    setattr(target, key, current)
    setattr(target, count_key, counts)


def unwrap_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def get_saved_tensors(
    experts=None,
    expert_idx=None,
    is_fused=False,
    expert=None,
    down_proj_t=None,
    up_proj=None,
    gate_proj=None,
    down_grad_t=None,
    up_grad_w=None,
    gate_grad_w=None,
):
    if is_fused:
        activation_owner = experts
        down_input = get_fused_saved_tensor(experts, "saved_down_input", expert_idx)
        down_output = get_fused_saved_tensor(experts, "saved_down_output", expert_idx)
        down_grad = get_fused_saved_tensor(experts, "saved_down_grad", expert_idx)
        down_out_grad = get_fused_saved_tensor(experts, "saved_down_out_grad", expert_idx)
        up_input = get_fused_saved_tensor(experts, "saved_up_input", expert_idx)
        up_output = get_fused_saved_tensor(experts, "saved_up_output", expert_idx)
        up_in_grad = get_fused_saved_tensor(experts, "saved_up_in_grad", expert_idx)
        up_out_grad = get_fused_saved_tensor(experts, "saved_up_out_grad", expert_idx)
        gate_input = get_fused_saved_tensor(experts, "saved_gate_input", expert_idx)
        gate_output = get_fused_saved_tensor(experts, "saved_gate_output", expert_idx)
        gate_in_grad = get_fused_saved_tensor(experts, "saved_gate_in_grad", expert_idx)
        gate_grad = get_fused_saved_tensor(experts, "saved_gate_grad", expert_idx)
        text_mask = get_fused_saved_tensor(experts, "saved_text_mask", expert_idx)
        visual_mask = get_fused_saved_tensor(experts, "saved_visual_mask", expert_idx)
        score_mask = get_fused_saved_tensor(experts, "saved_score_mask", expert_idx)
        router_weights = get_fused_saved_tensor(experts, "saved_router_weights", expert_idx)

        W_down = down_proj_t[expert_idx]
        W_up = up_proj[expert_idx]
        W_gate = gate_proj[expert_idx]
        W_down_grad = None if down_grad_t is None else down_grad_t[expert_idx]
        W_up_grad = None if up_grad_w is None else up_grad_w[expert_idx]
        W_gate_grad = None if gate_grad_w is None else gate_grad_w[expert_idx]
    else:
        activation_owner = expert
        down_input = getattr(expert.down_proj, "saved_input", None)
        down_output = getattr(expert.down_proj, "saved_output", None)
        down_grad = getattr(expert.down_proj, "saved_grad_in", None)
        down_out_grad = getattr(expert.down_proj, "saved_grad_out", None)
        expert.down_proj.saved_input = None
        expert.down_proj.saved_output = None
        expert.down_proj.saved_grad_in = None
        expert.down_proj.saved_grad_out = None

        up_input = getattr(expert.up_proj, "saved_input", None)
        up_output = getattr(expert.up_proj, "saved_output", None)
        up_in_grad = getattr(expert.up_proj, "saved_grad_in", None)
        up_out_grad = getattr(expert.up_proj, "saved_grad_out", None)
        expert.up_proj.saved_input = None
        expert.up_proj.saved_output = None
        expert.up_proj.saved_grad_in = None
        expert.up_proj.saved_grad_out = None

        gate_input = getattr(expert.gate_proj, "saved_input", None)
        gate_output = getattr(expert.gate_proj, "saved_output", None)
        gate_in_grad = getattr(expert.gate_proj, "saved_grad_in", None)
        gate_grad = getattr(expert.gate_proj, "saved_grad_out", None)
        expert.gate_proj.saved_input = None
        expert.gate_proj.saved_output = None
        expert.gate_proj.saved_grad_in = None
        expert.gate_proj.saved_grad_out = None
        text_mask = getattr(expert, "saved_text_mask", None)
        visual_mask = getattr(expert, "saved_visual_mask", None)
        score_mask = getattr(expert, "saved_score_mask", None)
        router_weights = getattr(expert, "saved_router_weights", None)
        expert.saved_text_mask = None
        expert.saved_visual_mask = None
        expert.saved_score_mask = None
        expert.saved_router_weights = None

        W_down, W_up, W_gate = expert.down_proj.weight, expert.up_proj.weight, expert.gate_proj.weight
        W_down_grad = W_down.grad
        W_up_grad = W_up.grad
        W_gate_grad = W_gate.grad

    return (
        activation_owner,
        down_input,
        down_output,
        down_grad,
        down_out_grad,
        up_input,
        up_output,
        up_in_grad,
        up_out_grad,
        gate_input,
        gate_output,
        gate_in_grad,
        gate_grad,
        text_mask,
        visual_mask,
        score_mask,
        router_weights,
        W_down,
        W_up,
        W_gate,
        W_down_grad,
        W_up_grad,
        W_gate_grad,
    )
