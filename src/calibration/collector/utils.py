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


def safe_add_with_ema(target, ema, value, key=None):
    if isinstance(value, torch.Tensor):
        value = value.detach()

    def ema_update(old, new):
        if isinstance(new, torch.Tensor):
            if old is None:
                return new.clone()
            old.mul_(ema).add_(new, alpha=1.0 - ema)
            return old
        return new if old is None else old * ema + new * (1.0 - ema)

    if key is None:
        return ema_update(target, value)

    assert isinstance(target, nn.Module), f"target must be nn.Module, got {type(target)}"
    old = getattr(target, key, None)
    setattr(target, key, ema_update(old, value))


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
        router_weights = getattr(expert, "saved_router_weights", None)
        expert.saved_text_mask = None
        expert.saved_visual_mask = None
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
        router_weights,
        W_down,
        W_up,
        W_gate,
        W_down_grad,
        W_up_grad,
        W_gate_grad,
    )
