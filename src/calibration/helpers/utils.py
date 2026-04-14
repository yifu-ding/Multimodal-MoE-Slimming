from typing import Any, List

import torch
from torch import nn

def is_fused_expert_container(experts) -> bool:
    return (
        experts is not None
        and getattr(experts, "__class__", type(None)).__name__ == "Qwen3VLMoeTextExperts"
        and hasattr(experts, "gate_up_proj")
        and hasattr(experts, "down_proj")
    )


def clear_fused_saved_tensors(experts: nn.Module) -> None:
    for name in (
        "saved_down_input",
        "saved_down_output",
        "saved_down_grad",
        "saved_down_out_grad",
        "saved_up_input",
        "saved_up_output",
        "saved_up_in_grad",
        "saved_up_out_grad",
        "saved_gate_input",
        "saved_gate_output",
        "saved_gate_in_grad",
        "saved_gate_grad",
        "saved_text_mask",
        "saved_visual_mask",
        "saved_router_weights",
    ):
        if hasattr(experts, name):
            setattr(experts, name, None)


def iter_experts(mlp) -> List[nn.Module]:
    experts = getattr(mlp, "experts", None)
    if experts is None or not hasattr(experts, "__iter__"):
        return []
    return list(experts)


def move_to_device_dtype(obj: Any, device: torch.device, dtype: torch.dtype):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, tuple):
        return tuple(move_to_device_dtype(x, device, dtype) for x in obj)
    if isinstance(obj, list):
        return [move_to_device_dtype(x, device, dtype) for x in obj]
    if isinstance(obj, dict):
        return {k: move_to_device_dtype(v, device, dtype) for k, v in obj.items()}
    return obj


def enable_input_grads(obj: Any):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.detach().requires_grad_(True)
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(enable_input_grads(x) for x in obj)
    if isinstance(obj, list):
        return [enable_input_grads(x) for x in obj]
    if isinstance(obj, dict):
        return {k: enable_input_grads(v) for k, v in obj.items()}
    return obj


def unwrap_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def clear_block_saved_tensors(block: nn.Module) -> None:
    experts = getattr(getattr(block, "mlp", None), "experts", None)
    if is_fused_expert_container(experts):
        clear_fused_saved_tensors(experts)
        return
    for expert in iter_experts(block.mlp):
        for attr in ("saved_text_mask", "saved_visual_mask", "saved_router_weights"):
            setattr(expert, attr, None)
        for proj_name in ("down_proj", "up_proj", "gate_proj"):
            proj = getattr(expert, proj_name, None)
            if proj is None:
                continue
            for attr in ("saved_input", "saved_output", "saved_grad_in", "saved_grad_out"):
                setattr(proj, attr, None)
