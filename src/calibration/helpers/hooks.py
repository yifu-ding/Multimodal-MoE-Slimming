from typing import Any, Dict, List

import torch
from torch import nn

from .utils import iter_experts, unwrap_output


def register_tensor_hooks(module: nn.Module) -> List[Any]:
    handles = []
    module.saved_input = None
    module.saved_output = None
    module.saved_grad_in = None
    module.saved_grad_out = None

    def _pre_hook(mod, args):
        if getattr(mod, "save_tensors", True) is False:
            mod.saved_input = None
            mod.saved_grad_in = None
            return
        tensor = args[0] if isinstance(args, tuple) and len(args) > 0 else None
        if not isinstance(tensor, torch.Tensor):
            mod.saved_input = None
            mod.saved_grad_in = None
            return
        mod.saved_input = tensor
        mod.saved_grad_in = None
        if tensor.requires_grad:
            tensor.register_hook(lambda grad, m=mod: setattr(m, "saved_grad_in", grad.detach()))

    def _fwd_hook(mod, args, output):
        if getattr(mod, "save_tensors", True) is False:
            mod.saved_output = None
            mod.saved_grad_out = None
            return
        out = unwrap_output(output)
        if not isinstance(out, torch.Tensor):
            mod.saved_output = None
            mod.saved_grad_out = None
            return
        mod.saved_output = out
        mod.saved_grad_out = None
        if out.requires_grad:
            out.register_hook(lambda grad, m=mod: setattr(m, "saved_grad_out", grad.detach()))

    handles.append(module.register_forward_pre_hook(_pre_hook))
    handles.append(module.register_forward_hook(_fwd_hook))
    return handles


def register_teacher_block_hook(block: nn.Module, state: Dict[str, Any]):
    def _hook(module, args, kwargs, output):
        state["in_args"] = args
        state["in_kwargs"] = kwargs
        # Qwen3-VL DeepStack adds visual features to the first decoder outputs
        # in place. Snapshot the block target before that post-block mutation.
        state["output"] = unwrap_output(output).detach().clone()

    return block.register_forward_hook(_hook, with_kwargs=True)


def register_copied_block_hooks(block: nn.Module) -> List[Any]:
    handles = []
    for expert in iter_experts(block.mlp):
        handles.extend(register_tensor_hooks(expert.down_proj))
        handles.extend(register_tensor_hooks(expert.up_proj))
        handles.extend(register_tensor_hooks(expert.gate_proj))
    return handles
