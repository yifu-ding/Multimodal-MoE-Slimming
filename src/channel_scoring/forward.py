import copy
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from tqdm import tqdm

from observations.common import move_inputs_to_model_device, prepare_inputs
from src.base.shared_utils import angle_loss
from src.channel_scoring.collector import collect_scores_attn_mlp

__all__ = [
    "block_forward",
]


def _iter_experts(mlp) -> List[nn.Module]:
    experts = getattr(mlp, "experts", None)
    if experts is None or not hasattr(experts, "__iter__"):
        return []
    return list(experts)


def _move_to_device_dtype(obj: Any, device: torch.device, dtype: torch.dtype):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device_dtype(x, device, dtype) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device_dtype(x, device, dtype) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device_dtype(v, device, dtype) for k, v in obj.items()}
    return obj


def _enable_input_grads(obj: Any):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.detach().requires_grad_(True)
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(_enable_input_grads(x) for x in obj)
    if isinstance(obj, list):
        return [_enable_input_grads(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _enable_input_grads(v) for k, v in obj.items()}
    return obj


def _unwrap_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _clear_block_saved_tensors(block: nn.Module) -> None:
    for expert in _iter_experts(block.mlp):
        for proj_name in ("down_proj", "up_proj", "gate_proj"):
            proj = getattr(expert, proj_name, None)
            if proj is None:
                continue
            for attr in ("saved_input", "saved_output", "saved_grad_in", "saved_grad_out"):
                setattr(proj, attr, None)


def _register_tensor_hooks(module: nn.Module) -> List[Any]:
    handles = []

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
        out = _unwrap_output(output)
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


def _register_teacher_block_hook(block: nn.Module, state: Dict[str, Any]):
    def _hook(module, args, kwargs, output):
        state["in_args"] = args
        state["in_kwargs"] = kwargs
        state["output"] = output

    return block.register_forward_hook(_hook, with_kwargs=True)


def _register_copied_block_hooks(block: nn.Module) -> List[Any]:
    handles = []
    for expert in _iter_experts(block.mlp):
        handles.extend(_register_tensor_hooks(expert.down_proj))
        handles.extend(_register_tensor_hooks(expert.up_proj))
        handles.extend(_register_tensor_hooks(expert.gate_proj))
    return handles


def _compute_block_loss(
    pred: torch.Tensor,
    teacher_target: torch.Tensor,
    attn_mask: torch.Tensor,
    loss_fn: str = "rel_l2",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_f = attn_mask.float()
    rel_l2_inv_base_mean = None

    if loss_fn == "l2":
        token_mse = (pred.float() - teacher_target.float()).pow(2).mean(dim=-1)
        return (token_mse * mask_f).sum(), rel_l2_inv_base_mean

    if loss_fn == "rel_l2":
        pred_f = pred.float().view(-1, pred.size(-1))
        target_f = teacher_target.float().view(-1, teacher_target.size(-1))
        mask_flat = mask_f.view(-1)
        diff2 = (pred_f - target_f).pow(2).sum(dim=-1)
        base2 = target_f.pow(2).sum(dim=-1)
        loss_vec = diff2 / (base2 + eps) * mask_flat
        valid = mask_flat > 0
        if valid.any():
            rel_l2_inv_base_mean = (1.0 / (base2[valid] + eps)).mean().detach().float()
        return loss_vec.sum(), rel_l2_inv_base_mean

    if loss_fn == "cosine":
        return (angle_loss(pred, teacher_target) * mask_f).sum(), rel_l2_inv_base_mean

    raise ValueError(f"Unsupported loss_fn: {loss_fn}")


def _kimi_teacher_block(model) -> Iterable[nn.Module]:
    return model.language_model.model.layers


def block_forward(
    bundle,
    cnt_block: nn.Module,
    layer_idx: int,
    dataloader,
    dataset_name: str,
    saliency_ema: float,
    loss_fn: str = "rel_l2",
    second_order_mode: str = "exact",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
):
    model = bundle.model
    model.eval()
    teacher_block = list(_kimi_teacher_block(model))[layer_idx]
    block_device = next(teacher_block.parameters()).device
    cnt_block = cnt_block.to(device=block_device, dtype=dtype)
    cnt_block.eval()

    teacher_state: Dict[str, Any] = {}
    teacher_handle = _register_teacher_block_hook(teacher_block, teacher_state)
    copied_handles = _register_copied_block_hooks(cnt_block)

    total_loss = 0.0
    total_batches = 0
    device_type = block_device.type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    try:
        iterator = tqdm(dataloader, desc=f"Calibrating L{layer_idx}", disable=not verbose, leave=False)
        for batch in iterator:
            teacher_state.clear()
            inputs = prepare_inputs(bundle, batch, dataset_name)
            inputs = move_inputs_to_model_device(model, inputs)
            attn_mask = inputs["attention_mask"].to(block_device)

            with torch.no_grad():
                model(**inputs, use_cache=False, return_dict=True)

            if not teacher_state:
                raise RuntimeError(f"Teacher block hook did not capture layer {layer_idx} inputs.")

            in_args = _enable_input_grads(
                _move_to_device_dtype(teacher_state["in_args"], block_device, dtype)
            )
            in_kwargs = _enable_input_grads(
                _move_to_device_dtype(teacher_state["in_kwargs"], block_device, dtype)
            )
            teacher_target = _unwrap_output(teacher_state["output"])
            teacher_target = _move_to_device_dtype(teacher_target, block_device, dtype)

            cnt_block.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
                pred = _unwrap_output(cnt_block(*in_args, **in_kwargs))
                loss_sum, rel_l2_inv_base_mean = _compute_block_loss(
                    pred=pred,
                    teacher_target=teacher_target,
                    attn_mask=attn_mask,
                    loss_fn=loss_fn,
                )
            loss_sum.backward()
            total_loss += float(loss_sum.detach().float().item())
            total_batches += 1

            collect_scores_attn_mlp(
                cnt_block,
                ema=saliency_ema,
                compute_H_scores_kwargs={
                    "use_mlp_scores": True,
                    "use_attn_scores": False,
                    "attn_mask": attn_mask,
                    "block_in_args": in_args,
                    "block_in_kwargs": in_kwargs,
                    "teacher_target": teacher_target,
                    "loss_fn": loss_fn,
                    "loss_reduction": "sum",
                    "loss_eps": 1e-6,
                    "rel_l2_inv_base_mean": rel_l2_inv_base_mean,
                    "second_order_mode": second_order_mode,
                    "autocast_dtype": dtype,
                    "autocast_device_type": device_type,
                },
            )
            _clear_block_saved_tensors(cnt_block)
    finally:
        teacher_handle.remove()
        for handle in copied_handles:
            handle.remove()
        _clear_block_saved_tensors(cnt_block)

    return total_loss / max(total_batches, 1)
