import copy
import os
import types
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from tqdm import tqdm

from observations.common import move_inputs_to_model_device, prepare_inputs
from src.base.shared_utils import angle_loss
from src.calibration.collector import collect_scores_from_moe_module

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


def unwrap_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _clear_block_saved_tensors(block: nn.Module) -> None:
    for expert in _iter_experts(block.mlp):
        for attr in ("saved_text_mask", "saved_visual_mask", "saved_router_weights"):
            setattr(expert, attr, None)
        for proj_name in ("down_proj", "up_proj", "gate_proj"):
            proj = getattr(expert, proj_name, None)
            if proj is None:
                continue
            for attr in ("saved_input", "saved_output", "saved_grad_in", "saved_grad_out"):
                setattr(proj, attr, None)


def _register_tensor_hooks(module: nn.Module) -> List[Any]:
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


def compute_block_loss(
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


def _patch_grad_enabled_kimi_moe_infer(block: nn.Module, layer_idx: int):
    mlp = getattr(block, "mlp", None)
    if mlp is None or not hasattr(mlp, "moe_infer"):
        return None

    original = mlp.moe_infer
    debug_routing = os.environ.get("MODES_DEBUG_ROUTING", "0") == "1"
    debug_max_calls = int(os.environ.get("MODES_DEBUG_ROUTING_MAX_CALLS", "2"))
    debug_state = {"calls": 0}

    def _grad_enabled_moe_infer(self, x, topk_ids, topk_weight, **kwargs):
        original_topk_ids = topk_ids
        skip_expert_idx = kwargs.get("skip_expert_idx", None)
        skip_modality = kwargs.get("skip_modality", None)
        batch_idx = kwargs.get("batch_idx", None)

        text_mask = getattr(self, "moe_text_mask", None)
        visual_mask = getattr(self, "moe_media_mask", None)
        if text_mask is None:
            text_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            text_mask = text_mask.to(x.device).view(-1)
        if visual_mask is None:
            visual_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            visual_mask = visual_mask.to(x.device).view(-1)

        for expert in self.experts:
            expert.saved_text_mask = None
            expert.saved_visual_mask = None
            expert.saved_router_weights = None

        if skip_expert_idx is not None:
            if batch_idx is not None:
                skip_mask = (
                    self.moe_text_mask_list[batch_idx]
                    if skip_modality == "text"
                    else self.moe_media_mask_list[batch_idx]
                )
            else:
                skip_mask = (
                    self.moe_text_mask if skip_modality == "text" else self.moe_media_mask
                )
            target_mask = (topk_ids == skip_expert_idx) & skip_mask.to(topk_ids.device)
            topk_weight = topk_weight.clone()
            topk_ids = topk_ids.clone()
            topk_weight.mul_(~target_mask)
            topk_ids[target_mask] = len(self.experts)

        if hasattr(self, "gate_dict") and self.gate_dict is not None:
            valid_mask = self.valid_expert_mask.to(topk_weight.device)
            topk_weight = topk_weight * valid_mask
            topk_ids = topk_ids.clone()
            topk_ids[~valid_mask] = len(self.experts)

        idxs = topk_ids.view(-1).argsort()
        flat_token_idx = idxs // topk_ids.shape[1]
        flat_routing_weight = topk_weight.reshape(-1)[idxs]
        sorted_tokens = x[flat_token_idx]
        sorted_text_mask = text_mask[flat_token_idx]
        sorted_visual_mask = visual_mask[flat_token_idx]
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts) + 1))
        src = torch.ones_like(topk_ids, dtype=cnts.dtype, device=cnts.device)
        cnts.scatter_add_(1, topk_ids, src)
        tokens_per_expert = cnts.sum(dim=0).tolist()

        outputs = []
        start_idx = 0
        called_experts = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + int(num_tokens)
            if num_tokens == 0:
                continue
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            if i == len(self.experts):
                outputs.append(tokens_for_this_expert)
                break
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            expert.saved_text_mask = sorted_text_mask[start_idx:end_idx]
            expert.saved_visual_mask = sorted_visual_mask[start_idx:end_idx]
            expert.saved_router_weights = flat_routing_weight[start_idx:end_idx]
            outputs.append(expert(tokens_for_this_expert))
            called_experts += 1
            start_idx = end_idx

        if debug_routing and debug_state["calls"] < debug_max_calls:
            total_experts = len(self.experts)
            pre_unique = torch.unique(original_topk_ids)
            pre_unique = pre_unique[pre_unique < total_experts]
            post_unique = torch.unique(topk_ids)
            post_unique = post_unique[post_unique < total_experts]
            skipped_ratio = float((topk_ids == total_experts).float().mean().item())
            has_gate_dict = hasattr(self, "gate_dict") and self.gate_dict is not None
            valid_ratio = (
                float(self.valid_expert_mask.float().mean().item())
                if has_gate_dict and hasattr(self, "valid_expert_mask")
                else 1.0
            )
            print(
                f"[routing-debug] L{layer_idx} call={debug_state['calls']} "
                f"gate_dict={has_gate_dict} valid_ratio={valid_ratio:.4f} "
                f"pre_unique={int(pre_unique.numel())} post_unique={int(post_unique.numel())} "
                f"called_experts={called_experts} skipped_ratio={skipped_ratio:.4f} "
                f"post_sample={post_unique[:12].detach().cpu().tolist()}",
                flush=True,
            )
            debug_state["calls"] += 1

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty((0, x.shape[-1]))
        new_x = torch.empty_like(outs)
        if idxs.numel() > 0:
            new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    mlp.moe_infer = types.MethodType(_grad_enabled_moe_infer, mlp)
    return mlp, original


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
    moe_infer_state = _patch_grad_enabled_kimi_moe_infer(cnt_block, layer_idx=layer_idx)

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
            input_ids = inputs.get("input_ids", None)

            # Ensure copied-block moe_infer can access per-token modality masks.
            # Unlike full-model forward, block-only calibration does not automatically
            # refresh these fields on the copied block.
            # 构造 block 级别的 modality mask（用于 channel 二阶计算）
            if input_ids is not None and hasattr(cnt_block, "mlp"):
                special_ids = getattr(model, "special_token_id_tensor", None)
                media_token_id = getattr(model.config, "media_placeholder_token_id", None)
                if special_ids is not None and media_token_id is not None:
                    special_ids = special_ids.to(input_ids.device)
                    moe_text_mask = ~torch.isin(input_ids, special_ids).view(-1)
                    moe_media_mask = (input_ids == media_token_id).view(-1)
                    cnt_block.mlp.moe_text_mask = moe_text_mask[:, None]
                    cnt_block.mlp.moe_media_mask = moe_media_mask[:, None]

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
            teacher_target = unwrap_output(teacher_state["output"])
            teacher_target = _move_to_device_dtype(teacher_target, block_device, dtype)

            cnt_block.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
                pred = unwrap_output(cnt_block(*in_args, **in_kwargs))
                loss_sum, rel_l2_inv_base_mean = compute_block_loss(
                    pred=pred,
                    teacher_target=teacher_target,
                    attn_mask=attn_mask,
                    loss_fn=loss_fn,
                )
            loss_sum.backward()
            total_loss += float(loss_sum.detach().float().item())
            total_batches += 1

            collect_scores_from_moe_module(
                cnt_block,
                ema=saliency_ema,
                _kwargs={
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
                    "layer_idx": layer_idx,
                    "debug_batch_idx": total_batches - 1,
                    "moe_text_mask": moe_text_mask.view_as(attn_mask),
                    "moe_media_mask": moe_media_mask.view_as(attn_mask),
                },
            )
            # if debug_routing and total_batches <= 1:
            #     experts = _iter_experts(cnt_block.mlp)
            #     hook_hits = sum(
            #         1
            #         for expert in experts
            #         if getattr(expert.down_proj, "saved_input", None) is not None
            #     )
            #     gateup_hits = 0
            #     for expert in experts:
            #         gateup = getattr(expert, "gateup_act", None)
            #         if isinstance(gateup, torch.Tensor) and float(gateup.abs().sum().item()) > 0:
            #             gateup_hits += 1
            #     print(
            #         f"[routing-debug] L{layer_idx} post_collect hook_hits={hook_hits}/{len(experts)} "
            #         f"gateup_hits={gateup_hits}/{len(experts)}",
            #         flush=True,
            #     )
            _clear_block_saved_tensors(cnt_block)
    finally:
        teacher_handle.remove()
        for handle in copied_handles:
            handle.remove()
        if moe_infer_state is not None:
            mlp, original_moe_infer = moe_infer_state
            mlp.moe_infer = original_moe_infer
        _clear_block_saved_tensors(cnt_block)

    return total_loss / max(total_batches, 1)
