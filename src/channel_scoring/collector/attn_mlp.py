import contextlib
import types

import torch
import torch.nn as nn

from src.base.shared_utils import angle_loss

from .utils import *


def _is_fused_expert_container(experts: nn.Module) -> bool:
    return (
        experts is not None
        and getattr(experts, "__class__", type(None)).__name__ == "Qwen3VLMoeTextExperts"
        and hasattr(experts, "gate_up_proj")
        and hasattr(experts, "down_proj")
    )


def _collect_scores_fused_experts(
    experts: nn.Module,
    ema: float = 0.9,
) -> None:
    gate_up_proj = experts.gate_up_proj
    down_proj = experts.down_proj

    e, hidden_size, doubled_intermediate = gate_up_proj.shape
    intermediate_size = doubled_intermediate // 2

    gate_up_proj_t = gate_up_proj.detach().transpose(1, 2)  # [E, 2I, H]
    gate_proj = gate_up_proj_t[:, :intermediate_size, :]
    up_proj = gate_up_proj_t[:, intermediate_size:, :]
    down_proj_t = down_proj.detach().transpose(1, 2)  # [E, H, I]

    weight_scores = []
    for expert_idx in range(e):
        weight_scores.append(
            (
                weight_rms(down_proj_t[expert_idx], channel_dim=1)
                + weight_rms(up_proj[expert_idx], channel_dim=0)
                + weight_rms(gate_proj[expert_idx], channel_dim=0)
            ).to(torch.float32)
            / 3.0
        )
    weight_scores = torch.stack(weight_scores, dim=0)
    safe_add_with_ema(experts, ema, weight_scores.to(torch.float32), "weight_scores")

    gate_up_grad = gate_up_proj.grad
    down_grad = down_proj.grad
    if gate_up_grad is None or down_grad is None:
        wg_scores = torch.zeros((e, intermediate_size), dtype=torch.float32, device=gate_up_proj.device)
    else:
        gate_up_grad_t = gate_up_grad.detach().transpose(1, 2)
        gate_grad = gate_up_grad_t[:, :intermediate_size, :]
        up_grad = gate_up_grad_t[:, intermediate_size:, :]
        down_grad_t = down_grad.detach().transpose(1, 2)
        wg_scores = []
        for expert_idx in range(e):
            wg_scores.append(
                compute_wg_I(
                    W_down=down_proj_t[expert_idx],
                    W_up=up_proj[expert_idx],
                    W_gate=gate_proj[expert_idx],
                    W_down_grad=down_grad_t[expert_idx],
                    W_up_grad=up_grad[expert_idx],
                    W_gate_grad=gate_grad[expert_idx],
                ).to(torch.float32)
            )
        wg_scores = torch.stack(wg_scores, dim=0)
    safe_add_with_ema(experts, ema, wg_scores, "wg_scores")


def _unwrap_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _compute_block_loss(
    pred: torch.Tensor,
    teacher_target: torch.Tensor,
    attn_mask: torch.Tensor,
    loss_fn: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    mask_f = attn_mask.float()

    if loss_fn == "l2":
        token_mse = (pred.float() - teacher_target.float()).pow(2).mean(dim=-1)
        return (token_mse * mask_f).sum()

    if loss_fn == "rel_l2":
        pred_f = pred.float().view(-1, pred.size(-1))
        target_f = teacher_target.float().view(-1, teacher_target.size(-1))
        mask_flat = mask_f.view(-1)
        diff2 = (pred_f - target_f).pow(2).sum(dim=-1)
        base2 = target_f.pow(2).sum(dim=-1)
        return (diff2 / (base2 + eps) * mask_flat).sum()

    if loss_fn == "cosine":
        return (angle_loss(pred, teacher_target) * mask_f).sum()

    raise ValueError(f"Unsupported loss_fn for second-order scoring: {loss_fn}")


@contextlib.contextmanager
def _suspend_tensor_saving(module: nn.Module):
    states = []
    for submodule in module.modules():
        had_attr = hasattr(submodule, "save_tensors")
        old_value = getattr(submodule, "save_tensors", None)
        states.append((submodule, had_attr, old_value))
        submodule.save_tensors = False
    try:
        yield
    finally:
        for submodule, had_attr, old_value in states:
            if had_attr:
                submodule.save_tensors = old_value
            else:
                delattr(submodule, "save_tensors")


def _patch_expert_output_alpha(expert: nn.Module, alpha: torch.Tensor):
    state = {"expert": expert, "forward": expert.forward}
    base_forward = expert.forward

    def _forward_with_alpha(self, x, _alpha=alpha, _base_forward=base_forward):
        out = _base_forward(x)
        return out * _alpha.to(device=out.device, dtype=out.dtype)

    expert.forward = types.MethodType(_forward_with_alpha, expert)
    return state


def _patch_full_expert_mask(expert: nn.Module):
    state = {"expert": expert, "forward": expert.forward}

    def _forward_zero_expert(self, x):
        out_dim = self.down_proj.weight.size(0)
        return x.new_zeros((*x.shape[:-1], out_dim))

    expert.forward = types.MethodType(_forward_zero_expert, expert)
    return state


def _restore_patched_expert(state: dict) -> None:
    state["expert"].forward = state["forward"]


def _get_block_eval_context(compute_H_scores_kwargs: dict):
    if compute_H_scores_kwargs is None:
        return None

    in_args = compute_H_scores_kwargs.get("block_in_args", None)
    in_kwargs = compute_H_scores_kwargs.get("block_in_kwargs", None)
    teacher_target = compute_H_scores_kwargs.get("teacher_target", None)
    attn_mask = compute_H_scores_kwargs.get("attn_mask", None)
    if in_args is None or in_kwargs is None or teacher_target is None or attn_mask is None:
        return None

    autocast_dtype = compute_H_scores_kwargs.get("autocast_dtype", None)
    return {
        "in_args": in_args,
        "in_kwargs": in_kwargs,
        "teacher_target": teacher_target,
        "attn_mask": attn_mask,
        "loss_fn": compute_H_scores_kwargs.get("loss_fn", "rel_l2"),
        "loss_eps": compute_H_scores_kwargs.get("loss_eps", 1e-6),
        "autocast_dtype": autocast_dtype,
        "autocast_enabled": autocast_dtype in (torch.float16, torch.bfloat16),
        "autocast_device_type": compute_H_scores_kwargs.get(
            "autocast_device_type",
            teacher_target.device.type if isinstance(teacher_target, torch.Tensor) else "cuda",
        ),
    }


def _compute_second_approx_attr(
    down_output: torch.Tensor,
    down_out_grad: torch.Tensor,
    expert_out_token_contrib: torch.Tensor,
    usage: float,
    compute_H_scores_kwargs: dict,
):
    if down_output is None or down_out_grad is None or compute_H_scores_kwargs is None:
        return None

    attn_mask = compute_H_scores_kwargs.get("attn_mask", None)
    if attn_mask is None:
        return None

    z = down_output.detach().float()
    g = down_out_grad.detach().float()
    total_tokens = max(float(attn_mask.sum().item()), 1.0)
    hidden_size = float(z.size(-1))
    loss_reduction = compute_H_scores_kwargs.get("loss_reduction", "sum")

    # 按 hidden-MSE 的闭式二阶展开来构造 approx-second:
    # Delta L_e ≈ [ -(2/NH) sum_i <r_i, z_i> + (1/NH) sum_i ||z_i||^2 ]_+
    # 若 backward 的是 token loss 的 sum, 则 g = 2r/H, 故一阶项为 -(1/N) sum <g, z>.
    # 若 backward 的是 token loss 的 mean, 则 g = 2r/(NH), 故一阶项为 -sum <g, z>.
    raw_inner = (g * z).sum()
    if loss_reduction == "sum":
        first_term = -raw_inner / total_tokens
    elif loss_reduction == "mean":
        first_term = -raw_inner
    else:
        raise ValueError(f"Unsupported loss_reduction for second_approx_attr: {loss_reduction}")

    second_term = z.pow(2).sum() / (total_tokens * hidden_size)
    second_approx_attr = (first_term + second_term).detach().float().clamp_min(0.0)
    return second_approx_attr


def _compute_second_exact_attr(
    cnt_block: nn.Module,
    expert: nn.Module,
    compute_H_scores_kwargs: dict,
):
    context = _get_block_eval_context(compute_H_scores_kwargs)
    if context is None:
        return None

    alpha = torch.ones((), device=context["teacher_target"].device, dtype=torch.float32, requires_grad=True)
    state = _patch_expert_output_alpha(expert, alpha=alpha)

    try:
        cnt_block.zero_grad(set_to_none=True)
        with _suspend_tensor_saving(cnt_block):
            with torch.enable_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = _unwrap_output(cnt_block(*context["in_args"], **context["in_kwargs"]))
                    loss = _compute_block_loss(
                        pred=pred,
                        teacher_target=context["teacher_target"],
                        attn_mask=context["attn_mask"],
                        loss_fn=context["loss_fn"],
                        eps=context["loss_eps"],
                    )
                d1 = torch.autograd.grad(loss, alpha, create_graph=True, allow_unused=True)[0]
                if d1 is None:
                    second_value = torch.zeros((), dtype=torch.float32, device=context["teacher_target"].device)
                elif d1.requires_grad:
                    d2 = torch.autograd.grad(d1, alpha, retain_graph=False, create_graph=False, allow_unused=True)[0]
                    if d2 is None:
                        second_value = (-d1).detach().float().clamp_min(0.0)
                    else:
                        second_value = (-d1 + 0.5 * d2).detach().float().clamp_min(0.0)
                else:
                    second_value = (-d1).detach().float().clamp_min(0.0)
    finally:
        _restore_patched_expert(state)
        cnt_block.zero_grad(set_to_none=True)

    return second_value


def _compute_true_ablate_attr(
    cnt_block: nn.Module,
    expert: nn.Module,
    compute_H_scores_kwargs: dict,
):
    context = _get_block_eval_context(compute_H_scores_kwargs)
    if context is None:
        return None

    base_loss = compute_H_scores_kwargs.get("_true_ablate_base_loss", None)
    if base_loss is None:
        with _suspend_tensor_saving(cnt_block):
            with torch.no_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = _unwrap_output(cnt_block(*context["in_args"], **context["in_kwargs"]))
                    base_loss = _compute_block_loss(
                        pred=pred,
                        teacher_target=context["teacher_target"],
                        attn_mask=context["attn_mask"],
                        loss_fn=context["loss_fn"],
                        eps=context["loss_eps"],
                    )
        compute_H_scores_kwargs["_true_ablate_base_loss"] = base_loss.detach()

    state = _patch_full_expert_mask(expert)
    try:
        with _suspend_tensor_saving(cnt_block):
            with torch.no_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = _unwrap_output(cnt_block(*context["in_args"], **context["in_kwargs"]))
                    masked_loss = _compute_block_loss(
                        pred=pred,
                        teacher_target=context["teacher_target"],
                        attn_mask=context["attn_mask"],
                        loss_fn=context["loss_fn"],
                        eps=context["loss_eps"],
                    )
    finally:
        _restore_patched_expert(state)

    return (masked_loss - base_loss).detach().float().clamp_min(0.0)
        
def collect_scores_attn_mlp(cnt_block, 
                            ema: float = 0.9, 
                            compute_H_scores_kwargs: dict = None) -> None:
    experts = getattr(cnt_block.mlp, "experts", None)
    if _is_fused_expert_container(experts):
        _collect_scores_fused_experts(experts, ema=ema)
    else:
        for expert in cnt_block.mlp.experts:
            # 取出并清空 hook 保存的张量
            down_input    = expert.down_proj.saved_input
            down_output   = expert.down_proj.saved_output
            down_grad     = expert.down_proj.saved_grad_in  
            down_out_grad = expert.down_proj.saved_grad_out
            expert.down_proj.saved_input    = None
            expert.down_proj.saved_output   = None
            expert.down_proj.saved_grad_in  = None
            expert.down_proj.saved_grad_out = None

            up_input    = expert.up_proj.saved_input
            up_output   = expert.up_proj.saved_output
            up_in_grad  = expert.up_proj.saved_grad_in
            up_out_grad = expert.up_proj.saved_grad_out
            expert.up_proj.saved_input    = None
            expert.up_proj.saved_output   = None
            expert.up_proj.saved_grad_in  = None
            expert.up_proj.saved_grad_out = None

            gate_input      = expert.gate_proj.saved_input
            gate_output     = expert.gate_proj.saved_output
            gate_in_grad    = expert.gate_proj.saved_grad_in
            gate_grad       = expert.gate_proj.saved_grad_out
            expert.gate_proj.saved_input    = None
            expert.gate_proj.saved_output   = None
            expert.gate_proj.saved_grad_in  = None
            expert.gate_proj.saved_grad_out = None
            
            W_down, W_up, W_gate = expert.down_proj.weight, expert.up_proj.weight, expert.gate_proj.weight
            
            # intermediate channel 分数收集
            with torch.no_grad():
                w_norm_mean = (weight_rms(W_down, channel_dim=1) + weight_rms(W_up, channel_dim=0) + weight_rms(W_gate, channel_dim=0)) / 3.0
                safe_add_with_ema(expert, ema, w_norm_mean, "weight")
                
                if down_input is not None:
                    wg_mean = compute_wg_I(W_down, W_up, W_gate, W_down.grad, W_up.grad, W_gate.grad)
                    safe_add_with_ema(expert, ema, wg_mean, "wg")
                
                    token_contrib_I = compute_token_contrib_I(down_input, down_grad, up_output, up_out_grad, gate_output, gate_grad)
                    safe_add_with_ema(expert, ema, token_contrib_I, "token_contrib")
                        
                    grad_I, down_grad, up_out_grad, gate_grad = compute_grad_I(down_grad, up_out_grad, gate_grad)
                    safe_add_with_ema(expert, ema, grad_I, "grad")
                        
                    saliency_I = compute_saliency_I(down_input, down_grad, up_output, up_out_grad, gate_output, gate_grad)
                    safe_add_with_ema(expert, ema, saliency_I, "saliency")

                    act_mean, down_ch_act = compute_activation_I(down_input, up_output, gate_output)
                    safe_add_with_ema(expert, ema, act_mean, "activation")
                    
                    wa_mean = compute_wa_I(W_down, W_up, W_gate, down_ch_act, up_input, gate_input)
                    safe_add_with_ema(expert, ema, wa_mean, "wa")
                    
                    # expert output 
                    attn_mask = compute_H_scores_kwargs.get("attn_mask", None)
                    total_tokens = float(attn_mask.sum().item())
                    usage = float(down_output.shape[0]) / max(total_tokens, 1.0)
                    expert_out_token_contrib = token_contrib(down_out_grad, down_output).sum() * usage
                    safe_add_with_ema(expert, ema, expert_out_token_contrib, "expert_out_token_contrib")
                    safe_add_with_ema(expert, ema, usage, "usage")
                    
                    second_exact_attr = _compute_second_exact_attr(
                        cnt_block=cnt_block,
                        expert=expert,
                        compute_H_scores_kwargs=compute_H_scores_kwargs,
                    )
                    if second_exact_attr is not None:
                        safe_add_with_ema(expert, ema, second_exact_attr, "second_exact_attr")

                    true_ablate = _compute_true_ablate_attr(
                        cnt_block=cnt_block,
                        expert=expert,
                        compute_H_scores_kwargs=compute_H_scores_kwargs,
                    )
                    if true_ablate is not None:
                        safe_add_with_ema(expert, ema, true_ablate, "true_ablate")