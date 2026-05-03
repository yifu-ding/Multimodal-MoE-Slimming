import contextlib
import os
import torch.nn as nn
import types
from typing import Optional
import torch.nn.functional as F

import torch
from src.base.shared_utils import angle_loss
from .utils import unwrap_output
from src.calibration.helpers.utils import (
    get_fused_expert_layout,
    get_fused_intermediate_size,
    split_fused_gate_up_tensor,
)


class _LinearWeightView:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight


class FusedExpertProxy(nn.Module):
    def __init__(self, fused_container: nn.Module, expert_idx: int):
        super().__init__()
        self.fused_container = fused_container
        self.expert_idx = int(expert_idx)
        self.act_fn = getattr(fused_container, "act_fn", torch.nn.functional.silu)
        self.fused_layout = get_fused_expert_layout(fused_container)

        gate_up = fused_container.gate_up_proj[self.expert_idx].detach().transpose(0, 1)
        gate_proj, up_proj = split_fused_gate_up_tensor(fused_container, gate_up)
        self.gate_proj = _LinearWeightView(gate_proj)
        self.up_proj = _LinearWeightView(up_proj)
        self.down_proj = _LinearWeightView(
            fused_container.down_proj[self.expert_idx].detach().transpose(0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_single_fused_expert(self.fused_container, self.expert_idx, x)


def _is_fused_expert_proxy(expert: nn.Module) -> bool:
    return hasattr(expert, "fused_container") and hasattr(expert, "expert_idx")


def make_fused_expert_proxy(fused_container: nn.Module, expert_idx: int) -> FusedExpertProxy:
    return FusedExpertProxy(fused_container, expert_idx)


def _fused_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D fused expert weight, got shape={tuple(weight.shape)}")
    if weight.shape[0] == x.shape[-1]:
        return x @ weight
    return torch.nn.functional.linear(x, weight)


def _run_single_fused_expert(
    fused_container: nn.Module,
    expert_idx: int,
    hidden_states: torch.Tensor,
    *,
    output_alpha: torch.Tensor | None = None,
    channel_alpha: torch.Tensor | None = None,
    zero_output: bool = False,
) -> torch.Tensor:
    gate_up = _fused_linear(hidden_states, fused_container.gate_up_proj[expert_idx])
    fused_layout = get_fused_expert_layout(fused_container)
    if fused_layout == "gpt_oss":
        gate_up = gate_up + fused_container.gate_up_proj_bias[expert_idx].to(
            device=gate_up.device, dtype=gate_up.dtype
        )
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=fused_container.limit)
        up = up.clamp(min=-fused_container.limit, max=fused_container.limit)
        hidden = (up + 1) * (gate * torch.sigmoid(gate * fused_container.alpha))
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = fused_container.act_fn(gate) * up
    if channel_alpha is not None:
        hidden = hidden * channel_alpha.to(device=hidden.device, dtype=hidden.dtype)
    out = _fused_linear(hidden, fused_container.down_proj[expert_idx])
    if fused_layout == "gpt_oss":
        out = out + fused_container.down_proj_bias[expert_idx].to(
            device=out.device, dtype=out.dtype
        )
    if zero_output:
        out = torch.zeros_like(out)
    if output_alpha is not None:
        out = out * output_alpha.to(device=out.device, dtype=out.dtype)
    return out


def _patch_fused_container_forward(
    fused_container: nn.Module,
    expert_idx: int,
    *,
    output_alpha: torch.Tensor | None = None,
    channel_alpha: torch.Tensor | None = None,
    zero_output: bool = False,
):
    state = {"container": fused_container, "forward": fused_container.forward}

    def _forward_with_patch(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        _expert_idx=expert_idx,
        _output_alpha=output_alpha,
        _channel_alpha=channel_alpha,
        _zero_output=zero_output,
    ):
        fused_layout = get_fused_expert_layout(self)
        if fused_layout == "gpt_oss":
            batch_size = hidden_states.shape[0]
            hidden_size = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(-1, hidden_size)
        next_states = torch.zeros_like(hidden_states)
        num_classes = self.num_experts + 1 if fused_layout == "gpt_oss" else self.num_experts
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=num_classes
        ).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_tensor in expert_hit:
            current_expert_idx = int(expert_tensor[0].item())
            if current_expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[current_expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = _run_single_fused_expert(
                self,
                current_expert_idx,
                current_state,
                output_alpha=_output_alpha if current_expert_idx == _expert_idx else None,
                channel_alpha=_channel_alpha if current_expert_idx == _expert_idx else None,
                zero_output=_zero_output and current_expert_idx == _expert_idx,
            )
            if fused_layout == "gpt_oss":
                expert_routing_weights = routing_weights[token_idx, current_expert_idx]
            else:
                expert_routing_weights = routing_weights[token_idx, top_k_pos]
            current_hidden_states = current_hidden_states * expert_routing_weights[:, None]
            next_states.index_add_(0, token_idx, current_hidden_states.to(next_states.dtype))
        if fused_layout == "gpt_oss":
            return next_states.view(batch_size, -1, hidden_size)
        return next_states

    fused_container.forward = types.MethodType(_forward_with_patch, fused_container)
    return state


def compute_block_loss(
    pred: torch.Tensor,
    teacher_target: torch.Tensor,
    attn_mask: torch.Tensor,
    loss_fn: str,
    eps: float = 1e-6,
    token_mask: Optional[torch.Tensor] = None,  # 如果指定了模态的话
) -> torch.Tensor:
    mask_f = attn_mask.float()
    rel_l2_inv_base_mean = None
    if token_mask is not None:
        mask_f = mask_f * token_mask.float()

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

    if loss_fn == "kl_div":
        pred_logprob = F.log_softmax(pred.float(), dim=-1)
        teacher_prob = F.softmax(teacher_target.float(), dim=-1)
        token_kl = F.kl_div(pred_logprob, teacher_prob, reduction="none").sum(dim=-1)
        return (token_kl * mask_f).sum()

    raise ValueError(f"Unsupported loss_fn for second-order scoring: {loss_fn}")


@contextlib.contextmanager
def suspend_tensor_saving(module: nn.Module):
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


def patch_expert_output_alpha(expert: nn.Module, alpha: torch.Tensor):
    if _is_fused_expert_proxy(expert):
        return _patch_fused_container_forward(
            expert.fused_container,
            expert.expert_idx,
            output_alpha=alpha,
        )
    state = {"expert": expert, "forward": expert.forward}
    base_forward = expert.forward

    def _forward_with_alpha(self, x, _alpha=alpha, _base_forward=base_forward):
        out = _base_forward(x)
        return out * _alpha.to(device=out.device, dtype=out.dtype)

    expert.forward = types.MethodType(_forward_with_alpha, expert)
    return state


def patch_expert_output_alpha_vector(experts, alpha: torch.Tensor):
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        state = {"container": experts, "forward": experts.forward}

        def _forward_with_alpha_vector(
            self,
            hidden_states: torch.Tensor,
            router_indices: torch.Tensor,
            routing_weights: torch.Tensor,
            _alpha=alpha,
        ):
            fused_layout = get_fused_expert_layout(self)
            if fused_layout == "gpt_oss":
                batch_size = hidden_states.shape[0]
                hidden_size = hidden_states.shape[-1]
                hidden_states = hidden_states.reshape(-1, hidden_size)
            next_states = torch.zeros_like(hidden_states)
            num_classes = self.num_experts + 1 if fused_layout == "gpt_oss" else self.num_experts
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=num_classes
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_tensor in expert_hit:
                current_expert_idx = int(expert_tensor[0].item())
                if current_expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[current_expert_idx])
                current_state = hidden_states[token_idx]
                current_hidden_states = _run_single_fused_expert(
                    self,
                    current_expert_idx,
                    current_state,
                    output_alpha=_alpha[current_expert_idx],
                )
                if fused_layout == "gpt_oss":
                    expert_routing_weights = routing_weights[token_idx, current_expert_idx]
                else:
                    expert_routing_weights = routing_weights[token_idx, top_k_pos]
                current_hidden_states = current_hidden_states * expert_routing_weights[:, None]
                next_states.index_add_(0, token_idx, current_hidden_states.to(next_states.dtype))
            if fused_layout == "gpt_oss":
                return next_states.view(batch_size, -1, hidden_size)
            return next_states

        experts.forward = types.MethodType(_forward_with_alpha_vector, experts)
        return state

    states = []
    for expert_idx, expert in enumerate(experts):
        states.append(
            {
                "expert": expert,
                "forward": expert.forward,
                "expert_idx": expert_idx,
            }
        )
        base_forward = expert.forward

        def _forward_with_alpha(self, x, _alpha=alpha, _expert_idx=expert_idx, _base_forward=base_forward):
            out = _base_forward(x)
            return out * _alpha[_expert_idx].to(device=out.device, dtype=out.dtype)

        expert.forward = types.MethodType(_forward_with_alpha, expert)
    return {"states": states}


def patch_expert_channel_alpha(expert: nn.Module, alpha: torch.Tensor):
    """
    在 expert.down_proj 的输入（即 intermediate activation）上
    插入 per-channel 缩放因子 alpha, shape (I,).
    alpha 初始化为全 1, 求导后 -d1 + 0.5*d2 即为二阶 saliency.
    """
    if _is_fused_expert_proxy(expert):
        return _patch_fused_container_forward(
            expert.fused_container,
            expert.expert_idx,
            channel_alpha=alpha,
        )

    state = {"expert": expert, "handle": None}

    def _pre_hook(module, args, _alpha=alpha):
        x = args[0]  # shape [..., I]
        return (x * _alpha.to(device=x.device, dtype=x.dtype),)

    handle = expert.down_proj.register_forward_pre_hook(_pre_hook, with_kwargs=False)
    state["handle"] = handle
    return state

def _restore_patched_channel_alpha(state: dict):
    handle = state.get("handle")
    if handle is not None:
        handle.remove()
        return
    container = state.get("container")
    if container is not None:
        container.forward = state["forward"]
        
def patch_full_expert_mask(expert: nn.Module):
    if _is_fused_expert_proxy(expert):
        return _patch_fused_container_forward(
            expert.fused_container,
            expert.expert_idx,
            zero_output=True,
        )
    state = {"expert": expert, "forward": expert.forward}

    def _forward_zero_expert(self, x):
        out_dim = self.down_proj.weight.size(0)
        return x.new_zeros((*x.shape[:-1], out_dim))

    expert.forward = types.MethodType(_forward_zero_expert, expert)
    return state


def _restore_patched_expert(state: dict) -> None:
    container = state.get("container")
    if container is not None:
        container.forward = state["forward"]
        return
    states = state.get("states")
    if states is not None:
        for item in states:
            item["expert"].forward = item["forward"]
        return
    state["expert"].forward = state["forward"]


def get_block_eval_context(_kwargs: dict):
    if _kwargs is None:
        return None

    in_args = _kwargs.get("block_in_args", None)
    in_kwargs = _kwargs.get("block_in_kwargs", None)
    teacher_target = _kwargs.get("teacher_target", None)
    attn_mask = _kwargs.get("attn_mask", None)
    if in_args is None or in_kwargs is None or teacher_target is None or attn_mask is None:
        return None

    autocast_dtype = _kwargs.get("autocast_dtype", None)
    return {
        "in_args": in_args,
        "in_kwargs": in_kwargs,
        "teacher_target": teacher_target,
        "attn_mask": attn_mask,
        "loss_fn": _kwargs.get("loss_fn", "rel_l2"),
        "loss_eps": _kwargs.get("loss_eps", 1e-6),
        "autocast_dtype": autocast_dtype,
        "autocast_enabled": autocast_dtype in (torch.float16, torch.bfloat16),
        "autocast_device_type": _kwargs.get(
            "autocast_device_type",
            teacher_target.device.type if isinstance(teacher_target, torch.Tensor) else "cuda",
        ),
    }


def compute_second_approx_attr(
    down_output: torch.Tensor,
    down_out_grad: torch.Tensor,
    _kwargs: dict,
):
    if down_output is None or down_out_grad is None or _kwargs is None:
        return None

    attn_mask = _kwargs.get("attn_mask", None)
    if attn_mask is None:
        return None

    z = down_output.detach().float()
    g = down_out_grad.detach().float()
    total_tokens = max(float(attn_mask.sum().item()), 1.0)
    hidden_size = float(z.size(-1))
    loss_reduction = _kwargs.get("loss_reduction", "sum")

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


def compute_expert_second_order(
    cnt_block: nn.Module,
    expert: nn.Module,
    _kwargs: dict,
):
    context = get_block_eval_context(_kwargs)
    if context is None:
        return None

    alpha = torch.ones((), device=context["teacher_target"].device, dtype=torch.float32, requires_grad=True)
    state = patch_expert_output_alpha(expert, alpha=alpha)

    try:
        cnt_block.zero_grad(set_to_none=True)
        with suspend_tensor_saving(cnt_block):
            with torch.enable_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = unwrap_output(cnt_block(*context["in_args"], **context["in_kwargs"]))
                    loss = compute_block_loss(
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


def compute_expert_second_order_batched(
    cnt_block: nn.Module,
    experts,
    has_activation_mask,
    _kwargs: dict,
):
    context = get_block_eval_context(_kwargs)
    if context is None:
        return None

    device = context["teacher_target"].device
    has_activation = torch.as_tensor(has_activation_mask, device=device, dtype=torch.bool)
    num_experts = int(has_activation.numel())
    active_indices = has_activation.nonzero(as_tuple=False).flatten()
    chunk_size = int(os.getenv("SECOND_ORDER_CHUNK_SIZE", "8"))
    if chunk_size <= 0:
        raise ValueError(f"SECOND_ORDER_CHUNK_SIZE must be positive, got {chunk_size}")
    alpha = torch.ones(num_experts, device=device, dtype=torch.float32, requires_grad=True)
    state = patch_expert_output_alpha_vector(experts, alpha=alpha)

    try:
        cnt_block.zero_grad(set_to_none=True)
        with suspend_tensor_saving(cnt_block):
            with torch.enable_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = unwrap_output(cnt_block(*context["in_args"], **context["in_kwargs"]))
                    loss = compute_block_loss(
                        pred=pred,
                        teacher_target=context["teacher_target"],
                        attn_mask=context["attn_mask"],
                        loss_fn=context["loss_fn"],
                        eps=context["loss_eps"],
                    )
                d1 = torch.autograd.grad(loss, alpha, create_graph=True, allow_unused=True)[0]
                if d1 is None:
                    second_value = torch.zeros(num_experts, dtype=torch.float32, device=device)
                elif d1.requires_grad:
                    active_count = int(active_indices.numel())
                    if active_count == 0:
                        second_value = torch.zeros(num_experts, dtype=torch.float32, device=device)
                    else:
                        d2_diag = torch.zeros(active_count, dtype=torch.float32, device=device)
                        missing_hessian = False
                        for start in range(0, active_count, chunk_size):
                            end = min(start + chunk_size, active_count)
                            chunk_indices = active_indices[start:end]
                            grad_outputs = torch.eye(end - start, device=device, dtype=d1.dtype)
                            d2_rows = torch.autograd.grad(
                                d1[chunk_indices],
                                alpha,
                                grad_outputs=grad_outputs,
                                is_grads_batched=True,
                                retain_graph=end < active_count,
                                create_graph=False,
                                allow_unused=True,
                            )[0]
                            if d2_rows is None:
                                missing_hessian = True
                                break
                            local_positions = torch.arange(end - start, device=device)
                            d2_diag[start:end] = d2_rows[local_positions, chunk_indices].detach().float()

                        if missing_hessian:
                            second_value = (-d1).detach().float().clamp_min(0.0)
                        else:
                            second_value = torch.zeros(num_experts, dtype=torch.float32, device=device)
                            second_value[active_indices] = (
                                -d1[active_indices].detach().float() + 0.5 * d2_diag
                            ).clamp_min(0.0)
                else:
                    second_value = (-d1).detach().float().clamp_min(0.0)
    finally:
        _restore_patched_expert(state)
        cnt_block.zero_grad(set_to_none=True)

    return second_value.masked_fill(~has_activation, 0.0)


def compute_true_ablate_attr(
    cnt_block: nn.Module,
    expert: nn.Module,
    _kwargs: dict,
):
    context = get_block_eval_context(_kwargs)
    if context is None:
        return None

    base_loss = _kwargs.get("_true_ablate_base_loss", None)
    if base_loss is None:
        with suspend_tensor_saving(cnt_block):
            with torch.no_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = unwrap_output(cnt_block(*context["in_args"], **context["in_kwargs"]))
                    base_loss = compute_block_loss(
                        pred=pred,
                        teacher_target=context["teacher_target"],
                        attn_mask=context["attn_mask"],
                        loss_fn=context["loss_fn"],
                        eps=context["loss_eps"],
                    )
        _kwargs["_true_ablate_base_loss"] = base_loss.detach()

    state = patch_full_expert_mask(expert)
    try:
        with suspend_tensor_saving(cnt_block):
            with torch.no_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = unwrap_output(cnt_block(*context["in_args"], **context["in_kwargs"]))
                    masked_loss = compute_block_loss(
                        pred=pred,
                        teacher_target=context["teacher_target"],
                        attn_mask=context["attn_mask"],
                        loss_fn=context["loss_fn"],
                        eps=context["loss_eps"],
                    )
    finally:
        _restore_patched_expert(state)

    return (masked_loss - base_loss).detach().float().clamp_min(0.0)
        
        
def compute_down_second_order(
    cnt_block: nn.Module,
    expert: nn.Module,
    _kwargs: dict,
    modality_mask: Optional[torch.Tensor] = None,  # [B, S] bool, 哪些 token 参与 loss
):
    context = get_block_eval_context(_kwargs)
    if context is None:
        return None

    device = context["teacher_target"].device
    num_channels = expert.up_proj.weight.shape[0] if hasattr(expert.up_proj, "weight") else get_fused_intermediate_size(expert.fused_container)
    alpha = torch.ones(num_channels, device=device, dtype=torch.float32, requires_grad=True)

    # patch: h = h * alpha，在 expert 的 intermediate 输出处
    state = patch_expert_channel_alpha(expert, alpha=alpha)

    try:
        cnt_block.zero_grad(set_to_none=True)
        with suspend_tensor_saving(cnt_block):
            with torch.enable_grad():
                with torch.autocast(
                    device_type=context["autocast_device_type"],
                    dtype=context["autocast_dtype"],
                    enabled=context["autocast_enabled"],
                ):
                    pred = unwrap_output(
                        cnt_block(*context["in_args"], **context["in_kwargs"])
                    )
                    loss = compute_block_loss(
                        pred=pred,
                        teacher_target=context["teacher_target"],
                        attn_mask=context["attn_mask"],
                        loss_fn=context["loss_fn"],
                        eps=context["loss_eps"],
                        token_mask=modality_mask,  # 只在 text 或 visual token 上算 loss
                    )

                # 一阶: shape (I,)
                d1 = torch.autograd.grad(
                    loss, alpha, create_graph=True, allow_unused=True
                )[0]

                if d1 is None:
                    return torch.zeros(num_channels, device=device)

                # 二阶对角近似: 用 d1.sum() 对 alpha 求梯度
                if d1.requires_grad:
                    d2_approx = torch.autograd.grad(
                        d1.sum(), alpha,
                        retain_graph=False, create_graph=False, allow_unused=True
                    )[0]
                    if d2_approx is None:
                        saliency = (-d1).detach().float().clamp_min(0.0)
                    else:
                        saliency = (-d1 + 0.5 * d2_approx).detach().float().clamp_min(0.0)
                else:
                    saliency = (-d1).detach().float().clamp_min(0.0)
    finally:
        _restore_patched_channel_alpha(state)
        cnt_block.zero_grad(set_to_none=True)

    return saliency  # shape (I,)
