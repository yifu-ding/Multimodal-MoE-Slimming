import time

import torch
from .utils import *
from .loop_1_score_collector import *
from .loop_2_score_collector import *
from src.calibration.helpers.utils import split_fused_gate_up_tensor


def collect_scores_from_moe_module(cnt_block,
                            ema: float = 0.9,
                            _kwargs: dict = None) -> None:
    kw = {} if _kwargs is None else _kwargs
    # profile = bool(kw.get("profile_inner_collector", False))
    profile = True

    def _mark():
        return time.perf_counter()

    t_prev = _mark() if profile else None

    def _elapsed_ms():
        nonlocal t_prev
        now = _mark()
        delta_ms = (now - t_prev) * 1000.0
        t_prev = now
        return delta_ms

    fill_zero_for_unrouted = bool(kw.get("fill_zero_for_unrouted", False))
    experts = getattr(cnt_block.mlp, "experts", None)
    is_fused = is_fused_expert_container(experts)
    fused_metric_stacks = {}
    if is_fused:
        gate_up_proj = experts.gate_up_proj
        down_proj = experts.down_proj

        e, _, doubled_intermediate = gate_up_proj.shape
        intermediate_size = doubled_intermediate // 2

        gate_up_proj_t = gate_up_proj.detach().transpose(1, 2)  # [E, 2I, H]
        gate_proj = torch.empty(
            (e, intermediate_size, gate_up_proj_t.shape[-1]),
            dtype=gate_up_proj_t.dtype,
            device=gate_up_proj_t.device,
        )
        up_proj = torch.empty_like(gate_proj)
        for expert_idx in range(e):
            gate_proj[expert_idx], up_proj[expert_idx] = split_fused_gate_up_tensor(
                experts, gate_up_proj_t[expert_idx]
            )
        down_proj_t = down_proj.detach().transpose(1, 2)  # [E, H, I]

        gate_up_grad = gate_up_proj.grad
        down_proj_grad = down_proj.grad
        if gate_up_grad is not None:
            gate_up_grad_t = gate_up_grad.detach().transpose(1, 2)
            gate_grad_w = torch.empty_like(gate_proj)
            up_grad_w = torch.empty_like(up_proj)
            for expert_idx in range(e):
                gate_grad_w[expert_idx], up_grad_w[expert_idx] = split_fused_gate_up_tensor(
                    experts, gate_up_grad_t[expert_idx]
                )
        else:
            gate_grad_w = None
            up_grad_w = None
        down_grad_t = (
            down_proj_grad.detach().transpose(1, 2) if down_proj_grad is not None else None
        )
        expert_iter = range(e)
    else:
        expert_iter = cnt_block.mlp.experts
        down_proj_t = None
        up_proj = None
        gate_proj = None
        down_grad_t = None
        up_grad_w = None
        gate_grad_w = None

    t_prep_ms = _elapsed_ms() if profile else None

    expert_records, debug_down_input_hits, debug_gateup_hits, debug_total_experts = (
        loop_1_score_collector(
            expert_iter,
            experts=experts,
            is_fused=is_fused,
            down_proj_t=down_proj_t,
            up_proj=up_proj,
            gate_proj=gate_proj,
            down_grad_t=down_grad_t,
            up_grad_w=up_grad_w,
            gate_grad_w=gate_grad_w,
            fused_metric_stacks=fused_metric_stacks,
            ema=ema,
            fill_zero_for_unrouted=fill_zero_for_unrouted,
            _kwargs=_kwargs,
        )
    )

    t_loop1_ms = _elapsed_ms() if profile else None

    second_order_sum = loop_2_score_collector(
        expert_records,
        cnt_block=cnt_block,
        experts=experts,
        is_fused=is_fused,
        fused_metric_stacks=fused_metric_stacks,
        ema=ema,
        _kwargs=_kwargs,
    )

    t_loop2_ms = _elapsed_ms() if profile else None

    t_fused_commit_ms = None
    if is_fused:
        num_experts = experts.gate_up_proj.shape[0]
        device = experts.gate_up_proj.device
        for key, per_expert in fused_metric_stacks.items():
            if not per_expert:
                continue
            # Keep loop_1 semantics consistent with non-fused: only routed experts
            # are updated for each key. Unrouted experts are untouched.
            template = next(iter(per_expert.values())).detach().to(device=device, dtype=torch.float32)
            current = getattr(experts, key, None)
            if current is None:
                current = torch.zeros((num_experts, *template.shape), dtype=torch.float32, device=device)
                is_first_update = True
            else:
                current = current.detach().to(device=device, dtype=torch.float32)
                is_first_update = False

            # if key in ("usage_text", "usage_visual"):
            #     for eid, v in per_expert.items():
            #         current[eid] += v.detach().to(device=device, dtype=torch.float32)
            # else:
            for eid, v in per_expert.items():
                value = v.detach().to(device=device, dtype=torch.float32)
                if key in ("token_count_text", "token_count_visual"):
                    if is_first_update:
                        current[eid] = value
                    else:
                        current[eid].add_(value)
                elif is_first_update:
                    current[eid] = value
                else:
                    current[eid].mul_(ema).add_(value, alpha=1.0 - ema)
            setattr(experts, key, current)
        clear_fused_saved_tensors(experts)
        if profile:
            t_fused_commit_ms = _elapsed_ms()

    if profile:
        parts = [
            ("prep_fused_or_iter", t_prep_ms),
            ("loop_1_score_collector", t_loop1_ms),
            ("loop_2_score_collector", t_loop2_ms),
        ]
        if t_fused_commit_ms is not None:
            parts.append(("fused_commit_clear", t_fused_commit_ms))
        total = sum(x for _, x in parts if x is not None)
        lines = [f"[inner_collector profile] total={total:.3f}ms"] + [
            f"  {name}: {ms:.3f}ms" for name, ms in parts if ms is not None
        ]
        print("\n".join(lines), flush=True)

    return second_order_sum