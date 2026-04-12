import os

import torch
import torch.nn as nn

from .utils import *
from .second_order_fn import *
from .naive_metric_fn import *



def loop_1_channelwise_scores(
    expert_iter,
    *,
    experts,
    is_fused: bool,
    down_proj_t,
    up_proj,
    gate_proj,
    down_grad_t,
    up_grad_w,
    gate_grad_w,
    fused_metric_stacks: dict,
    ema: float,
    _kwargs: dict = None,
):
    debug_down_input_hits = 0
    debug_gateup_hits = 0
    debug_total_experts = 0
    expert_records = []
    
        
    def _compute_metric_bundle(
        activation_owner: nn.Module,
        W_down: torch.Tensor,
        W_up: torch.Tensor,
        W_gate: torch.Tensor,
        *,
        W_down_grad: torch.Tensor = None,
        W_up_grad: torch.Tensor = None,
        W_gate_grad: torch.Tensor = None,
        down_input: torch.Tensor = None,
        down_output: torch.Tensor = None,
        down_grad: torch.Tensor = None,
        down_out_grad: torch.Tensor = None,
        up_input: torch.Tensor = None,
        up_output: torch.Tensor = None,
        up_in_grad: torch.Tensor = None,
        up_out_grad: torch.Tensor = None,
        gate_input: torch.Tensor = None,
        gate_output: torch.Tensor = None,
        gate_in_grad: torch.Tensor = None,
        gate_grad: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        visual_mask: torch.Tensor = None,
        router_weights: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ):
        metrics = {
            "weight": (
                weight_rms(W_down, channel_dim=1)
                + weight_rms(W_up, channel_dim=0)
                + weight_rms(W_gate, channel_dim=0)
            ).to(torch.float32)
            / 3.0
        }

        if W_down_grad is not None and W_up_grad is not None and W_gate_grad is not None:
            metrics["wg"] = compute_wg_I(
                W_down=W_down,
                W_up=W_up,
                W_gate=W_gate,
                W_down_grad=W_down_grad,
                W_up_grad=W_up_grad,
                W_gate_grad=W_gate_grad,
            ).to(torch.float32)

        if down_grad is not None and up_out_grad is not None and gate_grad is not None:
            grad_I, down_grad_ch, up_out_grad_ch, gate_grad_ch = compute_grad_I(
                down_grad, up_out_grad, gate_grad
            )
            metrics["3proj_grad"] = grad_I.to(torch.float32)
            grad_text = compute_grad_I_masked(down_grad, up_out_grad, gate_grad, text_mask)
            if grad_text is not None:
                metrics["3proj_grad_text"] = grad_text.to(torch.float32)
            grad_visual = compute_grad_I_masked(down_grad, up_out_grad, gate_grad, visual_mask)
            if grad_visual is not None:
                metrics["3proj_grad_visual"] = grad_visual.to(torch.float32)
        else:
            down_grad_ch = None
            up_out_grad_ch = None
            gate_grad_ch = None

        _want_text = _kwargs is not None and _kwargs.get("moe_text_mask", None) is not None
        _want_visual = _kwargs is not None and _kwargs.get("moe_media_mask", None) is not None
        _num_ch = W_down.size(1)
        _ch_dev = W_down.device

        if down_input is not None:
            metrics["down_second_order"] = compute_channel_hessian_diag(
                W_down, down_input, None
            )
        else:
            metrics["down_second_order"] = torch.zeros(
                _num_ch, dtype=torch.float32, device=_ch_dev
            )

        if down_input is not None and up_output is not None and gate_output is not None:
            metrics["3proj_second_order"] = compute_3linear_hessian_diag(
                W_down, W_up, W_gate, down_input, up_output, gate_output, None
            )
        else:
            metrics["3proj_second_order"] = torch.zeros(
                _num_ch, dtype=torch.float32, device=_ch_dev
            )

        if _want_text:
            if down_input is not None and text_mask is not None and isinstance(text_mask, torch.Tensor) and bool(text_mask.any()):
                metrics["down_second_order_text"] = compute_channel_hessian_diag(W_down, down_input, text_mask)
                if up_output is not None and gate_output is not None:
                    metrics["3proj_second_order_text"] = compute_3linear_hessian_diag(
                        W_down, W_up, W_gate, down_input, up_output, gate_output, text_mask
                    )
                else:
                    metrics["3proj_second_order_text"] = torch.zeros(_num_ch, dtype=torch.float32, device=_ch_dev)
            else:
                metrics["down_second_order_text"] = torch.zeros(_num_ch, dtype=torch.float32, device=_ch_dev)
                metrics["3proj_second_order_text"] = torch.zeros(_num_ch, dtype=torch.float32, device=_ch_dev)

        if _want_visual:
            if down_input is not None and visual_mask is not None and isinstance(visual_mask, torch.Tensor) and bool(visual_mask.any()):
                metrics["down_second_order_visual"] = compute_channel_hessian_diag(W_down, down_input, visual_mask)
                if up_output is not None and gate_output is not None:
                    metrics["3proj_second_order_visual"] = compute_3linear_hessian_diag(
                        W_down, W_up, W_gate, down_input, up_output, gate_output, visual_mask
                    )
                else:
                    metrics["3proj_second_order_visual"] = torch.zeros(_num_ch, dtype=torch.float32, device=_ch_dev)
            else:
                metrics["down_second_order_visual"] = torch.zeros(_num_ch, dtype=torch.float32, device=_ch_dev)
                metrics["3proj_second_order_visual"] = torch.zeros(_num_ch, dtype=torch.float32, device=_ch_dev)

        if down_input is None or up_output is None or gate_output is None:
            return metrics

        # gate*up activation
        gateup_act = compute_gateup_act(activation_owner, gate_output, up_output)
        if gateup_act is not None:
            metrics["gateup_act"] = gateup_act.to(torch.float32)
        # gate*up activation
        text_act = compute_gateup_act(
            activation_owner, gate_output, up_output, token_mask=text_mask
        )
        if text_act is not None:
            metrics["gateup_text"] = text_act.to(torch.float32)
        # gate*up activation
        visual_act = compute_gateup_act(
            activation_owner, gate_output, up_output, token_mask=visual_mask
        )
        if visual_act is not None:
            metrics["gateup_visual"] = visual_act.to(torch.float32)
        # three proj activation
        act_mean, down_act = compute_activation_I(down_input, up_output, gate_output)
        metrics["3proj_act"] = act_mean.to(torch.float32)
        activation_text = compute_activation_I_masked(
            down_input, up_output, gate_output, token_mask=text_mask
        )
        if activation_text is not None:
            metrics["3proj_act_text"] = activation_text.to(torch.float32)
        activation_visual = compute_activation_I_masked(
            down_input, up_output, gate_output, token_mask=visual_mask
        )
        if activation_visual is not None:
            metrics["3proj_act_visual"] = activation_visual.to(torch.float32)
        if up_input is not None and gate_input is not None:
            wa_I = compute_wa_I(
                W_down=W_down,
                W_up=W_up,
                W_gate=W_gate,
                down_ch_act=down_act,
                up_input=up_input,
                gate_input=gate_input,
            )
            metrics["wa"] = wa_I.to(torch.float32)
            wa_text = compute_wa_I_masked(
                W_down=W_down,
                W_up=W_up,
                W_gate=W_gate,
                down_input=down_input,
                up_input=up_input,
                gate_input=gate_input,
                token_mask=text_mask,
            )
            if wa_text is not None:
                metrics["wa_text"] = wa_text.to(torch.float32)
            wa_visual = compute_wa_I_masked(
                W_down=W_down,
                W_up=W_up,
                W_gate=W_gate,
                down_input=down_input,
                up_input=up_input,
                gate_input=gate_input,
                token_mask=visual_mask,
            )
            if wa_visual is not None:
                metrics["wa_visual"] = wa_visual.to(torch.float32)

        if (
            down_grad_ch is not None
            and up_out_grad_ch is not None
            and gate_grad_ch is not None
        ):  
            # this is down proj saliency
            saliency_I = compute_saliency_I(
                down_input,
                down_grad_ch,
                up_output,
                up_out_grad_ch,
                gate_output,
                gate_grad_ch,
            )
            metrics["down_saliency"] = saliency_I.to(torch.float32)
            saliency_3proj = compute_3proj_saliency_I(
                down_input,
                down_grad_ch,
                up_output,
                up_out_grad_ch,
                gate_output,
                gate_grad_ch,
            )
            metrics["3proj_saliency"] = saliency_3proj.to(torch.float32)

            # down_proj only, per-modality（与 activation_text / activation_visual 相同 token_mask 语义）
            # this is down proj saliency, per-modality
            sal_text = channel_saliency_masked(down_input, down_grad, text_mask)
            if sal_text is not None:
                metrics["down_saliency_text"] = sal_text.to(torch.float32)
            sal_3proj_text = compute_3proj_saliency_I_masked(
                down_input,
                down_grad,
                up_output,
                up_out_grad,
                gate_output,
                gate_grad,
                text_mask,
            )
            if sal_3proj_text is not None:
                metrics["3proj_saliency_text"] = sal_3proj_text.to(torch.float32)
            sal_visual = channel_saliency_masked(down_input, down_grad, visual_mask)
            if sal_visual is not None:
                metrics["down_saliency_visual"] = sal_visual.to(torch.float32)
            sal_3proj_visual = compute_3proj_saliency_I_masked(
                down_input,
                down_grad,
                up_output,
                up_out_grad,
                gate_output,
                gate_grad,
                visual_mask,
            )
            if sal_3proj_visual is not None:
                metrics["3proj_saliency_visual"] = sal_3proj_visual.to(torch.float32)

        if down_output is None or attn_mask is None:
            return metrics

        total_tokens = float(attn_mask.sum().item())
        usage = float(down_output.shape[0]) / max(total_tokens, 1.0)
        metrics["usage"] = usage
        if router_weights is not None:
            metrics["router"] = float(router_weights.detach().float().sum().item()) / max(total_tokens, 1.0)
        t_count = float(text_mask.sum().item()) if isinstance(text_mask, torch.Tensor) else 0.0
        v_count = float(visual_mask.sum().item()) if isinstance(visual_mask, torch.Tensor) else 0.0
        metrics["usage_text"] = t_count
        metrics["usage_visual"] = v_count
        # ema_matrix 在外部计算过了，我感觉不用 ema 平滑来算
        # Expert Modality Affinity: (visual - text) / (visual + text + eps), per batch, EMA-accumulated.
        # +1 means visual-preferring, -1 means text-preferring, 0 means balanced.
        # metrics["expert_modality_affinity"] = (v_count - t_count) / (v_count + t_count + 1e-8)
        if down_out_grad is not None:
            first_attr_usage = token_contrib(down_out_grad, down_output).sum() * usage
            metrics["first_attr"] = float(first_attr_usage.item())

        return metrics


    for expert_idx, expert in enumerate(expert_iter):
        debug_total_experts += 1

        activation_owner, down_input, down_output, down_grad, \
        down_out_grad, up_input, up_output, up_in_grad, up_out_grad, \
        gate_input, gate_output, gate_in_grad, gate_grad, text_mask, \
        visual_mask, router_weights, W_down, W_up, W_gate, W_down_grad, \
        W_up_grad, W_gate_grad = get_saved_tensors(
            experts=experts, expert_idx=expert_idx, is_fused=is_fused, expert=expert,
            down_proj_t=down_proj_t, up_proj=up_proj, gate_proj=gate_proj, down_grad_t=down_grad_t,
            up_grad_w=up_grad_w, gate_grad_w=gate_grad_w
        )

        with torch.no_grad():
            metrics = _compute_metric_bundle(
                activation_owner=activation_owner,
                W_down=W_down,
                W_up=W_up,
                W_gate=W_gate,
                W_down_grad=W_down_grad,
                W_up_grad=W_up_grad,
                W_gate_grad=W_gate_grad,
                down_input=down_input,
                down_output=down_output,
                down_grad=down_grad,
                down_out_grad=down_out_grad,
                up_input=up_input,
                up_output=up_output,
                up_in_grad=up_in_grad,
                up_out_grad=up_out_grad,
                gate_input=gate_input,
                gate_output=gate_output,
                gate_in_grad=gate_in_grad,
                gate_grad=gate_grad,
                text_mask=text_mask,
                visual_mask=visual_mask,
                router_weights=router_weights,
                attn_mask=None if _kwargs is None else _kwargs.get("attn_mask", None),
            )

            gateup_act = metrics.get("gateup_act", None)
            if down_input is not None:
                debug_down_input_hits += 1
            if isinstance(gateup_act, torch.Tensor) and float(gateup_act.abs().sum().item()) > 0:
                debug_gateup_hits += 1

            if is_fused:
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        stacked_value = value.to(torch.float32)
                    else:
                        stacked_value = torch.tensor(
                            float(value), dtype=torch.float32, device=experts.gate_up_proj.device
                        )
                    fused_metric_stacks.setdefault(key, []).append(stacked_value)
            else:
                for key, value in metrics.items():
                    if key in ("usage_text", "usage_visual"):
                        current = float(getattr(expert, key, 0.0))
                        setattr(expert, key, current + float(value))
                    else:
                        safe_add_with_ema(expert, ema, value, key)

            expert_records.append(
                {
                    "expert_idx": expert_idx,
                    "expert": expert,
                    "has_activation": down_input is not None,
                    "num_channels": int(W_up.shape[0]),
                }
            )

    return expert_records, debug_down_input_hits, debug_gateup_hits, debug_total_experts


def loop_2_expertwise_scores(
    expert_records,
    *,
    cnt_block,
    experts,
    is_fused: bool,
    fused_metric_stacks: dict,
    ema: float,
    _kwargs: dict = None,
):
    for record in expert_records:
        if not record["has_activation"]:
            if is_fused:
                device = experts.gate_up_proj.device
                fused_metric_stacks.setdefault("second_attr", []).append(
                    torch.zeros((), dtype=torch.float32, device=device)
                )
            continue

        expert_idx = record["expert_idx"]
        expert = record["expert"]
        expert_proxy = make_fused_expert_proxy(experts, expert_idx) if is_fused else expert

        second_exact_attr = compute_expert_second_order(
            cnt_block=cnt_block,
            expert=expert_proxy,
            _kwargs=_kwargs,
        )
        if second_exact_attr is not None:
            if is_fused:
                fused_metric_stacks.setdefault("second_attr", []).append(
                    second_exact_attr.detach().to(torch.float32)
                )
            else:
                safe_add_with_ema(expert, ema, second_exact_attr, "second_attr")

        # true_ablate = compute_true_ablate_attr(
        #     cnt_block=cnt_block,
        #     expert=expert_proxy,
        #     _kwargs=_kwargs,
        # )
        # if true_ablate is not None:
        #     if is_fused:
        #         fused_metric_stacks.setdefault("true_ablate", []).append(
        #             true_ablate.detach().to(torch.float32)
        #         )
        #     else:
        #         safe_add_with_ema(expert, ema, true_ablate, "true_ablate")


def collect_scores_from_moe_module(cnt_block, 
                            ema: float = 0.9, 
                            _kwargs: dict = None) -> None:
    experts = getattr(cnt_block.mlp, "experts", None)
    is_fused = is_fused_expert_container(experts)
    fused_metric_stacks = {}
    if is_fused:
        gate_up_proj = experts.gate_up_proj
        down_proj = experts.down_proj

        e, _, doubled_intermediate = gate_up_proj.shape
        intermediate_size = doubled_intermediate // 2

        gate_up_proj_t = gate_up_proj.detach().transpose(1, 2)  # [E, 2I, H]
        gate_proj = gate_up_proj_t[:, :intermediate_size, :]
        up_proj = gate_up_proj_t[:, intermediate_size:, :]
        down_proj_t = down_proj.detach().transpose(1, 2)  # [E, H, I]

        gate_up_grad = gate_up_proj.grad
        down_proj_grad = down_proj.grad
        if gate_up_grad is not None:
            gate_up_grad_t = gate_up_grad.detach().transpose(1, 2)
            gate_grad_w = gate_up_grad_t[:, :intermediate_size, :]
            up_grad_w = gate_up_grad_t[:, intermediate_size:, :]
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

    expert_records, debug_down_input_hits, debug_gateup_hits, debug_total_experts = (
        loop_1_channelwise_scores(
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
            _kwargs=_kwargs,
        )
    )

    loop_2_expertwise_scores(
        expert_records,
        cnt_block=cnt_block,
        experts=experts,
        is_fused=is_fused,
        fused_metric_stacks=fused_metric_stacks,
        ema=ema,
        _kwargs=_kwargs,
    )

    if is_fused:
        for key, values in fused_metric_stacks.items():
            if not values:
                continue
            stacked = torch.stack(values, dim=0)
            if key in ("usage_text", "usage_visual"):
                current = getattr(experts, key, None)
                if current is None:
                    setattr(experts, key, stacked)
                else:
                    setattr(experts, key, current + stacked)
            else:
                safe_add_with_ema(experts, ema, stacked, key)
        clear_fused_saved_tensors(experts)
