import torch
import torch.nn as nn
from .utils import *
from .loop_1_helpers import *
from src.calibration.helpers.score_namespace import CHANNEL_METRICS

def loop_1_score_collector(
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
    fill_zero_for_unrouted: bool = False,
    _kwargs: dict = None,
):
    debug_down_input_hits = 0
    debug_gateup_hits = 0
    debug_total_experts = 0
    expert_records = []

    def _make_zero_tensor(num_channels: int, *, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, num_channels), dtype=ref.dtype, device=ref.device)

    def _sanity_check_tensors(expert_idx: int, **tensors):
        missing = [name for name, value in tensors.items() if value is None]
        if missing:
            raise RuntimeError(
                f"Expert {expert_idx} is routed but missing required saved tensors: {', '.join(missing)}"
            )

    def _compute_metric_bundle(
        expert_idx: int,
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
        down_in_grad: torch.Tensor = None,
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
        force_zero_metrics: bool = False,
    ):
        if force_zero_metrics:
            num_channels = int(W_down.shape[1])
            hidden_size = int(W_down.shape[0])
            down_input = _make_zero_tensor(num_channels, ref=W_down)
            down_output = _make_zero_tensor(hidden_size, ref=W_down)
            down_in_grad = _make_zero_tensor(num_channels, ref=W_down)
            down_out_grad = _make_zero_tensor(hidden_size, ref=W_down)
            up_input = _make_zero_tensor(hidden_size, ref=W_up)
            up_output = _make_zero_tensor(num_channels, ref=W_up)
            up_in_grad = _make_zero_tensor(hidden_size, ref=W_up)
            up_out_grad = _make_zero_tensor(num_channels, ref=W_up)
            gate_input = _make_zero_tensor(hidden_size, ref=W_gate)
            gate_output = _make_zero_tensor(num_channels, ref=W_gate)
            gate_in_grad = _make_zero_tensor(hidden_size, ref=W_gate)
            gate_grad = _make_zero_tensor(num_channels, ref=W_gate)
            text_mask = torch.zeros(1, dtype=torch.bool, device=W_down.device)
            visual_mask = torch.zeros(1, dtype=torch.bool, device=W_down.device)
            router_weights = torch.zeros(1, dtype=torch.float32, device=W_down.device)

        _sanity_check_tensors(
            expert_idx,
            down_input=down_input,
            down_output=down_output,
            down_in_grad=down_in_grad,
            down_out_grad=down_out_grad,
            up_input=up_input,
            up_output=up_output,
            up_in_grad=up_in_grad,
            up_out_grad=up_out_grad,
            gate_input=gate_input,
            gate_output=gate_output,
            gate_in_grad=gate_in_grad,
            gate_grad=gate_grad,
        )
        
        # 三层线性层取平均
        metrics = {"weight": (weight_rms(W_down, channel_dim=1) + \
            weight_rms(W_up, channel_dim=0) + weight_rms(W_gate, channel_dim=0)) / 3.0}
        metrics["wg"] = compute_wg_I(
            W_down=W_down,
            W_up=W_up,
            W_gate=W_gate,
            W_down_grad=W_down_grad,
            W_up_grad=W_up_grad,
            W_gate_grad=W_gate_grad,)
        metrics["3proj_grad"], down_grad_ch, up_out_grad_ch, gate_grad_ch = compute_grad_I(
            down_in_grad, up_out_grad, gate_grad,)
        metrics["3proj_second_order"] = compute_3linear_hessian_diag(W_down, W_up, W_gate, down_input, up_output, gate_output, None)
        metrics["3proj_act"], down_act = compute_activation_I(down_input, up_output, gate_output)
        metrics["wa"] = compute_wa_I(W_down=W_down, W_up=W_up, W_gate=W_gate, down_ch_act=down_act, up_input=up_input, gate_input=gate_input)
        metrics["3proj_saliency"] = compute_3proj_saliency_I(down_input, down_grad_ch, up_output, up_out_grad_ch, gate_output, gate_grad_ch)
        
        # 只取中间一层来算
        metrics["gateup_act"] = compute_gateup_act(activation_owner, gate_output, up_output)
        metrics["down_second_order_approx"] = compute_channel_hessian_diag(W_down, down_input, None)
        metrics["down_saliency"] = compute_saliency_I(down_input, down_grad_ch)

        # 算 expertwise 的 usage、router 
        total_tokens = float(down_output.shape[0])
        if attn_mask is not None:
            total_tokens = float(attn_mask.sum().item())
            metrics["usage"] = float(down_output.shape[0]) / max(total_tokens, 1.0)
        if router_weights is not None:
            metrics["router"] = float(router_weights.detach().float().sum().item()) / max(total_tokens, 1.0)

        # 算 expert 输出 first_attr，这个其实算的是 epertwise loss
        first_attr = token_contrib(down_out_grad, down_output).sum() # * usage
        metrics["first_attr"] = float(first_attr.item())

        t_count = float(text_mask.sum().item()) if isinstance(text_mask, torch.Tensor) else 0.0
        v_count = float(visual_mask.sum().item()) if isinstance(visual_mask, torch.Tensor) else 0.0
        metrics["token_count_text"] = t_count
        metrics["token_count_visual"] = v_count

        if t_count > 0:
            metrics["3proj_grad_text"] = compute_grad_I_masked(down_in_grad, up_out_grad, gate_grad, text_mask)
            metrics["wg_text"] = metrics["wg"] * (t_count / max(total_tokens, 1.0))
            metrics["down_second_order_approx_text"] = compute_channel_hessian_diag(W_down, down_input, text_mask)
            metrics["3proj_second_order_text"] = compute_3linear_hessian_diag(W_down, W_up, W_gate, down_input, up_output, gate_output, text_mask)
            metrics["gateup_act_text"] = compute_gateup_act(activation_owner, gate_output, up_output, token_mask=text_mask)
            metrics["3proj_act_text"] = compute_activation_I_masked(down_input, up_output, gate_output, token_mask=text_mask)
            metrics["wa_text"] = compute_wa_I_masked(W_down=W_down, W_up=W_up, W_gate=W_gate, down_input=down_input, up_input=up_input, gate_input=gate_input, token_mask=text_mask)
            metrics["down_saliency_text"] = channel_saliency_masked(down_input, down_in_grad, text_mask)
            metrics["3proj_saliency_text"] = compute_3proj_saliency_I_masked(down_input, down_in_grad, up_output, up_out_grad, gate_output, gate_grad, text_mask)
            metrics["usage_text"] = t_count / max(total_tokens, 1.0)

        if v_count > 0:
            metrics["3proj_grad_visual"] = compute_grad_I_masked(down_in_grad, up_out_grad, gate_grad, visual_mask)
            metrics["wg_visual"] = metrics["wg"] * (v_count / max(total_tokens, 1.0))
            metrics["down_second_order_approx_visual"] = compute_channel_hessian_diag(W_down, down_input, visual_mask)
            metrics["3proj_second_order_visual"] = compute_3linear_hessian_diag(W_down, W_up, W_gate, down_input, up_output, gate_output, visual_mask)
            metrics["gateup_act_visual"] = compute_gateup_act(activation_owner, gate_output, up_output, token_mask=visual_mask)
            metrics["3proj_act_visual"] = compute_activation_I_masked(down_input, up_output, gate_output, token_mask=visual_mask)
            metrics["wa_visual"] = compute_wa_I_masked(W_down=W_down, W_up=W_up, W_gate=W_gate, down_input=down_input, up_input=up_input, gate_input=gate_input, token_mask=visual_mask)
            metrics["down_saliency_visual"] = channel_saliency_masked(down_input, down_in_grad, visual_mask)
            metrics["3proj_saliency_visual"] = compute_3proj_saliency_I_masked(down_input, down_in_grad, up_output, up_out_grad, gate_output, gate_grad, visual_mask)
            metrics["usage_visual"] = v_count / max(total_tokens, 1.0)
        
        # ema_matrix 在外部计算过了，我感觉不用 ema 平滑来算
        # Expert Modality Affinity: (visual - text) / (visual + text + eps), per batch, EMA-accumulated.
        # +1 means visual-preferring, -1 means text-preferring, 0 means balanced.
        # metrics["expert_modality_affinity"] = (v_count - t_count) / (v_count + t_count + 1e-8)
        if force_zero_metrics:
            zero_channel = torch.zeros_like(metrics["weight"])
            for key in CHANNEL_METRICS:
                if key.startswith("down_second_order_exact"):
                    continue
                metrics.setdefault(key, zero_channel.clone())
            metrics["usage"] = 0.0
            metrics["usage_text"] = 0.0
            metrics["usage_visual"] = 0.0
            metrics["router"] = 0.0
            metrics["first_attr"] = 0.0
            metrics["token_count_text"] = 0.0
            metrics["token_count_visual"] = 0.0
        return metrics

    for expert_idx, expert in enumerate(expert_iter):
        debug_total_experts += 1

        activation_owner, down_input, down_output, down_in_grad, \
        down_out_grad, up_input, up_output, up_in_grad, up_out_grad, \
        gate_input, gate_output, gate_in_grad, gate_grad, text_mask, \
        visual_mask, router_weights, W_down, W_up, W_gate, W_down_grad, \
        W_up_grad, W_gate_grad = get_saved_tensors(
            experts=experts, expert_idx=expert_idx, is_fused=is_fused, expert=expert,
            down_proj_t=down_proj_t, up_proj=up_proj, gate_proj=gate_proj, down_grad_t=down_grad_t,
            up_grad_w=up_grad_w, gate_grad_w=gate_grad_w
        )

        # Gate on routing: if this expert was not routed to this sample,
        # loop_1 metrics are strictly "no-update" for both fused/non-fused.
        # loop_2 handles second_attr / second_attr_fillzero policies.
        force_zero_metrics = down_input is None and fill_zero_for_unrouted
        if down_input is None and not force_zero_metrics:
            expert_records.append(
                {
                    "expert_idx": expert_idx,
                    "expert": expert,
                    "has_activation": False,
                    "num_channels": int(W_up.shape[0]),
                }
            )
            # 算 expert 输出 first_attr , 加上零值需要 fillzero 来补全之后 ema
            if is_fused:
                for key in ("first_attr_fillzero", "usage_fillzero", "router_fillzero"):
                    fused_metric_stacks.setdefault(key, {})[expert_idx] = \
                        torch.zeros((), dtype=torch.float32, device=experts.gate_up_proj.device)
            else:
                for key in ("first_attr_fillzero", "usage_fillzero", "router_fillzero"):
                    safe_add_with_ema(expert, ema, 0.0, key)
            continue

        with torch.no_grad():
            metrics = _compute_metric_bundle(
                expert_idx=expert_idx,
                activation_owner=activation_owner,
                W_down=W_down,
                W_up=W_up,
                W_gate=W_gate,
                W_down_grad=W_down_grad,
                W_up_grad=W_up_grad,
                W_gate_grad=W_gate_grad,
                down_input=down_input,
                down_output=down_output,
                down_in_grad=down_in_grad,
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
                force_zero_metrics=force_zero_metrics,
            )
            
            # gateup_act = metrics.get("gateup_act", None)
            # if isinstance(gateup_act, torch.Tensor) and float(gateup_act.abs().sum().item()) > 0:
            #     debug_gateup_hits += 1
            for key in ("usage", "router", "first_attr"):
                metrics[key+"_fillzero"] = float(metrics[key])
            if is_fused:
                for key, value in metrics.items():
                    stacked_value = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), dtype=torch.float32, device=experts.gate_up_proj.device)
                    fused_metric_stacks.setdefault(key, {})[expert_idx] = stacked_value
            else:
                for key, value in metrics.items():
                    if key in ("token_count_text", "token_count_visual"):
                        current = float(getattr(expert, key, 0.0))
                        setattr(expert, key, current + float(value))
                    else:
                        safe_add_with_ema(expert, ema, value, key)
            expert_records.append({"expert_idx": expert_idx, 
                                   "expert": expert, 
                                   "has_activation": down_input is not None, 
                                   "num_channels": int(W_up.shape[0])})
       

    return expert_records, debug_down_input_hits, debug_gateup_hits, debug_total_experts
