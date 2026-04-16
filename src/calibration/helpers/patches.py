import os
import types

import torch
import torch.nn.functional as F
from torch import nn

from .utils import get_fused_expert_layout, is_fused_expert_container

from .helpers import fused_linear


def patch_qwen_fused_experts_forward(block: nn.Module):
    experts = getattr(getattr(block, "mlp", None), "experts", None)
    if not is_fused_expert_container(experts):
        return None

    original = experts.forward
    fused_layout = get_fused_expert_layout(experts)

    def _save_grad_attr(obj, index: int, name: str):
        def _hook(grad):
            saved = getattr(obj, name)
            saved[index] = grad.detach()

        return _hook

    def _instrumented_forward(self, hidden_states, router_indices, routing_weights):
        num_experts = int(self.num_experts)
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
            setattr(self, name, [None] * num_experts)

        text_mask = getattr(block.mlp, "moe_text_mask", None)
        visual_mask = getattr(block.mlp, "moe_media_mask", None)
        padding_mask = getattr(block.mlp, "moe_padding_mask", None)
        if fused_layout == "gpt_oss":
            batch_size = hidden_states.shape[0]
            hidden_size = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(-1, hidden_size)
        if text_mask is None:
            text_mask = torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
        else:
            text_mask = text_mask.to(hidden_states.device).view(-1)
        if visual_mask is None:
            visual_mask = torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
        else:
            visual_mask = visual_mask.to(hidden_states.device).view(-1)
        if padding_mask is not None:
            keep = ~padding_mask.to(hidden_states.device).view(-1)
            text_mask = text_mask[keep]
            visual_mask = visual_mask[keep]

        next_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_tensor in expert_hit:
            expert_idx = int(expert_tensor[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = fused_linear(current_state, self.gate_up_proj[expert_idx])
            if fused_layout == "gpt_oss":
                gate_up_bias = self.gate_up_proj_bias[expert_idx]
                gate_up = gate_up + gate_up_bias.to(device=gate_up.device, dtype=gate_up.dtype)
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                current_hidden_states = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
            down_out = fused_linear(current_hidden_states, self.down_proj[expert_idx])
            if fused_layout == "gpt_oss":
                down_out = down_out + self.down_proj_bias[expert_idx].to(
                    device=down_out.device, dtype=down_out.dtype
                )
            if fused_layout == "gpt_oss":
                expert_routing_weights = routing_weights[token_idx, expert_idx]
            else:
                expert_routing_weights = routing_weights[token_idx, top_k_pos]
            weighted = down_out * expert_routing_weights[:, None]

            self.saved_gate_input[expert_idx] = current_state
            self.saved_gate_output[expert_idx] = gate
            self.saved_up_input[expert_idx] = current_state
            self.saved_up_output[expert_idx] = up
            self.saved_down_input[expert_idx] = current_hidden_states
            self.saved_down_output[expert_idx] = down_out
            self.saved_text_mask[expert_idx] = text_mask[token_idx]
            self.saved_visual_mask[expert_idx] = visual_mask[token_idx]
            self.saved_router_weights[expert_idx] = expert_routing_weights

            if current_state.requires_grad:
                current_state.register_hook(_save_grad_attr(self, expert_idx, "saved_gate_in_grad"))
                current_state.register_hook(_save_grad_attr(self, expert_idx, "saved_up_in_grad"))
            if gate.requires_grad:
                gate.register_hook(_save_grad_attr(self, expert_idx, "saved_gate_grad"))
            if up.requires_grad:
                up.register_hook(_save_grad_attr(self, expert_idx, "saved_up_out_grad"))
            if current_hidden_states.requires_grad:
                current_hidden_states.register_hook(
                    _save_grad_attr(self, expert_idx, "saved_down_grad")
                )
            if down_out.requires_grad:
                down_out.register_hook(_save_grad_attr(self, expert_idx, "saved_down_out_grad"))

            next_states.index_add_(0, token_idx, weighted.to(next_states.dtype))

        if fused_layout == "gpt_oss":
            return next_states.view(batch_size, -1, hidden_size)
        return next_states

    experts.forward = types.MethodType(_instrumented_forward, experts)
    return experts, original


def patch_grad_enabled_kimi_moe_infer(block: nn.Module, layer_idx: int):
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
