"""Compatibility helpers for the Mistral4 text backbone in vLLM 0.11.2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


def register_transformers_config() -> None:
    from transformers import AutoConfig
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
        DeepseekV2Config,
    )

    class Mistral4Config(DeepseekV2Config):
        model_type = "mistral4"

        def __init__(
            self,
            rope_parameters: dict[str, Any] | None = None,
            architectures: list[str] | None = None,
            **kwargs: Any,
        ) -> None:
            if rope_parameters is not None and "rope_scaling" not in kwargs:
                rope_scaling = dict(rope_parameters)
                rope_scaling.pop("type", None)
                rope_scaling.pop("rope_theta", None)
                rope_scaling.pop("llama_4_scaling_beta", None)
                kwargs["rope_scaling"] = rope_scaling
                kwargs.setdefault("rope_theta", rope_parameters.get("rope_theta", 10000.0))
            super().__init__(**kwargs)
            self.rope_parameters = rope_parameters
            self.architectures = architectures or ["DeepseekV2ForCausalLM"]

    try:
        AutoConfig.register("mistral4", Mistral4Config)
    except ValueError as error:
        if "already used" not in str(error):
            raise


def expand_mistral4_fused_weights(
    weights: Iterable[tuple[str, Any]],
) -> Iterator[tuple[str, Any]]:
    """Translate Mistral4 fused expert tensors to DeepseekV2 loader names."""
    for name, tensor in weights:
        if ".mlp.experts.gate_up_proj" in name:
            base, suffix = name.split(".mlp.experts.gate_up_proj", 1)
            expert_base = f"{base}.mlp.experts"
            if suffix == "":
                for expert_id, packed in enumerate(tensor):
                    gate, up = packed.transpose(0, 1).chunk(2, dim=0)
                    yield f"{expert_base}.{expert_id}.gate_proj.weight", gate
                    yield f"{expert_base}.{expert_id}.up_proj.weight", up
            elif suffix == "_scale_inv":
                for expert_id, scale in enumerate(tensor):
                    scale = scale.squeeze()
                    yield f"{expert_base}.{expert_id}.gate_proj.weight_scale_inv", scale
                    yield f"{expert_base}.{expert_id}.up_proj.weight_scale_inv", scale
            elif suffix == "_activation_scale":
                for expert_id, scale in enumerate(tensor):
                    yield f"{expert_base}.{expert_id}.gate_proj.input_scale", scale
                    yield f"{expert_base}.{expert_id}.up_proj.input_scale", scale
            else:
                yield name, tensor
            continue
        if ".mlp.experts.down_proj" in name:
            base, suffix = name.split(".mlp.experts.down_proj", 1)
            expert_base = f"{base}.mlp.experts"
            if suffix == "":
                for expert_id, packed in enumerate(tensor):
                    yield (
                        f"{expert_base}.{expert_id}.down_proj.weight",
                        packed.transpose(0, 1),
                    )
            elif suffix == "_scale_inv":
                for expert_id, scale in enumerate(tensor):
                    yield (
                        f"{expert_base}.{expert_id}.down_proj.weight_scale_inv",
                        scale.squeeze(),
                    )
            elif suffix == "_activation_scale":
                for expert_id, scale in enumerate(tensor):
                    yield f"{expert_base}.{expert_id}.down_proj.input_scale", scale
            else:
                yield name, tensor
            continue
        if name.endswith(".activation_scale"):
            yield name.removesuffix(".activation_scale") + ".input_scale", tensor
            continue
        yield name, tensor


def install_deepseek_loader_into(model_module) -> None:
    if getattr(model_module, "_maes_mistral4_loader_installed", False):
        return
    import os

    from src.vllm_ep4_plan import load_ep4_plan

    original = model_module.DeepseekV2ForCausalLM.load_weights

    def matching_params(params_dict, base: str, leaf: str):
        exact = f"{base}.{leaf}"
        if exact in params_dict:
            return [(exact, params_dict[exact])]
        group_prefix = f"{base}.groups."
        return [
            (name, param)
            for name, param in params_dict.items()
            if name.startswith(group_prefix) and name.endswith(f".{leaf}")
        ]

    def per_expert_scale(scale_tensor):
        # Checkpoint stores fused per-expert FP8 scales with trailing
        # singleton dims (e.g. [num_experts, 1, 1]); vLLM's per-tensor
        # loader expects a scalar per expert. Block-quantized scales keep
        # their real (blocks_n, blocks_k) grid since those dims are > 1.
        while scale_tensor.dim() > 1 and scale_tensor.shape[-1] == 1:
            scale_tensor = scale_tensor.squeeze(-1)
        return scale_tensor

    def load_fused(params_dict, name, tensor, loaded_names):
        if ".mlp.experts.gate_up_proj" in name:
            base, suffix = name.split(".gate_up_proj", 1)
            if suffix == "":
                shards = tensor.transpose(-1, -2).chunk(2, dim=-2)
                entries = (("w13_weight", "w1", shards[0]), ("w13_weight", "w3", shards[1]))
            elif suffix == "_scale_inv":
                scale = per_expert_scale(tensor)
                entries = (("w13_weight_scale_inv", "w1", scale), ("w13_weight_scale_inv", "w3", scale))
            elif suffix == "_activation_scale":
                entries = (("w13_input_scale", "w1", tensor), ("w13_input_scale", "w3", tensor))
            else:
                return False
        elif ".mlp.experts.down_proj" in name:
            base, suffix = name.split(".down_proj", 1)
            if suffix == "":
                # Unlike gate_up_proj (checkpoint layout [experts, hidden,
                # 2*intermediate], needing a transpose to vLLM's w13 layout
                # [experts, 2*intermediate, hidden]), down_proj is already
                # stored as [experts, hidden, intermediate] -- exactly
                # vLLM's w2_weight layout (see Fp8MoEMethod.create_weights).
                # Transposing it here would silently swap the hidden and
                # intermediate dims and break width-sliced EP4 strategies.
                entries = (("w2_weight", "w2", tensor),)
            elif suffix == "_scale_inv":
                entries = (("w2_weight_scale_inv", "w2", per_expert_scale(tensor)),)
            elif suffix == "_activation_scale":
                entries = (("w2_input_scale", "w2", tensor),)
            else:
                return False
        else:
            return False

        for leaf, shard_id, expert_tensor in entries:
            targets = matching_params(params_dict, base, leaf)
            if not targets and leaf.endswith("_scale_inv"):
                # Non-block (per-tensor) FP8 checkpoints register the scale
                # parameter without the "_inv" suffix (see vLLM's
                # Fp8MoEMethod.create_weights: block_quant is False when the
                # model's quantization_config has weight_block_size=null).
                targets = matching_params(params_dict, base, leaf.removesuffix("_inv"))
            if not targets:
                raise KeyError(f"no vLLM parameter matches {base}.{leaf}")
            for target_name, param in targets:
                loaded_local = False
                for expert_id in range(expert_tensor.shape[0]):
                    value = expert_tensor[expert_id]
                    success = param.weight_loader(
                        param,
                        value,
                        target_name,
                        shard_id,
                        expert_id,
                        return_success=True,
                    )
                    loaded_local = bool(success) or loaded_local
                if loaded_local:
                    loaded_names.add(target_name)
        return True

    def compatible_load_weights(self, weights):
        if getattr(self.config, "model_type", None) != "mistral4":
            return original(self, weights)
        params_dict = dict(self.named_parameters())
        directly_loaded = set()

        def remaining_weights():
            for name, tensor in weights:
                if load_fused(params_dict, name, tensor, directly_loaded):
                    continue
                if name.endswith(".activation_scale"):
                    name = name.removesuffix(".activation_scale") + ".input_scale"
                yield name, tensor

        # Non-block (per-tensor) FP8 checkpoints name every weight scale with
        # a "_inv" suffix (block-quant DeepSeek-v2 convention), but vLLM's
        # Fp8LinearMethod only registers that suffix when block_quant is
        # True (see matching_params/load_fused above for the equivalent
        # fused-MoE-expert case). For ordinary quantized Linear layers,
        # DeepseekV2ForCausalLM.load_weights (`original`) also renames
        # checkpoint names internally (e.g. shared_experts' separate
        # gate_proj/up_proj -> fused gate_up_proj) before its own final
        # `params_dict[name]` lookup, so the mismatch can't be fixed by
        # renaming the checkpoint name up front here. Instead, alias every
        # "*.weight_scale" parameter under "*.weight_scale_inv" in the
        # named_parameters() that `original` builds its own params_dict
        # from, so whatever "_inv"-suffixed name it ends up looking for
        # still resolves to the real (non-block) parameter.
        real_named_parameters = self.named_parameters

        def aliased_named_parameters(*args, **kwargs):
            for name, param in real_named_parameters(*args, **kwargs):
                yield name, param
                if name.endswith(".weight_scale"):
                    yield name + "_inv", param

        self.named_parameters = aliased_named_parameters
        try:
            return set(original(self, remaining_weights())) | directly_loaded
        finally:
            del self.named_parameters

    model_module.DeepseekV2ForCausalLM.load_weights = compatible_load_weights

    plan_path = os.environ.get("MAES_EP4_PLAN")
    strategy = os.environ.get("MAES_EP4_STRATEGY", "cross_layer").strip().lower()
    if plan_path and strategy == "multi_kernel":
        import torch
        from torch import nn
        from vllm.model_executor.layers.fused_moe import FusedMoE

        from src.vllm_ep4_runtime import _CONTEXT, _multi_kernel_layer_plan

        plan = load_ep4_plan(plan_path)
        if "mistral" not in str(plan.get("model", "")).lower():
            model_module._maes_mistral4_loader_installed = True
            return
        original_moe_init = model_module.DeepseekV2MoE.__init__
        layer_id_to_position = {
            int(layer_id): position
            for position, layer_id in enumerate(plan["model_layer_ids"])
        }

        class MultiKernelExperts(nn.Module):
            is_internal_router = False

            def __init__(self, owner, config, parallel_config, quant_config, prefix, layer_id):
                super().__init__()
                from vllm.distributed import get_ep_group

                position = layer_id_to_position[layer_id]
                ep_rank = int(get_ep_group().rank_in_group)
                self.shared_experts = owner.shared_experts
                self.groups = nn.ModuleDict()
                for width in sorted(plan["active_widths"], reverse=True):
                    active = _multi_kernel_layer_plan(
                        plan, position, layer_id, ep_rank, int(width)
                    )
                    _CONTEXT.layer_plan = active
                    try:
                        group = FusedMoE(
                            num_experts=config.n_routed_experts,
                            top_k=config.num_experts_per_tok,
                            hidden_size=config.hidden_size,
                            intermediate_size=int(plan["intermediate_masks"].shape[2]),
                            reduce_results=False,
                            renormalize=config.norm_topk_prob,
                            quant_config=quant_config,
                            use_grouped_topk=True,
                            num_expert_group=getattr(config, "n_group", 1),
                            topk_group=getattr(config, "topk_group", 1),
                            prefix=f"{prefix}.experts.groups.{width}",
                            scoring_func=getattr(config, "scoring_func", "softmax"),
                            routed_scaling_factor=1.0,
                            e_score_correction_bias=owner.gate.e_score_correction_bias,
                            enable_eplb=False,
                            is_sequence_parallel=parallel_config.use_sequence_parallel_moe,
                        )
                        if not active.local_to_global:
                            with torch.no_grad():
                                for param in group.parameters():
                                    param.zero_()
                        self.groups[str(width)] = group
                    finally:
                        _CONTEXT.layer_plan = None

            def forward(self, hidden_states, router_logits):
                routed = None
                for group in self.groups.values():
                    output = group(hidden_states=hidden_states, router_logits=router_logits)
                    routed = output if routed is None else routed + output
                shared = (
                    None
                    if self.shared_experts is None
                    else self.shared_experts(hidden_states)
                )
                return shared, routed

        def compatible_moe_init(self, config, parallel_config, quant_config=None, prefix=""):
            original_moe_init(self, config, parallel_config, quant_config, prefix)
            import re

            match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", prefix)
            if match is None:
                return
            layer_id = int(match.group(1))
            if layer_id not in layer_id_to_position:
                return
            self.experts = MultiKernelExperts(
                self, config, parallel_config, quant_config, prefix, layer_id
            )

        model_module.DeepseekV2MoE.__init__ = compatible_moe_init
    model_module._maes_mistral4_loader_installed = True
