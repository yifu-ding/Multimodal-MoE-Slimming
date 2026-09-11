"""Opt-in vLLM 0.11.2 adapter for MAES heterogeneous EP4 plans."""

from __future__ import annotations

import inspect
import os
import re
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any


_LAYER_PATTERN = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.groups\.\d+)?$"
)
_CONTEXT = threading.local()
_VALID_STRATEGIES = {"cross_layer", "multi_kernel", "padded"}


@dataclass(frozen=True)
class _LayerPlan:
    plan_position: int
    model_layer_id: int
    ep_rank: int
    rank_width: int
    full_width: int
    num_experts: int
    expert_to_rank: Any
    expert_to_local: Any
    channel_masks: Any
    local_to_global: list[int]


def _strategy() -> str:
    value = os.environ.get("MAES_EP4_STRATEGY", "cross_layer").strip().lower()
    if value not in _VALID_STRATEGIES:
        raise ValueError(
            f"MAES_EP4_STRATEGY must be one of {sorted(_VALID_STRATEGIES)}, got {value!r}"
        )
    return value


def _slice_expert_weight(loaded_weight, shard_id: str, channel_indices, full_width: int):
    """Select planned FFN channels before vLLM performs any TP sharding."""
    if loaded_weight.ndim == 2:
        if shard_id in ("w1", "w3"):
            if loaded_weight.shape[0] != full_width:
                raise ValueError(
                    f"expected {shard_id} width {full_width}, got {tuple(loaded_weight.shape)}"
                )
            return loaded_weight.index_select(0, channel_indices.to(loaded_weight.device))
        if shard_id == "w2":
            if loaded_weight.shape[1] != full_width:
                raise ValueError(
                    f"expected w2 width {full_width}, got {tuple(loaded_weight.shape)}"
                )
            return loaded_weight.index_select(1, channel_indices.to(loaded_weight.device))
    elif loaded_weight.ndim == 3:
        dim = 1 if shard_id in ("w1", "w3") else 2
        if loaded_weight.shape[dim] != full_width:
            raise ValueError(
                f"expected packed {shard_id} width {full_width}, got {tuple(loaded_weight.shape)}"
            )
        return loaded_weight.index_select(dim, channel_indices.to(loaded_weight.device))
    raise ValueError(
        f"unsupported {shard_id} expert weight shape {tuple(loaded_weight.shape)}"
    )


def _zero_pruned_channels(loaded_weight, shard_id: str, channel_mask, full_width: int):
    """Keep the full kernel shape while zero-padding pruned channels."""
    import torch

    if loaded_weight.ndim == 2:
        dim = 0 if shard_id in ("w1", "w3") else 1
    elif loaded_weight.ndim == 3:
        dim = 1 if shard_id in ("w1", "w3") else 2
    else:
        raise ValueError(
            f"unsupported {shard_id} expert weight shape {tuple(loaded_weight.shape)}"
        )
    if loaded_weight.shape[dim] != full_width:
        raise ValueError(
            f"expected {shard_id} width {full_width}, got {tuple(loaded_weight.shape)}"
        )
    shape = [1] * loaded_weight.ndim
    shape[dim] = full_width
    mask = channel_mask.to(device=loaded_weight.device).view(shape)
    return loaded_weight * mask.to(dtype=loaded_weight.dtype)


def _rank_expert_map(expert_to_rank, expert_to_local, ep_rank: int, device=None):
    import torch

    expert_map = torch.where(
        expert_to_rank == ep_rank,
        expert_to_local,
        torch.full_like(expert_to_local, -1),
    ).to(dtype=torch.int32)
    return expert_map.to(device=device) if device is not None else expert_map


def _multi_kernel_layer_plan(plan, position: int, model_layer_id: int, ep_rank: int, width: int):
    import torch

    layer_widths = plan["expert_widths"][position]
    tier_experts = torch.where(layer_widths == width)[0].tolist()
    local_to_global = [
        int(expert_id)
        for index, expert_id in enumerate(tier_experts)
        if index % 4 == ep_rank
    ]
    expert_to_rank = torch.full_like(layer_widths, -1, dtype=torch.int64)
    expert_to_local = torch.full_like(layer_widths, -1, dtype=torch.int64)
    rank_local_counts = [0, 0, 0, 0]
    for index, expert_id in enumerate(tier_experts):
        rank = index % 4
        expert_to_rank[expert_id] = rank
        if rank == ep_rank:
            expert_to_local[expert_id] = rank_local_counts[rank]
        rank_local_counts[rank] += 1
    return _LayerPlan(
        plan_position=position,
        model_layer_id=model_layer_id,
        ep_rank=ep_rank,
        rank_width=width,
        full_width=int(plan["intermediate_masks"].shape[2]),
        num_experts=int(layer_widths.shape[0]),
        expert_to_rank=expert_to_rank,
        expert_to_local=expert_to_local,
        channel_masks=plan["intermediate_masks"][position],
        local_to_global=local_to_global,
    )


def install_into(layer_module: ModuleType) -> None:
    """Patch one imported vLLM fused-MoE module in the current worker."""
    if getattr(layer_module, "_maes_ep4_installed", False):
        return

    import torch

    from src.vllm_ep4_plan import load_ep4_plan
    from vllm.distributed import get_ep_group

    plan_path = os.environ.get("MAES_EP4_PLAN")
    if not plan_path:
        return
    plan = load_ep4_plan(plan_path)
    strategy = _strategy()
    layer_id_to_position = {
        int(layer_id): position
        for position, layer_id in enumerate(plan["model_layer_ids"])
    }
    plan_num_experts = int(plan["expert_widths"].shape[1])
    full_width = int(plan["intermediate_masks"].shape[2])

    original_determine_expert_map = layer_module.determine_expert_map
    original_init = layer_module.FusedMoE.__init__
    original_weight_loader = layer_module.FusedMoE.weight_loader
    init_signature = inspect.signature(original_init)

    def determine_expert_map(*args, **kwargs):
        active: _LayerPlan | None = getattr(_CONTEXT, "layer_plan", None)
        if active is None:
            return original_determine_expert_map(*args, **kwargs)
        bound = inspect.signature(original_determine_expert_map).bind_partial(*args, **kwargs)
        ep_size = int(bound.arguments["ep_size"])
        ep_rank = int(bound.arguments["ep_rank"])
        global_num_experts = int(bound.arguments["global_num_experts"])
        if ep_size != 4 or ep_rank != active.ep_rank:
            raise ValueError(
                f"EP4 plan requires ep_size=4/rank={active.ep_rank}, got {ep_size}/{ep_rank}"
            )
        if global_num_experts != active.num_experts:
            raise ValueError(
                f"plan has {active.num_experts} experts, layer requested {global_num_experts}"
            )
        expert_map = _rank_expert_map(
            active.expert_to_rank,
            active.expert_to_local,
            active.ep_rank,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        # Triton cannot create a zero-expert weight tensor. A map with no live
        # IDs and one unreachable storage slot preserves the zero contribution.
        local_num_experts = max(1, len(active.local_to_global))
        return_mask = bool(bound.arguments.get("return_expert_mask", False))
        num_shared = int(bound.arguments.get("num_fused_shared_experts", 0))
        expert_mask = None
        if return_mask:
            expert_mask = torch.ones(
                global_num_experts + num_shared + 1,
                dtype=torch.int32,
                device=expert_map.device,
            )
            expert_mask[-1] = 0
            expert_mask[:global_num_experts] = expert_map > -1
            if num_shared:
                expert_map = torch.cat(
                    [
                        expert_map,
                        torch.arange(
                            local_num_experts,
                            local_num_experts + num_shared,
                            dtype=torch.int32,
                            device=expert_map.device,
                        ),
                    ]
                )
        return local_num_experts, expert_map, expert_mask

    def patched_init(self, *args, **kwargs):
        bound = init_signature.bind_partial(self, *args, **kwargs)
        prefix = str(bound.arguments.get("prefix", ""))
        match = _LAYER_PATTERN.search(prefix)
        num_experts = int(bound.arguments.get("num_experts", -1))
        intermediate_size = int(bound.arguments.get("intermediate_size", -1))
        if (
            match is None
            or num_experts != plan_num_experts
            or intermediate_size != full_width
        ):
            return original_init(*bound.args, **bound.kwargs)

        model_layer_id = int(match.group(1))
        if model_layer_id not in layer_id_to_position:
            return original_init(*bound.args, **bound.kwargs)
        ep_group = get_ep_group()
        ep_rank = int(ep_group.rank_in_group)
        ep_size = int(ep_group.world_size)
        if ep_size != 4:
            raise ValueError(f"MAES EP4 plan requires four EP ranks, got {ep_size}")
        position = layer_id_to_position[model_layer_id]
        if strategy == "padded":
            original_init(*bound.args, **bound.kwargs)
            local_ids = [
                expert_id
                for expert_id in range(plan_num_experts)
                if self.expert_map is None or int(self.expert_map[expert_id]) >= 0
            ]
            active = _LayerPlan(
                plan_position=position,
                model_layer_id=model_layer_id,
                ep_rank=ep_rank,
                rank_width=full_width,
                full_width=full_width,
                num_experts=plan_num_experts,
                expert_to_rank=plan["expert_to_rank"][position],
                expert_to_local=plan["expert_to_local_id"][position],
                channel_masks=plan["intermediate_masks"][position],
                local_to_global=local_ids,
            )
        else:
            active = getattr(_CONTEXT, "layer_plan", None)
            if active is None:
                rank_width = int(plan["rank_widths"][position, ep_rank])
                local_ids = [
                    int(value) for value in plan["local_to_global"][position][ep_rank]
                ]
                active = _LayerPlan(
                    plan_position=position,
                    model_layer_id=model_layer_id,
                    ep_rank=ep_rank,
                    rank_width=rank_width,
                    full_width=full_width,
                    num_experts=plan_num_experts,
                    expert_to_rank=plan["expert_to_rank"][position],
                    expert_to_local=plan["expert_to_local_id"][position],
                    channel_masks=plan["intermediate_masks"][position],
                    local_to_global=local_ids,
                )
            else:
                local_ids = active.local_to_global
            bound.arguments["intermediate_size"] = active.rank_width
            _CONTEXT.layer_plan = active
            try:
                original_init(*bound.args, **bound.kwargs)
            finally:
                _CONTEXT.layer_plan = None
        self._maes_ep4_layer_plan = active
        print(
            "[MAES EP4] "
            f"strategy={strategy} layer={model_layer_id} rank={ep_rank} "
            f"width={active.rank_width} "
            f"local_experts={len(local_ids)} removed="
            f"{int((plan['expert_widths'][position] == 0).sum())}",
            flush=True,
        )

    def patched_weight_loader(
        self,
        param,
        loaded_weight,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ):
        active: _LayerPlan | None = getattr(self, "_maes_ep4_layer_plan", None)
        if (
            active is not None
            and (
                strategy == "padded"
                or int(active.expert_to_rank[expert_id]) == active.ep_rank
            )
        ):
            if "weight" in weight_name:
                channel_mask = active.channel_masks[expert_id]
                if strategy == "padded":
                    loaded_weight = _zero_pruned_channels(
                        loaded_weight,
                        shard_id=shard_id,
                        channel_mask=channel_mask,
                        full_width=active.full_width,
                    )
                else:
                    channel_indices = torch.where(channel_mask)[0]
                    if channel_indices.numel() != active.rank_width:
                        raise ValueError(
                            f"layer {active.model_layer_id} expert {expert_id} has "
                            f"{channel_indices.numel()} channels, expected {active.rank_width}"
                        )
                    loaded_weight = _slice_expert_weight(
                        loaded_weight,
                        shard_id=shard_id,
                        channel_indices=channel_indices,
                        full_width=active.full_width,
                    )
        return original_weight_loader(
            self,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success,
        )

    layer_module.determine_expert_map = determine_expert_map
    layer_module.FusedMoE.__init__ = patched_init
    layer_module.FusedMoE.weight_loader = patched_weight_loader
    layer_module._maes_ep4_installed = True
    print(
        f"[MAES EP4] installed strategy={strategy} plan={plan_path} model={plan['model']}",
        flush=True,
    )


def install_qwen_moe_into(model_module: ModuleType) -> None:
    """Replace Qwen's single expert module with four width-specific kernels."""
    if getattr(model_module, "_maes_ep4_installed", False):
        return
    if not os.environ.get("MAES_EP4_PLAN") or _strategy() != "multi_kernel":
        return

    import torch
    from torch import nn

    from src.vllm_ep4_plan import load_ep4_plan
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    plan = load_ep4_plan(os.environ["MAES_EP4_PLAN"])
    layer_id_to_position = {
        int(layer_id): position
        for position, layer_id in enumerate(plan["model_layer_ids"])
    }
    original_init = model_module.Qwen3MoeSparseMoeBlock.__init__

    class MultiKernelExperts(nn.Module):
        def __init__(self, vllm_config, prefix: str, model_layer_id: int):
            super().__init__()
            config = vllm_config.model_config.hf_text_config
            parallel_config = vllm_config.parallel_config
            position = layer_id_to_position[model_layer_id]
            ep_rank = int(get_ep_group().rank_in_group)
            self.groups = nn.ModuleDict()
            for width in sorted(plan["active_widths"], reverse=True):
                active = _multi_kernel_layer_plan(
                    plan, position, model_layer_id, ep_rank, int(width)
                )
                _CONTEXT.layer_plan = active
                try:
                    group = FusedMoE(
                        num_experts=config.num_experts,
                        top_k=config.num_experts_per_tok,
                        hidden_size=config.hidden_size,
                        intermediate_size=int(plan["intermediate_masks"].shape[2]),
                        reduce_results=True,
                        renormalize=config.norm_topk_prob,
                        quant_config=vllm_config.quant_config,
                        prefix=f"{prefix}.experts.groups.{width}",
                        enable_eplb=False,
                        is_sequence_parallel=parallel_config.use_sequence_parallel_moe,
                        routing_method_type=RoutingMethodType.Renormalize,
                    )
                    if not active.local_to_global:
                        with torch.no_grad():
                            for param in group.parameters():
                                param.zero_()
                    self.groups[str(width)] = group
                finally:
                    _CONTEXT.layer_plan = None

        def forward(self, hidden_states, router_logits):
            result = None
            for group in self.groups.values():
                group_output = group(
                    hidden_states=hidden_states, router_logits=router_logits
                )
                result = group_output if result is None else result + group_output
            return result

    def patched_init(self, vllm_config, prefix: str = ""):
        original_init(self, vllm_config, prefix)
        match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", prefix)
        if match is None:
            return
        model_layer_id = int(match.group(1))
        if model_layer_id not in layer_id_to_position:
            return
        self.experts = MultiKernelExperts(vllm_config, prefix, model_layer_id)
        torch.cuda.empty_cache()

    model_module.Qwen3MoeSparseMoeBlock.__init__ = patched_init
    model_module._maes_ep4_installed = True
    print("[MAES EP4] installed Qwen multi-kernel block", flush=True)


def install_qwen_vl_loader_into(model_module: ModuleType) -> None:
    """Teach the fused Qwen checkpoint loader about the four group modules."""
    if getattr(model_module, "_maes_ep4_loader_installed", False):
        return
    if not os.environ.get("MAES_EP4_PLAN") or _strategy() != "multi_kernel":
        return

    cls = model_module.Qwen3MoeLLMModel
    original = cls.load_fused_expert_weights
    original_load_weights = cls.load_weights

    def patched(
        self, name, params_dict, loaded_weight, shard_id, num_experts
    ):
        base, leaf = name.rsplit(".", 1)
        group_prefix = f"{base}.groups."
        group_names = [key for key in params_dict if key.startswith(group_prefix) and key.endswith(f".{leaf}")]
        if not group_names:
            return original(
                self, name, params_dict, loaded_weight, shard_id, num_experts
            )
        loaded_local = False
        for expert_id in range(num_experts):
            for group_name in group_names:
                param = params_dict[group_name]
                success = param.weight_loader(
                    param,
                    loaded_weight[expert_id],
                    group_name,
                    shard_id,
                    expert_id,
                    return_success=True,
                )
                loaded_local = bool(success) or loaded_local
        return loaded_local

    def patched_load_weights(self, weights):
        loaded_params = original_load_weights(self, weights)
        params_dict = dict(self.named_parameters())
        expanded = set(loaded_params)
        for name in loaded_params:
            if not name.endswith(("experts.w13_weight", "experts.w2_weight")):
                continue
            base, leaf = name.rsplit(".", 1)
            group_prefix = f"{base}.groups."
            expanded.update(
                key
                for key in params_dict
                if key.startswith(group_prefix) and key.endswith(f".{leaf}")
            )
        return expanded

    cls.load_fused_expert_weights = patched
    cls.load_weights = patched_load_weights
    model_module._maes_ep4_loader_installed = True
    print("[MAES EP4] installed Qwen fused checkpoint multi-kernel loader", flush=True)
