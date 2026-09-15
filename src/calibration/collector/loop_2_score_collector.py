import os
import time
import torch
from .utils import *
from .loop_2_helpers import *
from src.calibration.helpers.score_namespace import ACTIVE_CHANNEL_METRICS as CHANNEL_METRICS


def _format_cuda_mem_stats() -> str:
    if not torch.cuda.is_available():
        return "cuda_mem=unavailable"
    device = torch.cuda.current_device()
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return (
        f"cuda_mem_allocated={allocated_mb:.1f}MB "
        f"cuda_mem_reserved={reserved_mb:.1f}MB "
        f"cuda_mem_max_allocated={max_allocated_mb:.1f}MB"
    )


def loop_2_score_collector(
    expert_records,
    *,
    cnt_block,
    experts,
    is_fused: bool,
    fused_metric_stacks: dict,
    ema: float,
    aggregation: str = "mean",
    _kwargs: dict = None,
):
    profile_second_order = os.getenv("SECOND_ORDER_PROFILE", "0") == "1"
    second_order_impl = os.getenv("SECOND_ORDER_IMPL", "vectorized").strip().lower()
    if (
        bool((_kwargs or {}).get("hessian_probe_enabled", False))
        and second_order_impl != "vectorized"
    ):
        raise ValueError("Hessian probing requires SECOND_ORDER_IMPL=vectorized.")
    second_exact_attr_all = None
    second_attr_compute_ms = 0.0

    t_second_order_start = time.perf_counter() if profile_second_order else None
    if second_order_impl == "vectorized":
        second_exact_attr_all = compute_expert_second_order_batched(
            cnt_block=cnt_block,
            experts=experts,
            has_activation_mask=[record["has_activation"] for record in expert_records],
            _kwargs=_kwargs,
        )
        if profile_second_order:
            second_attr_compute_ms = (time.perf_counter() - t_second_order_start) * 1000.0
    elif second_order_impl != "legacy":
        raise ValueError(f"Unsupported SECOND_ORDER_IMPL: {second_order_impl}")

    second_order_sum = 0.0
    
    for record in expert_records:
        expert_idx = record["expert_idx"]
        expert = record["expert"]
        # 该 expert 本轮未被激活
        # if not record["has_activation"]:
        #     if is_fused:
        #         device = experts.gate_up_proj.device
        #         fused_metric_stacks.setdefault("second_attr_fillzero", {})[expert_idx] = \
        #             torch.zeros((), dtype=torch.float32, device=device)
        #     else:
        #         safe_add_with_ema(expert, ema, 0.0, "second_attr_fillzero")
        #     continue

        expert_proxy = make_fused_expert_proxy(experts, expert_idx) if is_fused else expert

        if second_order_impl == "vectorized":
            second_exact_attr = second_exact_attr_all[expert_idx]
        else:
            t_legacy_start = time.perf_counter() if profile_second_order else None
            second_exact_attr = compute_expert_second_order(
                cnt_block=cnt_block,
                expert=expert_proxy,
                _kwargs=_kwargs,
            )
            if profile_second_order:
                second_attr_compute_ms += (time.perf_counter() - t_legacy_start) * 1000.0
        second_order_sum += second_exact_attr.detach().float().item()
        if is_fused:
            if not record["has_activation"]:
                # 如果没有被激活，则fillzero累加，但不计入second_attr
                fused_metric_stacks.setdefault("second_attr_fillzero", {})[expert_idx] = \
                second_exact_attr.detach()
            else:
                # 如果被激活过，则两者都累积
                fused_metric_stacks.setdefault("second_attr", {})[expert_idx] = \
                    second_exact_attr.detach()
                fused_metric_stacks.setdefault("second_attr_fillzero", {})[expert_idx] = \
                    second_exact_attr.detach()
        else:
            if not record["has_activation"]:
                # 如果没有被激活，则fillzero累加，但不计入second_attr
                safe_update_running_stat(
                    expert,
                    second_exact_attr,
                    key="second_attr_fillzero",
                    aggregation=aggregation,
                    ema=ema,
                )
            else:
                # 如果被激活过，则两者都累积
                safe_update_running_stat(
                    expert,
                    second_exact_attr,
                    key="second_attr",
                    aggregation=aggregation,
                    ema=ema,
                )
                safe_update_running_stat(
                    expert,
                    second_exact_attr,
                    key="second_attr_fillzero",
                    aggregation=aggregation,
                    ema=ema,
                )

        if not record["has_activation"]:
            # 如果没有被激活，则channelwise second_order_exact不累积
            continue
        
        # channelwise second_order at down projection layer
        if "down_second_order_exact" in CHANNEL_METRICS:
            down_second_order_exact = compute_down_second_order(
                cnt_block=cnt_block,
                expert=expert_proxy,
                _kwargs=_kwargs,
            )
            text_mask = None if _kwargs is None else _kwargs.get("moe_text_mask", None)
            image_mask = None if _kwargs is None else _kwargs.get("moe_media_mask", None)
            down_second_order_exact_text = compute_down_second_order(
                cnt_block=cnt_block,
                expert=expert_proxy,
                _kwargs=_kwargs,
                modality_mask=text_mask,
            )
            down_second_order_exact_visual = compute_down_second_order(
                cnt_block=cnt_block,
                expert=expert_proxy,
                _kwargs=_kwargs,
                modality_mask=image_mask,
            )
            if is_fused:
                fused_metric_stacks.setdefault("down_second_order_exact", {})[expert_idx] = \
                    down_second_order_exact.detach()
                if down_second_order_exact_text is not None:
                    fused_metric_stacks.setdefault("down_second_order_exact_text", {})[expert_idx] = \
                        down_second_order_exact_text.detach()
                if down_second_order_exact_visual is not None:
                    fused_metric_stacks.setdefault("down_second_order_exact_visual", {})[expert_idx] = \
                        down_second_order_exact_visual.detach()
            else:
                safe_update_running_stat(
                    expert,
                    down_second_order_exact,
                    key="down_second_order_exact",
                    aggregation=aggregation,
                    ema=ema,
                )
                if down_second_order_exact_text is not None:
                    safe_update_running_stat(
                        expert,
                        down_second_order_exact_text,
                        key="down_second_order_exact_text",
                        aggregation=aggregation,
                        ema=ema,
                    )
                if down_second_order_exact_visual is not None:
                    safe_update_running_stat(
                        expert,
                        down_second_order_exact_visual,
                        key="down_second_order_exact_visual",
                        aggregation=aggregation,
                        ema=ema,
                    )

        '''
        true_ablate = compute_true_ablate_attr(
            cnt_block=cnt_block,
            expert=expert_proxy,
            _kwargs=_kwargs,
        )
        if true_ablate is not None:
            if is_fused:
                fused_metric_stacks.setdefault("true_ablate", []).append(
                    true_ablate.detach()
                )
            else:
                safe_add_with_ema(expert, ema, true_ablate, "true_ablate")
        '''
        
    if profile_second_order:
        batch_idx = None if _kwargs is None else _kwargs.get("debug_batch_idx")
        layer_idx = None if _kwargs is None else _kwargs.get("layer_idx")
        print(
            f"[loop_2 second_order profile] impl={second_order_impl} "
            f"layer={layer_idx} batch={batch_idx} "
            f"compute_second_attr={second_attr_compute_ms:.3f}ms "
            f"{_format_cuda_mem_stats()}",
            flush=True,
        )

    return second_order_sum
