import torch
from .utils import *
from .loop_2_helpers import *


def loop_2_score_collector(
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
        expert_idx = record["expert_idx"]
        expert = record["expert"]
        # 该 expert 本轮未被激活，则 second_attr_fillzero 需要补全
        if not record["has_activation"]:
            if is_fused:
                device = experts.gate_up_proj.device
                fused_metric_stacks.setdefault("second_attr_fillzero", {})[expert_idx] = \
                    torch.zeros((), dtype=torch.float32, device=device)
            else:
                safe_add_with_ema(expert, ema, 0.0, "second_attr_fillzero")
            continue

        expert_proxy = make_fused_expert_proxy(experts, expert_idx) if is_fused else expert

        second_exact_attr = compute_expert_second_order(
            cnt_block=cnt_block,
            expert=expert_proxy,
            _kwargs=_kwargs,
        )
        if is_fused:
            fused_metric_stacks.setdefault("second_attr", {})[expert_idx] = \
                second_exact_attr.detach()
            fused_metric_stacks.setdefault("second_attr_fillzero", {})[expert_idx] = \
                second_exact_attr.detach()
        else:
            safe_add_with_ema(expert, ema, second_exact_attr, "second_attr")
            safe_add_with_ema(expert, ema, second_exact_attr, "second_attr_fillzero")

        # channelwise second_order at down projection layer
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
        down_second_order_exact_image = compute_down_second_order(
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
            if down_second_order_exact_image is not None:
                fused_metric_stacks.setdefault("down_second_order_exact_image", {})[expert_idx] = \
                    down_second_order_exact_image.detach()
                # Keep visual alias for naming consistency with existing metric keys.
                fused_metric_stacks.setdefault("down_second_order_exact_visual", {})[expert_idx] = \
                    down_second_order_exact_image.detach()
        else:
            safe_add_with_ema(expert, ema, down_second_order_exact, "down_second_order_exact")
            if down_second_order_exact_text is not None:
                safe_add_with_ema(expert, ema, down_second_order_exact_text, "down_second_order_exact_text")
            if down_second_order_exact_image is not None:
                safe_add_with_ema(expert, ema, down_second_order_exact_image, "down_second_order_exact_image")
                # Keep visual alias for naming consistency with existing metric keys.
                safe_add_with_ema(expert, ema, down_second_order_exact_image, "down_second_order_exact_visual")

        # true_ablate = compute_true_ablate_attr(
        #     cnt_block=cnt_block,
        #     expert=expert_proxy,
        #     _kwargs=_kwargs,
        # )
        # if true_ablate is not None:
        #     if is_fused:
        #         fused_metric_stacks.setdefault("true_ablate", []).append(
        #             true_ablate.detach()
        #         )
        #     else:
        #         safe_add_with_ema(expert, ema, true_ablate, "true_ablate")
