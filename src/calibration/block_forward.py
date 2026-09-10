from typing import Any, Dict

import torch
from torch import nn
from tqdm import tqdm

from observations.common import (
    _resolve_multimodal_media_token_ids,
    filter_model_forward_inputs,
    move_inputs_to_model_device,
    prepare_inputs,
)
from src.calibration.collector import collect_scores_from_moe_module
from src.calibration.representation_distill.common import prepare_raw_batch_inputs

from .helpers.helpers import compute_block_loss, set_block_modality_masks, teacher_blocks
from .helpers.hooks import register_copied_block_hooks, register_teacher_block_hook
from .helpers.patches import (
    patch_grad_enabled_kimi_moe_infer,
    patch_qwen_fused_experts_forward,
)
from .helpers.utils import (
    clear_block_saved_tensors,
    enable_input_grads,
    move_to_device_dtype,
    unwrap_output,
)

__all__ = [
    "block_forward",
]


def _build_fixed_score_mask(bundle, inputs, tokens_per_sample: int | None):
    attention_mask = inputs["attention_mask"].to(torch.bool)
    if tokens_per_sample is None:
        return attention_mask
    if tokens_per_sample <= 0:
        raise ValueError(
            f"score_tokens_per_sample must be positive, got {tokens_per_sample}"
        )

    input_ids = inputs.get("input_ids", None)
    media_token_ids = _resolve_multimodal_media_token_ids(bundle)
    keep_masks = []

    def _uniform_positions(positions: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return positions[:0]
        if count >= positions.numel():
            return positions
        offsets = torch.linspace(
            0,
            positions.numel() - 1,
            steps=count,
            device=positions.device,
        ).round().long()
        return positions.index_select(0, offsets)

    for row_idx in range(attention_mask.shape[0]):
        valid_count = int(attention_mask[row_idx].sum().item())
        if valid_count < tokens_per_sample:
            raise RuntimeError(
                f"Manifest sample has only {valid_count} model tokens during collection; "
                f"expected at least {tokens_per_sample}. The processor/model may differ "
                "from the one used to build the manifest."
            )
        if input_ids is None:
            active = attention_mask[row_idx].nonzero(as_tuple=True)[0]
            keep = torch.zeros_like(attention_mask[row_idx])
            keep[_uniform_positions(active, tokens_per_sample)] = True
        else:
            valid = attention_mask[row_idx]
            active = valid.nonzero(as_tuple=True)[0]
            if media_token_ids:
                media_ids = torch.tensor(
                    media_token_ids,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                visual = active[
                    torch.isin(input_ids[row_idx].index_select(0, active), media_ids)
                ]
            else:
                visual = active[:0]
            visual_lookup = torch.zeros_like(valid)
            visual_lookup[visual] = True
            text = active[~visual_lookup[active]]

            visual_quota = int(
                round(tokens_per_sample * visual.numel() / max(active.numel(), 1))
            )
            visual_quota = min(visual_quota, visual.numel())
            text_quota = min(tokens_per_sample - visual_quota, text.numel())
            visual_quota = min(tokens_per_sample - text_quota, visual.numel())

            keep = torch.zeros_like(valid)
            keep[_uniform_positions(visual, visual_quota)] = True
            keep[_uniform_positions(text, text_quota)] = True
        if int(keep.sum().item()) != tokens_per_sample:
            raise RuntimeError(
                f"Failed to construct an exact {tokens_per_sample}-token score mask; "
                f"got {int(keep.sum().item())}."
            )
        keep_masks.append(keep)
    return torch.stack(keep_masks, dim=0)


def block_forward(
    bundle,
    cnt_block: nn.Module,
    layer_idx: int,
    dataloader,
    dataset_name: str,
    saliency_ema: float,
    score_aggregation: str = "mean",
    fill_zero_for_unrouted: bool = False,
    loss_fn: str = "rel_l2",
    second_order_mode: str = "exact",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
    raw_samples: bool = False,
    score_tokens_per_sample: int | None = None,
):
    model = bundle.model
    model.eval()
    teacher_block = list(teacher_blocks(bundle))[layer_idx]
    block_device = next(teacher_block.parameters()).device
    cnt_block = cnt_block.to(device=block_device, dtype=dtype)
    cnt_block.eval()

    teacher_state: Dict[str, Any] = {}
    teacher_handle = register_teacher_block_hook(teacher_block, teacher_state)
    copied_handles = register_copied_block_hooks(cnt_block)
    moe_infer_state = patch_grad_enabled_kimi_moe_infer(cnt_block, layer_idx=layer_idx)
    fused_expert_state = patch_qwen_fused_experts_forward(cnt_block)

    total_loss = 0.0
    total_batches = 0
    device_type = block_device.type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    total_second_order_sum = 0.0
    
    try:
        iterator = tqdm(dataloader, desc=f"Calibrating L{layer_idx}", disable=not verbose, leave=False)
        for batch in iterator:
            teacher_state.clear()
            inputs = (
                prepare_raw_batch_inputs(bundle, batch)
                if raw_samples
                else prepare_inputs(bundle, batch, dataset_name)
            )
            inputs = move_inputs_to_model_device(model, inputs)
            attn_mask = inputs["attention_mask"].to(block_device)
            score_mask = _build_fixed_score_mask(
                bundle, inputs, score_tokens_per_sample
            ).to(block_device)
            input_ids = inputs.get("input_ids", None)
            moe_text_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
            moe_media_mask = torch.zeros_like(attn_mask, dtype=torch.bool)

            # Ensure copied-block moe_infer can access per-token modality masks.
            # Unlike full-model forward, block-only calibration does not automatically
            # refresh these fields on the copied block.
            # 构造 block 级别的 modality mask（用于 channel 二阶计算）
            if input_ids is not None and hasattr(cnt_block, "mlp"):
                flat_text_mask, flat_media_mask = set_block_modality_masks(
                    bundle, cnt_block, input_ids, attn_mask
                )
                moe_text_mask = flat_text_mask.view_as(attn_mask)
                moe_media_mask = flat_media_mask.view_as(attn_mask)
            if hasattr(cnt_block, "mlp"):
                cnt_block.mlp.moe_score_mask = score_mask.view(-1, 1)

            with torch.no_grad():
                model(**filter_model_forward_inputs(model, inputs), use_cache=False, return_dict=True)

            if not teacher_state:
                raise RuntimeError(f"Teacher block hook did not capture layer {layer_idx} inputs.")

            in_args = enable_input_grads(
                move_to_device_dtype(teacher_state["in_args"], block_device, dtype)
            )
            in_kwargs = enable_input_grads(
                move_to_device_dtype(teacher_state["in_kwargs"], block_device, dtype)
            )
            teacher_target = unwrap_output(teacher_state["output"])
            teacher_target = move_to_device_dtype(teacher_target, block_device, dtype)

            cnt_block.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
                pred = unwrap_output(cnt_block(*in_args, **in_kwargs))
                loss_sum, rel_l2_inv_base_mean = compute_block_loss(
                    pred=pred,
                    teacher_target=teacher_target,
                    attn_mask=score_mask,
                    loss_fn=loss_fn,
                )

            # For first-order gradient scores, backward through output-energy
            # instead of reconstruction loss. Since cnt_block has identical
            # weights to the teacher, reconstruction loss ≈ 0 and produces
            # trivial gradients. Output-energy ||pred||² gives non-zero
            # gradients reflecting each channel's contribution to the block
            # output. This does NOT affect second-order scores (computed
            # independently from saved activations, not from backward).
            mask_flat = score_mask.float().view(-1)
            energy = pred.float().view(-1, pred.size(-1)).pow(2).sum(dim=-1)
            energy_loss = (energy * mask_flat).sum()
            energy_loss.backward()

            total_loss += float(loss_sum.detach().float().item())
            total_batches += 1

            second_order_sum = collect_scores_from_moe_module(
                cnt_block,
                ema=saliency_ema,
                aggregation=score_aggregation,
                _kwargs={
                    "use_mlp_scores": True,
                    "use_attn_scores": False,
                    "attn_mask": score_mask,
                    "block_in_args": in_args,
                    "block_in_kwargs": in_kwargs,
                    "teacher_target": teacher_target,
                    "loss_fn": loss_fn,
                    "loss_reduction": "sum",
                    "loss_eps": 1e-6,
                    "rel_l2_inv_base_mean": rel_l2_inv_base_mean,
                    "second_order_mode": second_order_mode,
                    "fill_zero_for_unrouted": fill_zero_for_unrouted,
                    "autocast_dtype": dtype,
                    "autocast_device_type": device_type,
                    "layer_idx": layer_idx,
                    "debug_batch_idx": total_batches - 1,
                    "moe_text_mask": moe_text_mask.view_as(score_mask),
                    "moe_media_mask": moe_media_mask.view_as(score_mask),
                },
            )
            total_second_order_sum += second_order_sum
            clear_block_saved_tensors(cnt_block)
    finally:
        teacher_handle.remove()
        for handle in copied_handles:
            handle.remove()
        if moe_infer_state is not None:
            mlp, original_moe_infer = moe_infer_state
            mlp.moe_infer = original_moe_infer
        if fused_expert_state is not None:
            experts, original_forward = fused_expert_state
            experts.forward = original_forward
        clear_block_saved_tensors(cnt_block)

    return total_loss / max(total_batches, 1), total_second_order_sum / max(total_batches, 1)
