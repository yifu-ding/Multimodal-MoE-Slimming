import inspect
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch import nn
from tqdm import tqdm

from src.calibration.common import (
    _resolve_multimodal_media_token_ids,
    filter_model_forward_inputs,
    move_inputs_to_model_device,
    prepare_inputs,
)
from src.calibration.collector import collect_scores_from_moe_module
from src.calibration.collector.loop_2_helpers import (
    _restore_patched_expert,
    patch_expert_output_alpha_vector,
    suspend_tensor_saving,
)
from src.calibration.representation_distill.common import prepare_raw_batch_inputs

from .helpers.helpers import compute_block_loss, set_block_modality_masks, teacher_blocks
from .helpers.hooks import register_copied_block_hooks, register_teacher_block_hook
from .helpers.patches import (
    patch_grad_enabled_kimi_moe_infer,
    patch_qwen_fused_experts_forward,
    patch_internvl_qwen3_moe_forward,
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


def _teacher_forward_inputs(model, inputs):
    """Avoid materializing full-sequence logits when only block hooks are used."""
    forward_inputs = filter_model_forward_inputs(model, inputs)
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return forward_inputs
    if "logits_to_keep" in parameters:
        forward_inputs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        forward_inputs["num_logits_to_keep"] = 1
    return forward_inputs


def _build_fixed_score_mask(bundle, inputs, tokens_per_sample):
    attention_mask = inputs["attention_mask"].to(torch.bool)
    if tokens_per_sample is None:
        return attention_mask
    if isinstance(tokens_per_sample, int):
        row_token_counts = [tokens_per_sample] * attention_mask.shape[0]
    elif isinstance(tokens_per_sample, torch.Tensor):
        row_token_counts = [int(value) for value in tokens_per_sample.view(-1).tolist()]
    else:
        row_token_counts = [int(value) for value in tokens_per_sample]
    if len(row_token_counts) != attention_mask.shape[0]:
        raise ValueError(
            f"Expected {attention_mask.shape[0]} score-token quotas, got "
            f"{len(row_token_counts)}."
        )
    if any(value <= 0 for value in row_token_counts):
        raise ValueError(f"Score-token quotas must be positive, got {row_token_counts}.")

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
        row_token_count = row_token_counts[row_idx]
        valid_count = int(attention_mask[row_idx].sum().item())
        if valid_count < row_token_count:
            raise RuntimeError(
                f"Manifest sample has only {valid_count} model tokens during collection; "
                f"expected at least {row_token_count}. The processor/model may differ "
                "from the one used to build the manifest."
            )
        if input_ids is None:
            active = attention_mask[row_idx].nonzero(as_tuple=True)[0]
            keep = torch.zeros_like(attention_mask[row_idx])
            keep[_uniform_positions(active, row_token_count)] = True
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
                round(row_token_count * visual.numel() / max(active.numel(), 1))
            )
            visual_quota = min(visual_quota, visual.numel())
            text_quota = min(row_token_count - visual_quota, text.numel())
            visual_quota = min(row_token_count - text_quota, visual.numel())

            keep = torch.zeros_like(valid)
            keep[_uniform_positions(visual, visual_quota)] = True
            keep[_uniform_positions(text, text_quota)] = True
        if int(keep.sum().item()) != row_token_count:
            raise RuntimeError(
                f"Failed to construct an exact {row_token_count}-token score mask; "
                f"got {int(keep.sum().item())}."
            )
        keep_masks.append(keep)
    return torch.stack(keep_masks, dim=0)


def _accumulate_hessian_probe(state: dict, batch: dict) -> None:
    hessian = batch["hessian"].detach().cpu().float()
    gradient = batch["gradient"].detach().cpu().float()
    active = batch["active_mask"].detach().cpu().bool()
    if state["hessian_sum"] is None:
        num_experts = int(gradient.numel())
        state["hessian_sum"] = torch.zeros((num_experts, num_experts), dtype=torch.float64)
        state["gradient_sum"] = torch.zeros(num_experts, dtype=torch.float64)
        state["active_batch_counts"] = torch.zeros(num_experts, dtype=torch.int64)
        state["coactive_batch_counts"] = torch.zeros(
            (num_experts, num_experts), dtype=torch.int64
        )
    state["hessian_sum"].add_(hessian.double())
    state["gradient_sum"].add_(gradient.double())
    state["base_loss_sum"] += float(batch["base_loss"])
    state["score_tokens"] += int(batch["score_tokens"])
    state["num_batches"] += 1
    state["active_batch_counts"].add_(active.to(torch.int64))
    state["coactive_batch_counts"].add_(
        active[:, None].logical_and(active[None, :]).to(torch.int64)
    )

    ablation_losses = batch.get("ablation_losses")
    if ablation_losses is not None:
        for name, value in ablation_losses.items():
            state["ablation_loss_sums"][name] = (
                state["ablation_loss_sums"].get(name, 0.0) + float(value)
            )


def _save_hessian_probe(
    state: dict,
    config: dict,
    *,
    layer_idx: int,
    loss_fn: str,
) -> None:
    if state["hessian_sum"] is None or state["score_tokens"] <= 0:
        raise RuntimeError(f"Hessian probe for layer {layer_idx} collected no data.")

    token_scale = float(state["score_tokens"])
    hessian_sum = state["hessian_sum"].float()
    gradient_sum = state["gradient_sum"].float()
    hessian_per_token = hessian_sum / token_scale
    gradient_per_token = gradient_sum / token_scale
    asymmetry = hessian_per_token - hessian_per_token.T
    max_hessian = float(hessian_per_token.abs().max().item())
    max_asymmetry = float(asymmetry.abs().max().item())

    pair = config.get("pair")
    pair_results = None
    if pair is not None:
        e, f = (int(pair[0]), int(pair[1]))
        num_experts = int(gradient_per_token.numel())
        if e == f or min(e, f) < 0 or max(e, f) >= num_experts:
            raise ValueError(
                f"Invalid Hessian probe pair {(e, f)} for {num_experts} experts."
            )
        h2 = hessian_per_token[[e, f]][:, [e, f]]
        g2 = gradient_per_token[[e, f]]
        removal_steps = {
            "remove_e": torch.tensor([-1.0, 0.0]),
            "remove_f": torch.tensor([0.0, -1.0]),
            "remove_ef": torch.tensor([-1.0, -1.0]),
        }
        first_order = {
            name: float(torch.dot(g2, step).item())
            for name, step in removal_steps.items()
        }
        exact_quadratic = {
            name: float((torch.dot(g2, step) + 0.5 * step @ h2 @ step).item())
            for name, step in removal_steps.items()
        }
        direct = None
        if state["ablation_loss_sums"]:
            base_loss_sum = state["base_loss_sum"]
            direct = {
                name: float((value - base_loss_sum) / token_scale)
                for name, value in state["ablation_loss_sums"].items()
            }
        pair_results = {
            "experts": (e, f),
            "hessian": h2,
            "gradient": g2,
            "first_order_delta": first_order,
            "exact_quadratic_delta": exact_quadratic,
            "direct_ablation_delta": direct,
        }

    payload = {
        "schema_version": 1,
        "layer_idx": int(layer_idx),
        "loss_fn": loss_fn,
        "normalization": "sum over calibration batches, divided by score-token count",
        "num_batches": int(state["num_batches"]),
        "num_score_tokens": int(state["score_tokens"]),
        "hessian_sum": hessian_sum,
        "hessian_per_token": hessian_per_token,
        "gradient_sum": gradient_sum,
        "gradient_per_token": gradient_per_token,
        "base_loss_sum": float(state["base_loss_sum"]),
        "base_loss_per_token": float(state["base_loss_sum"] / token_scale),
        "active_batch_counts": state["active_batch_counts"],
        "coactive_batch_counts": state["coactive_batch_counts"],
        "max_abs_hessian_asymmetry_per_token": max_asymmetry,
        "relative_hessian_asymmetry": max_asymmetry / max(max_hessian, 1e-30),
        "pair_results": pair_results,
        "metadata": dict(config.get("metadata", {})),
    }

    output_path = Path(config["out_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    print(
        f"[hessian-probe] Saved layer {layer_idx} full Hessian from "
        f"{state['num_batches']} batches / {state['score_tokens']} score tokens to "
        f"{output_path}",
        flush=True,
    )


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
    score_tokens_per_sample: int | Sequence[int] | torch.Tensor | None = None,
    hessian_probe_config: dict | None = None,
    layerwise_beta: float = 0.95,
):
    model = bundle.model
    model.eval()
    teacher_block = list(teacher_blocks(bundle))[layer_idx]
    block_device = next(teacher_block.parameters()).device
    cnt_block = cnt_block.to(device=block_device, dtype=dtype)
    cnt_block.eval()
    if not 0.0 <= layerwise_beta < 1.0:
        raise ValueError(
            f"layerwise_beta must be in [0, 1), got {layerwise_beta}."
        )

    teacher_state: Dict[str, Any] = {}
    teacher_handle = register_teacher_block_hook(teacher_block, teacher_state)
    copied_handles = register_copied_block_hooks(cnt_block)
    moe_infer_state = patch_grad_enabled_kimi_moe_infer(cnt_block, layer_idx=layer_idx)
    fused_expert_state = patch_qwen_fused_experts_forward(cnt_block)
    internvl_expert_state = patch_internvl_qwen3_moe_forward(cnt_block)

    total_loss = 0.0
    total_batches = 0
    device_type = block_device.type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    total_second_order_sum = 0.0
    probe_state = None
    if hessian_probe_config is not None:
        if loss_fn not in {"l2", "rel_l2"}:
            raise ValueError(
                "Exact Hessian landscape probing requires loss_fn='l2' or 'rel_l2'."
            )
        if hessian_probe_config.get("validate", False) and hessian_probe_config.get("pair") is None:
            raise ValueError("Hessian probe validation requires an explicit expert pair.")
        probe_state = {
            "hessian_sum": None,
            "gradient_sum": None,
            "base_loss_sum": 0.0,
            "score_tokens": 0,
            "num_batches": 0,
            "active_batch_counts": None,
            "coactive_batch_counts": None,
            "ablation_loss_sums": {},
        }
    
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
            batch_score_token_counts = score_tokens_per_sample
            if raw_samples and batch and all("score_token_count" in sample for sample in batch):
                batch_score_token_counts = [
                    int(sample["score_token_count"]) for sample in batch
                ]
            score_mask = _build_fixed_score_mask(
                bundle, inputs, batch_score_token_counts
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
                model(
                    **_teacher_forward_inputs(model, inputs),
                    use_cache=False,
                    return_dict=True,
                )

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

            total_batches += 1

            collector_kwargs = {
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
                "hessian_probe_enabled": probe_state is not None,
                "hessian_probe_pair": None
                if hessian_probe_config is None
                else hessian_probe_config.get("pair"),
                "hessian_probe_validate": False
                if hessian_probe_config is None
                else bool(hessian_probe_config.get("validate", False)),
            }
            second_order_sum = collect_scores_from_moe_module(
                cnt_block,
                ema=saliency_ema,
                aggregation=score_aggregation,
                _kwargs=collector_kwargs,
            )
            if probe_state is not None:
                probe_batch = collector_kwargs.get("_hessian_probe_batch")
                if probe_batch is None:
                    raise RuntimeError(
                        f"Hessian probe data was not returned for layer {layer_idx}, "
                        f"batch {total_batches - 1}."
                    )
                _accumulate_hessian_probe(probe_state, probe_batch)
            total_second_order_sum += second_order_sum

            experts = getattr(getattr(cnt_block, "mlp", None), "experts", None)
            if experts is None:
                raise RuntimeError(f"Layer {layer_idx} has no expert container.")
            num_experts = getattr(experts, "num_experts", None)
            if num_experts is None:
                num_experts = len(experts)
            beta = torch.full(
                (int(num_experts),),
                float(layerwise_beta),
                device=block_device,
                dtype=torch.float32,
            )
            beta_state = patch_expert_output_alpha_vector(experts, beta)
            try:
                with suspend_tensor_saving(cnt_block), torch.no_grad():
                    with torch.autocast(
                        device_type=device_type,
                        dtype=dtype,
                        enabled=autocast_enabled,
                    ):
                        perturbed_pred = unwrap_output(cnt_block(*in_args, **in_kwargs))
                        perturbed_loss_sum, _ = compute_block_loss(
                            pred=perturbed_pred,
                            teacher_target=teacher_target,
                            attn_mask=score_mask,
                            loss_fn=loss_fn,
                        )
                total_loss += float(perturbed_loss_sum.detach().float().item())
            finally:
                _restore_patched_expert(beta_state)
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
        if internvl_expert_state is not None:
            mlp, original_forward = internvl_expert_state
            mlp.forward = original_forward
        clear_block_saved_tensors(cnt_block)

    if probe_state is not None:
        _save_hessian_probe(
            probe_state,
            hessian_probe_config,
            layer_idx=layer_idx,
            loss_fn=loss_fn,
        )

    return total_loss / max(total_batches, 1), total_second_order_sum / max(total_batches, 1)
