import argparse
import copy
import math
import os
import sys
from types import SimpleNamespace
from typing import Iterable

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.calibration.collector import collect_scores_from_moe_module
from src.calibration.helpers.hooks import register_copied_block_hooks, register_teacher_block_hook
from src.calibration.helpers.patches import (
    patch_grad_enabled_kimi_moe_infer,
    patch_qwen_fused_experts_forward,
)
from src.calibration.helpers.utils import (
    clear_block_saved_tensors,
    enable_input_grads,
    move_to_device_dtype,
    unwrap_output,
)
from src.calibration.helpers.helpers import compute_block_loss
from src.calibration.representation_distill.common import (
    ensure_dir,
    get_decoder_layer,
    resolve_hidden_start_layer,
    sort_sequence_by_position_ids,
)
from src.calibration.representation_distill.runtime.forward_from_hidden import forward_from_hidden


def _save_score_artifacts(output_path: str, accumulator, args) -> None:
    snapshot = copy.deepcopy(accumulator)
    if output_path.endswith(".pt"):
        scores_path = output_path
        ensure_dir(os.path.dirname(os.path.abspath(scores_path)))
    else:
        ensure_dir(output_path)
        scores_path = os.path.join(output_path, "scores.pt")
    payload = snapshot.build_scores_payload(args)
    payload["metadata"]["source"] = args.source
    payload["metadata"]["input_hidden_path"] = args.input_hidden_path
    if args.source == "synthetic_hidden":
        payload["metadata"]["synthetic_start_layer"] = args.start_layer
    if args.source == "teacher_cache":
        payload["metadata"]["teacher_cache_start_layer"] = args.start_layer
    payload["metadata"]["representation_distillation"] = True
    torch.save(payload, scores_path)
    print(f"[representation_distill] Saved scores: {scores_path}")


def _prepare_output_path(output_path: str) -> str:
    if output_path.endswith(".pt"):
        scores_path = output_path
        ensure_dir(os.path.dirname(os.path.abspath(scores_path)))
        return scores_path
    ensure_dir(output_path)
    return os.path.join(output_path, "scores.pt")


def _torch_load_cpu(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _set_synthetic_modality_masks(
    cnt_block,
    attn_mask: torch.Tensor,
    modality_labels: torch.Tensor | None,
    bundle,
) -> None:
    device = attn_mask.device
    if modality_labels is not None:
        flat_labels = modality_labels.view(-1).to(device)
        cnt_block.mlp.moe_text_mask = (flat_labels == 0).unsqueeze(-1)   # (B*S, 1)
        cnt_block.mlp.moe_media_mask = (flat_labels >= 1).unsqueeze(-1)  # (B*S, 1)
    else:
        token_count = attn_mask.numel()
        cnt_block.mlp.moe_text_mask = torch.zeros(token_count, 1, dtype=torch.bool, device=device)
        cnt_block.mlp.moe_media_mask = torch.zeros(token_count, 1, dtype=torch.bool, device=device)
    if hasattr(cnt_block.mlp, "moe_padding_mask"):
        cnt_block.mlp.moe_padding_mask = (~attn_mask.to(torch.bool)).view(-1, 1)


def _synthetic_block_forward(
    *,
    bundle,
    cnt_block,
    layer_idx: int,
    dataloader,
    saliency_ema: float,
    start_layer: int,
    loss_fn: str,
    dtype: torch.dtype,
    total_batches_expected: int | None = None,
    has_modality_labels: bool = False,
    has_position_ids: bool = False,
) -> float:
    model = bundle.model
    model.eval()
    teacher_block = get_decoder_layer(bundle, layer_idx)
    block_device = next(teacher_block.parameters()).device
    cnt_block = cnt_block.to(device=block_device, dtype=dtype)
    cnt_block.eval()

    teacher_state = {}
    teacher_handle = register_teacher_block_hook(teacher_block, teacher_state)
    copied_handles = register_copied_block_hooks(cnt_block)
    moe_infer_state = patch_grad_enabled_kimi_moe_infer(cnt_block, layer_idx=layer_idx)
    fused_expert_state = patch_qwen_fused_experts_forward(cnt_block)

    total_loss = 0.0
    total_second_order_sum = 0.0
    total_batches = 0
    profile_one_batch = os.getenv("PROFILE_ONE_BATCH", "0") == "1"
    device_type = block_device.type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    try:
        iterator = tqdm(
            dataloader,
            total=total_batches_expected,
            desc=f"HiddenCal L{layer_idx}",
            leave=False,
        )
        for batch_tuple in iterator:
            if has_modality_labels and has_position_ids:
                hidden_batch, attn_batch, modality_batch, position_ids_batch = batch_tuple
            elif has_modality_labels:
                hidden_batch, attn_batch, modality_batch = batch_tuple
                position_ids_batch = None
            elif has_position_ids:
                hidden_batch, attn_batch, position_ids_batch = batch_tuple
                modality_batch = None
            else:
                hidden_batch, attn_batch = batch_tuple
                modality_batch = None
                position_ids_batch = None
            teacher_state.clear()
            hidden_batch = hidden_batch.to(device=block_device, dtype=dtype)
            attn_batch = attn_batch.to(device=block_device)
            if modality_batch is not None:
                modality_batch = modality_batch.to(device=block_device)
            if position_ids_batch is not None:
                position_ids_batch = position_ids_batch.to(device=block_device)
            _set_synthetic_modality_masks(cnt_block, attn_batch, modality_batch, bundle)

            with torch.no_grad():
                forward_from_hidden(
                    bundle=bundle,
                    hidden_states=hidden_batch,
                    attention_mask=attn_batch,
                    start_layer=start_layer,
                    end_layer=layer_idx,
                    position_ids=position_ids_batch,
                    apply_final_norm=False,
                )

            if not teacher_state:
                raise RuntimeError(f"Teacher block hook did not capture layer {layer_idx} inputs.")

            in_args = enable_input_grads(move_to_device_dtype(teacher_state["in_args"], block_device, dtype))
            in_kwargs = enable_input_grads(move_to_device_dtype(teacher_state["in_kwargs"], block_device, dtype))
            teacher_target = move_to_device_dtype(unwrap_output(teacher_state["output"]), block_device, dtype)

            cnt_block.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
                pred = unwrap_output(cnt_block(*in_args, **in_kwargs))
                loss_sum, rel_l2_inv_base_mean = compute_block_loss(
                    pred=pred,
                    teacher_target=teacher_target,
                    attn_mask=attn_batch,
                    loss_fn=loss_fn,
                )

            mask_flat = attn_batch.float().view(-1)
            energy = pred.float().view(-1, pred.size(-1)).pow(2).sum(dim=-1)
            energy_loss = (energy * mask_flat).sum()
            energy_loss.backward()

            total_loss += float(loss_sum.detach().float().item())
            total_batches += 1

            second_order_sum = collect_scores_from_moe_module(
                cnt_block,
                ema=saliency_ema,
                _kwargs={
                    "use_mlp_scores": True,
                    "use_attn_scores": False,
                    "attn_mask": attn_batch,
                    "block_in_args": in_args,
                    "block_in_kwargs": in_kwargs,
                    "teacher_target": teacher_target,
                    "loss_fn": loss_fn,
                    "loss_reduction": "sum",
                    "loss_eps": 1e-6,
                    "rel_l2_inv_base_mean": rel_l2_inv_base_mean,
                    "second_order_mode": "exact",
                    "fill_zero_for_unrouted": False,
                    "autocast_dtype": dtype,
                    "autocast_device_type": device_type,
                    "layer_idx": layer_idx,
                    "debug_batch_idx": total_batches - 1,
                    "moe_text_mask": (modality_batch == 0).to(torch.bool) if modality_batch is not None else torch.zeros_like(attn_batch, dtype=torch.bool),
                    "moe_media_mask": (modality_batch >= 1).to(torch.bool) if modality_batch is not None else torch.zeros_like(attn_batch, dtype=torch.bool),
                },
            )
            total_second_order_sum += second_order_sum
            clear_block_saved_tensors(cnt_block)
            if profile_one_batch:
                break
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

    return (total_loss / max(total_batches, 1), total_second_order_sum / max(total_batches, 1))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect channel scores by continuing forward from synthetic hidden states or teacher hidden cache."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--input_hidden_path",
        type=str,
        required=True,
        help="Path to either a synthetic hidden payload or a teacher hidden cache payload.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory (writes scores.pt inside) or a path ending in .pt to write the payload file directly.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ema", type=float, default=0.9)
    parser.add_argument("--loss_fn", type=str, default="rel_l2", choices=["l2", "rel_l2", "cosine"])
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--force", "-f", action="store_true")
    return parser


def _load_hidden_payload(input_hidden_path: str):
    payload = _torch_load_cpu(input_hidden_path)
    metadata = payload.get("metadata", {})

    modality_labels = payload.get("modality_labels", None)
    position_ids = payload.get("position_ids", None)

    def _flatten_batched_tensor(name: str, tensor: torch.Tensor | None, hidden_batch_shape: tuple[int, int]):
        if tensor is None:
            return None
        if tensor.ndim < 2:
            raise ValueError(
                f"Expected `{name}` to have at least 2 dims when flattening batched hidden states, "
                f"got shape {tuple(tensor.shape)}."
            )
        if tuple(tensor.shape[:2]) != hidden_batch_shape:
            raise ValueError(
                f"Batched hidden states have prefix {hidden_batch_shape}, but `{name}` has prefix "
                f"{tuple(tensor.shape[:2])}. Cannot flatten consistently."
            )
        return tensor.reshape(hidden_batch_shape[0] * hidden_batch_shape[1], *tensor.shape[2:])

    def _normalize_hidden_layout(
        *,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        modality_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if hidden.ndim == 4:
            hidden_batch_shape = tuple(hidden.shape[:2])
            hidden = hidden.reshape(hidden_batch_shape[0] * hidden_batch_shape[1], *hidden.shape[2:])
            attention_mask = _flatten_batched_tensor("attention_mask", attention_mask, hidden_batch_shape)
            position_ids = _flatten_batched_tensor("position_ids", position_ids, hidden_batch_shape)
            modality_labels = _flatten_batched_tensor("modality_labels", modality_labels, hidden_batch_shape)

        if hidden.ndim != 3:
            raise ValueError(
                f"Expected hidden states with shape [N, L, D] or [B, N, L, D], got {tuple(hidden.shape)}."
            )

        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], dtype=torch.long)
        elif attention_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"Attention mask shape {tuple(attention_mask.shape)} must match hidden prefix "
                f"{tuple(hidden.shape[:2])}."
            )

        if position_ids is not None and hidden.shape[:2] != position_ids.shape:
            synthetic_batch_size = metadata.get("synthetic_batch_size", None)
            synthetic_size = metadata.get("synthetic_size", None)
            raise ValueError(
                f"Position ids shape {tuple(position_ids.shape)} must match hidden prefix "
                f"{tuple(hidden.shape[:2])}. This payload is likely incomplete: hidden states only contain "
                f"a sampled synthetic batch instead of the full synthetic set. "
                f"metadata.synthetic_batch_size={synthetic_batch_size}, metadata.synthetic_size={synthetic_size}."
            )
        if modality_labels is not None and modality_labels.shape != hidden.shape[:2]:
            raise ValueError(
                f"Modality labels shape {tuple(modality_labels.shape)} must match hidden prefix "
                f"{tuple(hidden.shape[:2])}."
            )
        return hidden, attention_mask, position_ids, modality_labels

    if "synthetic_hidden" in payload:
        hidden = payload["synthetic_hidden"]
        attention_mask = payload.get("attention_mask")
        hidden, attention_mask, position_ids, modality_labels = _normalize_hidden_layout(
            hidden=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality_labels=modality_labels,
        )
        if position_ids is not None:
            hidden, position_ids, modality_labels = sort_sequence_by_position_ids(
                hidden, position_ids, modality_labels,
            )
        teacher_meta = metadata["teacher_metadata"]
        return {
            "hidden": hidden,
            "attention_mask": attention_mask,
            "modality_labels": modality_labels,
            "position_ids": position_ids,
            "teacher_meta": teacher_meta,
            "start_layer": resolve_hidden_start_layer(teacher_meta),
            "source": "synthetic_hidden",
        }

    if "teacher_cache" in payload:
        hidden = payload["teacher_cache"]
        hidden, attention_mask, position_ids, modality_labels = _normalize_hidden_layout(
            hidden=hidden,
            attention_mask=None,
            position_ids=position_ids,
            modality_labels=modality_labels,
        )
        if position_ids is not None:
            hidden, position_ids, modality_labels = sort_sequence_by_position_ids(
                hidden, position_ids, modality_labels,
            )
        return {
            "hidden": hidden,
            "attention_mask": attention_mask,
            "modality_labels": modality_labels,
            "position_ids": position_ids,
            "teacher_meta": metadata,
            "start_layer": resolve_hidden_start_layer(metadata),
            "source": "teacher_cache",
        }

    if "shards" in payload:
        total_samples = int(metadata.get("total_samples", sum(int(s["num_samples"]) for s in payload["shards"])))
        return {
            "manifest": payload,
            "teacher_meta": metadata,
            "start_layer": resolve_hidden_start_layer(metadata),
            "source": "teacher_cache",
            "num_samples": total_samples,
            "is_sharded": True,
            "input_hidden_path": input_hidden_path,
        }

    raise ValueError(
        f"Unsupported hidden payload at {input_hidden_path}. "
        "Expected keys `synthetic_hidden`, `teacher_cache`, or `shards`."
    )


def _build_tensor_dataloader(
    *,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    modality_labels: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    batch_size: int,
) -> DataLoader:
    if modality_labels is not None and position_ids is not None:
        dataset = TensorDataset(hidden, attention_mask, modality_labels, position_ids)
    elif modality_labels is not None:
        dataset = TensorDataset(hidden, attention_mask, modality_labels)
    elif position_ids is not None:
        dataset = TensorDataset(hidden, attention_mask, position_ids)
    else:
        dataset = TensorDataset(hidden, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _iter_sharded_hidden_batches(hidden_payload: dict, batch_size: int) -> Iterable:
    manifest = hidden_payload["manifest"]
    cache_root = os.path.dirname(hidden_payload["input_hidden_path"])
    for shard in manifest["shards"]:
        shard_path = shard["path"]
        if not os.path.isabs(shard_path):
            shard_path = os.path.join(cache_root, shard_path)
        shard_payload = _torch_load_cpu(shard_path)
        hidden = shard_payload["teacher_cache"]
        attention_mask = torch.ones(hidden.shape[:2], dtype=torch.long)
        modality_labels = shard_payload.get("modality_labels")
        position_ids = shard_payload.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(dtype=torch.long)
            hidden, position_ids, modality_labels = sort_sequence_by_position_ids(
                hidden, position_ids, modality_labels,
            )
        shard_loader = _build_tensor_dataloader(
            hidden=hidden,
            attention_mask=attention_mask,
            modality_labels=modality_labels,
            position_ids=position_ids,
            batch_size=batch_size,
        )
        for batch_tuple in shard_loader:
            yield batch_tuple


def main() -> None:
    args = build_arg_parser().parse_args()
    from observations.common import discover_layer_structure, load_model_bundle
    from src.calibration.score_accumulator import ScoreAccumulator

    out_path = args.output_path
    if os.path.exists(out_path) and not args.force:
        print(
            f"[representation_distill] Found existing scores at {out_path}. "
            "Pass --force or -f to overwrite."
        )
        return

    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"

    bundle = load_model_bundle(
        args.model_name_or_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )
    hidden_payload = _load_hidden_payload(args.input_hidden_path)
    start_layer = hidden_payload["start_layer"]
    is_sharded = bool(hidden_payload.get("is_sharded", False))

    layer_to_num_experts, layer_to_num_channels = discover_layer_structure(bundle)
    accumulator = ScoreAccumulator(layer_to_num_experts, layer_to_num_channels)
    available_layers = [layer for layer in accumulator.layers if layer >= start_layer]
    if args.layers is not None:
        requested = []
        for layer_idx in args.layers:
            if layer_idx not in available_layers:
                raise ValueError(
                    f"Requested layer {layer_idx} cannot be reached from hidden start_layer={start_layer}. "
                    f"Available layers: {available_layers}"
                )
            if layer_idx not in requested:
                requested.append(layer_idx)
        target_layers = requested
    else:
        target_layers = available_layers
    if not target_layers:
        raise ValueError(
            f"No target MoE layers remain after hidden start_layer={start_layer}."
        )
    prepared_output_path = _prepare_output_path(args.output_path)

    if is_sharded:
        num_samples = int(hidden_payload["num_samples"])
        has_modality_labels = True
        has_position_ids = True
    else:
        hidden = hidden_payload["hidden"]
        attention_mask = hidden_payload["attention_mask"]
        modality_labels = hidden_payload.get("modality_labels", None)
        position_ids = hidden_payload.get("position_ids", None)
        num_samples = int(hidden.shape[0])
        has_modality_labels = modality_labels is not None
        has_position_ids = position_ids is not None
    payload_args = SimpleNamespace(
        loss_fn=args.loss_fn,
        num_samples=num_samples,
        batch_size=args.batch_size,
        dataset=hidden_payload["source"],
        start_idx=0,
        model_name_or_path=args.model_name_or_path,
        subset_seed=None,
        ema=args.ema,
        fill_zero_for_unrouted=False,
        source=hidden_payload["source"],
        input_hidden_path=args.input_hidden_path,
        start_layer=start_layer,
    )

    for layer_idx in target_layers:
        teacher_block = get_decoder_layer(bundle, layer_idx)
        cnt_block = copy.deepcopy(teacher_block)
        block_dtype = next(teacher_block.parameters()).dtype
        total_batches_expected = math.ceil(num_samples / args.batch_size)
        if is_sharded:
            loader = _iter_sharded_hidden_batches(hidden_payload, args.batch_size)
        else:
            loader = _build_tensor_dataloader(
                hidden=hidden,
                attention_mask=attention_mask,
                modality_labels=modality_labels,
                position_ids=position_ids,
                batch_size=args.batch_size,
            )
        layer_loss, layer_second_order_sum = _synthetic_block_forward(
            bundle=bundle,
            cnt_block=cnt_block,
            layer_idx=layer_idx,
            dataloader=loader,
            saliency_ema=args.ema,
            start_layer=start_layer,
            loss_fn=args.loss_fn,
            dtype=block_dtype,
            total_batches_expected=total_batches_expected,
            has_modality_labels=has_modality_labels,
            has_position_ids=has_position_ids,
        )
        accumulator.layerwise_loss[layer_idx] = float(layer_loss)
        accumulator.layerwise_second_order_sum[layer_idx] = float(layer_second_order_sum)
        accumulator.absorb_layer_scores(layer_idx, cnt_block)
        print(f"[representation_distill] Layer {layer_idx}: layer loss={layer_loss:.6e}, layer second_order_sum={layer_second_order_sum:.6e}")
        _save_score_artifacts(prepared_output_path, accumulator, payload_args)

    _save_score_artifacts(prepared_output_path, accumulator, payload_args)


if __name__ == "__main__":
    main()
