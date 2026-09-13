"""Collect single-expert Hessian/energy/ablation validation data for selected layers."""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
import types
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from observations.common import (
    discover_layer_structure,
    filter_model_forward_inputs,
    load_model_bundle,
    move_inputs_to_model_device,
)
from src.calibration.block_forward import _build_fixed_score_mask
from src.calibration.collect_scores_main import _identity_collate, _load_selection_manifest
from src.calibration.collector.loop_2_helpers import (
    _restore_patched_expert,
    patch_expert_output_alpha_vector,
    suspend_tensor_saving,
)
from src.calibration.helpers.helpers import compute_block_loss, set_block_modality_masks, teacher_block
from src.calibration.helpers.hooks import register_teacher_block_hook
from src.calibration.helpers.patches import patch_qwen_fused_experts_forward
from src.calibration.helpers.utils import clear_block_saved_tensors, move_to_device_dtype, unwrap_output
from src.calibration.representation_distill.common import prepare_raw_batch_inputs
from src.calibration.representation_distill.runtime.dump_original_data import ManifestRawDataset


DEFAULT_BETAS = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
QUANTILES = (("low", 0.1), ("medium", 0.5), ("high", 0.9))


class RouteRecorder:
    """Assert that every beta forward receives identical router tensors."""

    def __init__(self, experts):
        self.experts = experts
        self.forward = experts.forward
        self.reference = None
        self.calls = 0

        def recording_forward(
            module,
            hidden_states,
            router_indices,
            routing_weights,
            _base_forward=self.forward,
            _recorder=self,
        ):
            current = (
                router_indices.detach().cpu(),
                routing_weights.detach().float().cpu(),
            )
            if _recorder.reference is None:
                _recorder.reference = current
            elif not (
                torch.equal(current[0], _recorder.reference[0])
                and torch.equal(current[1], _recorder.reference[1])
            ):
                raise RuntimeError("Router indices or gate weights changed between beta forwards.")
            _recorder.calls += 1
            return _base_forward(hidden_states, router_indices, routing_weights)

        experts.forward = types.MethodType(recording_forward, experts)

    def reset(self) -> None:
        self.reference = None
        self.calls = 0

    def restore(self) -> None:
        self.experts.forward = self.forward


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload).__name__}.")
    return payload


def _resolve_manifest(scores_path: Path, metadata: dict, override: str | None) -> Path:
    candidates = [Path(override)] if override else []
    if not override:
        stored = Path(str(metadata["selection_manifest"]))
        candidates.extend((stored, Path.cwd() / stored, scores_path.parent / stored))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError("Could not resolve frozen selection manifest: " + ", ".join(map(str, candidates)))


def _score_token_counts(batch: list[dict], metadata: dict):
    if bool(metadata.get("score_token_counts_variable", False)):
        return [int(sample["score_token_count"]) for sample in batch]
    return int(metadata["score_tokens_per_sample"])


def _set_score_mask(bundle, block, inputs: dict, batch: list[dict], metadata: dict) -> torch.Tensor:
    block_device = next(block.parameters()).device
    mask = _build_fixed_score_mask(bundle, inputs, _score_token_counts(batch, metadata)).to(block_device)
    if hasattr(block, "mlp"):
        block.mlp.moe_score_mask = mask.view(-1, 1)
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            set_block_modality_masks(
                bundle,
                block,
                input_ids,
                inputs["attention_mask"].to(block_device),
            )
    return mask


def fused_energy_and_ablation(
    experts,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute energy and exact single-removal output error from routed expert outputs."""
    energy = torch.zeros(num_experts, dtype=torch.float64)
    ablation = torch.zeros(num_experts, dtype=torch.float64)
    active = torch.zeros(num_experts, dtype=torch.bool)
    outputs = getattr(experts, "saved_down_output", None)
    weights = getattr(experts, "saved_router_weights", None)
    score_masks = getattr(experts, "saved_score_mask", None)
    token_indices = getattr(experts, "saved_token_indices", None)
    if not all(isinstance(value, (list, tuple)) for value in (outputs, weights, score_masks, token_indices)):
        raise RuntimeError("Instrumented fused experts did not expose routed contribution tensors.")

    for expert_idx in range(num_experts):
        out = outputs[expert_idx]
        route_weight = weights[expert_idx]
        score_mask = score_masks[expert_idx]
        token_idx = token_indices[expert_idx]
        if out is None:
            continue
        keep = score_mask.to(device=out.device).view(-1).bool()
        if not bool(keep.any()):
            continue
        weighted = out * route_weight.to(device=out.device, dtype=out.dtype).view(-1, 1)
        weighted = weighted[keep].float()
        selected_indices = token_idx.to(device=out.device).view(-1)[keep]
        energy[expert_idx] = weighted.square().mean(dim=-1).sum().double().cpu()

        unique_indices, inverse = torch.unique(selected_indices, sorted=False, return_inverse=True)
        combined = torch.zeros(
            (unique_indices.numel(), weighted.shape[-1]),
            device=weighted.device,
            dtype=torch.float32,
        )
        combined.index_add_(0, inverse, weighted)
        ablation[expert_idx] = combined.square().mean(dim=-1).sum().double().cpu()
        active[expert_idx] = True
    return energy, ablation, active


def select_quantile_experts(score: torch.Tensor, active_counts: torch.Tensor) -> list[dict[str, Any]]:
    eligible = ((score > 0) & torch.isfinite(score) & (active_counts > 0)).nonzero().flatten()
    if eligible.numel() < 3:
        raise ValueError("Need at least three active experts with positive Hessian scores.")
    ordered = eligible[torch.argsort(score[eligible])]
    selected = []
    used: set[int] = set()
    for label, quantile in QUANTILES:
        target = int(round(quantile * (ordered.numel() - 1)))
        positions = sorted(range(ordered.numel()), key=lambda position: (abs(position - target), position))
        expert_idx = next(int(ordered[position]) for position in positions if int(ordered[position]) not in used)
        used.add(expert_idx)
        selected.append(
            {
                "label": label,
                "quantile": quantile,
                "expert_idx": expert_idx,
                "hessian_score_per_token": float(score[expert_idx]),
            }
        )
    return selected


def _identity_point_gradient_and_hessian_diag(
    block,
    alpha: torch.Tensor,
    in_args,
    in_kwargs,
    baseline: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    block_dtype: torch.dtype,
    device_type: str,
    autocast_enabled: bool,
    active_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Real forward+double-backward at alpha=1 (no closed-form shortcut).

    Returns (dL/dalpha, diag(d^2L/dalpha^2), base_loss), each evaluated at the
    identity reconstruction point alpha=1. Halves the double-backward chunk
    size on CUDA OOM, matching the autotuning behavior previously provided by
    ``compute_expert_second_order_batched`` for this workload.
    """
    chunk_size = max(int(active_indices.numel()), 1)
    while True:
        try:
            with suspend_tensor_saving(block), torch.enable_grad(), torch.autocast(
                device_type=device_type, dtype=block_dtype, enabled=autocast_enabled
            ):
                pred = unwrap_output(block(*in_args, **in_kwargs))
                loss, _ = compute_block_loss(
                    pred=pred,
                    teacher_target=baseline,
                    attn_mask=score_mask,
                    loss_fn="l2",
                )
            base_loss_value = float(loss.detach().item())
            d1 = torch.autograd.grad(loss, alpha, create_graph=True, allow_unused=True)[0]
            hessian_diag = torch.zeros(num_experts, device=alpha.device, dtype=torch.float32)
            if d1 is None:
                gradient_value = torch.zeros(num_experts, device=alpha.device, dtype=torch.float32)
            else:
                gradient_value = d1.detach().float()
                if d1.requires_grad and active_indices.numel() > 0:
                    total_active = int(active_indices.numel())
                    for start in range(0, total_active, chunk_size):
                        end = min(start + chunk_size, total_active)
                        chunk_indices = active_indices[start:end]
                        grad_outputs = torch.eye(end - start, device=alpha.device, dtype=d1.dtype)
                        d2_rows = torch.autograd.grad(
                            d1[chunk_indices],
                            alpha,
                            grad_outputs=grad_outputs,
                            is_grads_batched=True,
                            retain_graph=end < total_active,
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                        if d2_rows is None:
                            continue
                        local_positions = torch.arange(end - start, device=alpha.device)
                        hessian_diag[chunk_indices] = d2_rows[local_positions, chunk_indices].detach().float()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or chunk_size <= 1:
                raise
            next_chunk_size = max(1, chunk_size // 2)
            print(
                "[hessian-diag autotune] CUDA OOM with expert group "
                f"size={chunk_size}; retrying with size={next_chunk_size}.",
                flush=True,
            )
            block.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk_size = next_chunk_size
            continue
        return gradient_value, hessian_diag, base_loss_value


def _prepare_block_inputs(bundle, model, source_block, block, batch, metadata, teacher_state):
    teacher_state.clear()
    inputs = prepare_raw_batch_inputs(bundle, batch)
    inputs = move_inputs_to_model_device(model, inputs)
    with torch.no_grad():
        model(**filter_model_forward_inputs(model, inputs), use_cache=False, return_dict=True)
    if not teacher_state:
        raise RuntimeError("Teacher hook did not capture block inputs.")
    block_device = next(source_block.parameters()).device
    block_dtype = next(block.parameters()).dtype
    in_args = move_to_device_dtype(teacher_state["in_args"], block_device, block_dtype)
    in_kwargs = move_to_device_dtype(teacher_state["in_kwargs"], block_device, block_dtype)
    score_mask = _set_score_mask(bundle, block, inputs, batch, metadata)
    return in_args, in_kwargs, score_mask


def collect_layer(
    bundle,
    loader,
    layer_idx: int,
    metadata: dict,
    betas: list[float],
    run_sweep: bool,
    validation_dtype: torch.dtype,
) -> dict:
    model = bundle.model
    source_block = teacher_block(bundle, layer_idx)
    block_device = next(source_block.parameters()).device
    block_dtype = validation_dtype
    block = copy.deepcopy(source_block).to(device=block_device, dtype=block_dtype).eval()
    experts = block.mlp.experts
    if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
        raise TypeError("Method-validation B currently requires the fused Qwen expert container.")
    num_experts = int(experts.num_experts)
    device_type = block_device.type
    autocast_enabled = device_type == "cuda" and block_dtype in (torch.float16, torch.bfloat16)
    teacher_state: dict[str, Any] = {}
    teacher_handle = register_teacher_block_hook(source_block, teacher_state)

    hessian_sum = torch.zeros(num_experts, dtype=torch.float64)
    gradient_sum = torch.zeros(num_experts, dtype=torch.float64)
    energy_sum = torch.zeros(num_experts, dtype=torch.float64)
    ablation_sum = torch.zeros(num_experts, dtype=torch.float64)
    active_counts = torch.zeros(num_experts, dtype=torch.int64)
    total_tokens = 0
    total_batches = 0
    base_loss_sum = 0.0

    try:
        instrumented_state = patch_qwen_fused_experts_forward(block)
        assert instrumented_state is not None
        original_expert_forward = instrumented_state[1]
        for batch in tqdm(loader, desc=f"Validation-B L{layer_idx}", leave=False):
            in_args, in_kwargs, score_mask = _prepare_block_inputs(
                bundle, model, source_block, block, batch, metadata, teacher_state
            )
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=block_dtype, enabled=autocast_enabled
            ):
                baseline = unwrap_output(block(*in_args, **in_kwargs)).detach()
            batch_energy, batch_ablation, active = fused_energy_and_ablation(
                experts, num_experts=num_experts
            )
            energy_sum.add_(batch_energy)
            ablation_sum.add_(batch_ablation)
            active_counts.add_(active.long())

            experts.forward = original_expert_forward
            active_indices = active.nonzero(as_tuple=False).flatten().to(block_device)
            probe_alpha = torch.ones(num_experts, device=block_device, dtype=torch.float32, requires_grad=True)
            probe_alpha_state = patch_expert_output_alpha_vector(experts, probe_alpha)
            try:
                gradient_value, hessian_diag, base_loss_value = _identity_point_gradient_and_hessian_diag(
                    block,
                    probe_alpha,
                    in_args,
                    in_kwargs,
                    baseline,
                    score_mask,
                    block_dtype=block_dtype,
                    device_type=device_type,
                    autocast_enabled=autocast_enabled,
                    active_indices=active_indices,
                    num_experts=num_experts,
                )
            finally:
                _restore_patched_expert(probe_alpha_state)
                block.zero_grad(set_to_none=True)
            hessian_sum.add_(hessian_diag.double().cpu())
            gradient_sum.add_(gradient_value.double().cpu())
            base_loss_sum += base_loss_value
            total_tokens += int(score_mask.sum().item())
            total_batches += 1
            patch_qwen_fused_experts_forward(block)
            clear_block_saved_tensors(block)

        experts.forward = original_expert_forward
        token_scale = float(total_tokens)
        hessian_score = 0.5 * hessian_sum / token_scale
        result = {
            "num_batches": total_batches,
            "num_score_tokens": total_tokens,
            "base_loss_per_token": base_loss_sum / token_scale,
            "gradient_per_token": (gradient_sum / token_scale).float(),
            "hessian_score_per_token": hessian_score.float(),
            "energy_per_token": (energy_sum / token_scale).float(),
            "ablation_per_token": (ablation_sum / token_scale).float(),
            "active_batch_counts": active_counts,
            "beta_sweep": None,
        }

        if run_sweep:
            selected = select_quantile_experts(hessian_score, active_counts)
            alpha = torch.ones(num_experts, device=block_device, dtype=torch.float32, requires_grad=True)
            alpha_state = patch_expert_output_alpha_vector(experts, alpha)
            route_recorder = RouteRecorder(experts)
            measured = {
                item["label"]: torch.zeros(len(betas), dtype=torch.float64) for item in selected
            }
            local_gradient = {
                item["label"]: torch.zeros(len(betas), dtype=torch.float64) for item in selected
            }
            sweep_tokens = 0
            sweep_batches = 0
            route_comparisons = 0
            try:
                for batch in tqdm(loader, desc=f"Beta sweep L{layer_idx}", leave=False):
                    in_args, in_kwargs, score_mask = _prepare_block_inputs(
                        bundle, model, source_block, block, batch, metadata, teacher_state
                    )
                    route_recorder.reset()
                    alpha.data.fill_(1.0)
                    with torch.enable_grad(), suspend_tensor_saving(block), torch.autocast(
                        device_type=device_type, dtype=block_dtype, enabled=autocast_enabled
                    ):
                        with torch.no_grad():
                            baseline = unwrap_output(block(*in_args, **in_kwargs)).detach()
                        for item in selected:
                            expert_idx = int(item["expert_idx"])
                            for beta_idx, beta in enumerate(betas):
                                alpha.data.fill_(1.0)
                                alpha.data[expert_idx] = beta
                                pred = unwrap_output(block(*in_args, **in_kwargs))
                                loss, _ = compute_block_loss(
                                    pred=pred,
                                    teacher_target=baseline,
                                    attn_mask=score_mask,
                                    loss_fn="l2",
                                )
                                measured[item["label"]][beta_idx] += float(loss.float().item())
                                # Plan (1): re-differentiate at this beta_0 instead of
                                # extrapolating identity_gradient from beta=1.
                                (grad,) = torch.autograd.grad(loss, alpha, allow_unused=True)
                                grad_value = 0.0 if grad is None else float(grad[expert_idx].double().item())
                                local_gradient[item["label"]][beta_idx] += grad_value
                    expected_calls = 1 + len(selected) * len(betas)
                    if route_recorder.calls != expected_calls:
                        raise RuntimeError(
                            f"Expected {expected_calls} routed calls, got {route_recorder.calls}."
                        )
                    route_comparisons += route_recorder.calls - 1
                    sweep_tokens += int(score_mask.sum().item())
                    sweep_batches += 1
            finally:
                route_recorder.restore()
                _restore_patched_expert(alpha_state)

            if sweep_tokens != total_tokens or sweep_batches != total_batches:
                raise RuntimeError("Beta sweep did not reuse the complete calibration set.")
            beta_tensor = torch.tensor(betas, dtype=torch.float64)
            baseline_idx = int(torch.argmin((beta_tensor - 1.0).abs()))
            curves = []
            for item in selected:
                values = measured[item["label"]] / float(sweep_tokens)
                values = values - values[baseline_idx]
                local_gradient_values = (local_gradient[item["label"]] / float(sweep_tokens)).float()
                curves.append(
                    {
                        **item,
                        "measured_delta_per_token": values.float(),
                        "local_gradient_at_beta_per_token": local_gradient_values,
                    }
                )
            result["beta_sweep"] = {
                "beta_values": beta_tensor.float(),
                "route_consistency_verified": True,
                "route_comparisons": route_comparisons,
                "curves": curves,
            }
        return result
    finally:
        teacher_handle.remove()
        clear_block_saved_tensors(block)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Existing scores.pt with frozen-run metadata.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--sweep-layer", type=int, default=None)
    parser.add_argument("--selection-manifest", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use a deterministic prefix of the frozen manifest for numerical validation.",
    )
    parser.add_argument(
        "--validation-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Copied-block precision; float32 is required for strict numerical equivalence.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--force", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Strictly validate and continue an existing layer checkpoint.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    scores_path = Path(args.scores).expanduser().resolve()
    scores = _torch_load(scores_path)
    metadata = dict(scores["metadata"])
    manifest_path = _resolve_manifest(scores_path, metadata, args.selection_manifest)
    manifest, manifest_sha256 = _load_selection_manifest(str(manifest_path))
    expected_sha = str(metadata.get("selection_manifest_sha256", ""))
    if expected_sha and manifest_sha256 != expected_sha:
        raise ValueError(f"Manifest SHA256 mismatch: {manifest_sha256} != {expected_sha}.")
    model_name = args.model_name_or_path or str(metadata["model_name_or_path"])
    batch_size = args.batch_size or int(metadata["batch_size"])
    samples = list(manifest["samples"])
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        samples = samples[: args.max_samples]
    dataset = ManifestRawDataset(
        samples,
        num_video_frames=int(manifest.get("num_video_frames", 8)),
        video_max_long_side=int(manifest.get("video_max_long_side", 480)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_identity_collate)
    attn_implementation = args.attn_implementation or str(
        metadata.get("attn_implementation", "flash_attention_2")
    )
    if args.sweep_layer is not None and args.sweep_layer not in args.layers:
        raise ValueError("--sweep-layer must belong to this worker's --layers.")

    output_path = Path(args.output).expanduser().resolve()
    run_metadata = {
        "model_name_or_path": model_name,
        "selection_manifest": str(manifest_path),
        "selection_manifest_sha256": manifest_sha256,
        "batch_size": batch_size,
        "num_samples": len(samples),
        "loss_fn": "l2",
        "normalization": "sum of hidden-dimension MSE divided by score-token count",
        "validation_dtype": args.validation_dtype,
        "sweep_layer": args.sweep_layer,
    }
    layer_results = {}
    if output_path.exists():
        if args.resume:
            previous = _torch_load(output_path)
            if previous.get("kind") != "method_validation_b_shard":
                raise ValueError(f"Cannot resume non-shard payload: {output_path}")
            if int(previous.get("schema_version", -1)) != 1:
                raise ValueError(f"Cannot resume unsupported shard schema: {output_path}")
            if previous.get("metadata") != run_metadata:
                raise ValueError(f"Resume metadata mismatch: {output_path}")
            layer_results = {int(key): value for key, value in previous["layers"].items()}
            unexpected = sorted(set(layer_results) - set(args.layers))
            if unexpected:
                raise ValueError(f"Resume shard contains unrequested layers: {unexpected}")
            print(
                f"[validation-b] resuming completed layers={sorted(layer_results)} from {output_path}",
                flush=True,
            )
        elif not args.force:
            raise FileExistsError(
                f"Output exists: {output_path}; pass --resume or --force."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(
        model_name,
        device_map=args.device_map,
        attn_implementation=attn_implementation,
    )
    layer_structure, _ = discover_layer_structure(bundle)
    invalid = sorted(set(args.layers) - set(layer_structure))
    if invalid:
        raise ValueError(f"Requested non-MoE layers {invalid}; available={sorted(layer_structure)}.")
    validation_dtype = torch.float32 if args.validation_dtype == "float32" else torch.bfloat16
    for layer_idx in args.layers:
        if int(layer_idx) in layer_results:
            continue
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        layer_results[int(layer_idx)] = collect_layer(
            bundle,
            loader,
            int(layer_idx),
            metadata,
            [float(beta) for beta in args.betas],
            run_sweep=int(layer_idx) == args.sweep_layer,
            validation_dtype=validation_dtype,
        )
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "kind": "method_validation_b_shard",
                "layers": layer_results,
                "metadata": run_metadata,
            },
            temporary,
        )
        temporary.replace(output_path)
        print(f"[validation-b] checkpointed layers={sorted(layer_results)} to {output_path}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    path = run(args)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
