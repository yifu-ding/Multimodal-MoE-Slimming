"""Measure real block-reconstruction loss along three expert beta directions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _path in (REPO_PARENT, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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
from src.calibration.collect_scores_main import (
    _assert_cuda_runtime_compat,
    _identity_collate,
    _load_selection_manifest,
)
from src.calibration.collector.loop_2_helpers import (
    _restore_patched_expert,
    patch_expert_output_alpha_vector,
)
from src.calibration.helpers.helpers import compute_block_loss, teacher_block
from src.calibration.helpers.hooks import register_teacher_block_hook
from src.calibration.helpers.utils import move_to_device_dtype, unwrap_output
from src.calibration.representation_distill.common import prepare_raw_batch_inputs
from src.calibration.representation_distill.runtime.dump_original_data import (
    ManifestRawDataset,
)


DEFAULT_BETAS = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_LABELS = ("low", "medium", "high")


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload).__name__}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probe(payload: dict, path: Path | None = None) -> None:
    source = str(path) if path is not None else "Hessian probe"
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported schema_version in {source}; expected 1.")
    required = (
        "layer_idx",
        "loss_fn",
        "num_batches",
        "num_score_tokens",
        "hessian_per_token",
        "gradient_per_token",
        "active_batch_counts",
        "metadata",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing required probe fields in {source}: {missing}.")
    if payload["loss_fn"] not in {"l2", "rel_l2"}:
        raise ValueError("Beta sweep supports only the exactly quadratic l2/rel_l2 losses.")
    hessian = payload["hessian_per_token"]
    gradient = payload["gradient_per_token"]
    active = payload["active_batch_counts"]
    if not isinstance(hessian, torch.Tensor) or hessian.ndim != 2:
        raise TypeError("hessian_per_token must be a rank-2 tensor.")
    if hessian.shape[0] != hessian.shape[1] or hessian.shape[0] == 0:
        raise ValueError(f"Hessian must be nonempty and square, got {tuple(hessian.shape)}.")
    if not isinstance(gradient, torch.Tensor) or tuple(gradient.shape) != (hessian.shape[0],):
        raise ValueError("gradient_per_token shape does not match the Hessian.")
    if not isinstance(active, torch.Tensor) or tuple(active.shape) != (hessian.shape[0],):
        raise ValueError("active_batch_counts shape does not match the Hessian.")
    if not bool(torch.isfinite(hessian).all()) or not bool(torch.isfinite(gradient).all()):
        raise ValueError("Hessian probe contains non-finite values.")
    if int(payload["num_batches"]) <= 0 or int(payload["num_score_tokens"]) <= 0:
        raise ValueError("Probe batch and score-token counts must be positive.")
    if not isinstance(payload["metadata"], dict):
        raise TypeError("Probe metadata must be a dictionary.")
    metadata_required = (
        "model_name_or_path",
        "selection_manifest",
        "selection_manifest_sha256",
        "batch_size",
    )
    missing_metadata = [key for key in metadata_required if not payload["metadata"].get(key)]
    if missing_metadata:
        raise KeyError(
            "Standalone beta sweep requires a frozen-manifest probe; missing metadata: "
            f"{missing_metadata}."
        )
    variable_score_counts = bool(
        payload["metadata"].get("score_token_counts_variable", False)
    )
    if variable_score_counts:
        if int(payload["metadata"].get("score_token_budget", 0)) <= 0:
            raise KeyError("Variable-quota probe metadata requires score_token_budget.")
    elif int(payload["metadata"].get("score_tokens_per_sample", 0)) <= 0:
        raise KeyError("Fixed-quota probe metadata requires score_tokens_per_sample.")
    symmetric = 0.5 * (hessian.double() + hessian.double().T)
    scale = max(float(symmetric.abs().max().item()), 1e-30)
    relative_asymmetry = float((hessian.double() - hessian.double().T).abs().max().item()) / scale
    if relative_asymmetry > 1e-3:
        raise ValueError(
            f"Hessian is unexpectedly asymmetric (relative max={relative_asymmetry:.3e})."
        )


def select_quantile_experts(payload: dict) -> list[dict[str, Any]]:
    hessian = payload["hessian_per_token"].detach().cpu().double()
    scores = 0.5 * torch.diag(0.5 * (hessian + hessian.T))
    active_counts = payload["active_batch_counts"].detach().cpu().long()
    positive_threshold = max(float(scores.max().item()) * 1e-12, 0.0)
    eligible = ((active_counts > 0) & torch.isfinite(scores) & (scores > positive_threshold)).nonzero(
        as_tuple=False
    ).flatten()
    if eligible.numel() < 3:
        raise ValueError(
            "Need at least three active experts with positive Hessian diagonal for P10/P50/P90."
        )
    ordered = eligible[torch.argsort(scores[eligible])]
    selected = []
    used: set[int] = set()
    for label, quantile in zip(QUANTILE_LABELS, QUANTILES):
        position = int(round(quantile * (ordered.numel() - 1)))
        candidates = sorted(
            range(ordered.numel()),
            key=lambda idx: (abs(idx - position), idx),
        )
        expert_idx = next(int(ordered[idx].item()) for idx in candidates if int(ordered[idx]) not in used)
        used.add(expert_idx)
        selected.append(
            {
                "label": label,
                "quantile": float(quantile),
                "expert_idx": expert_idx,
                "hessian_score_per_token": float(scores[expert_idx].item()),
                "active_batch_count": int(active_counts[expert_idx].item()),
                "active_batch_fraction": float(
                    active_counts[expert_idx].item() / int(payload["num_batches"])
                ),
            }
        )
    return selected


def _resolve_manifest_path(stored_path: str, probe_path: Path, override: str | None) -> Path:
    candidates = [Path(override)] if override else []
    stored = Path(stored_path)
    if not override:
        candidates.extend([stored, Path(REPO_ROOT) / stored, probe_path.parent / stored])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find the frozen selection manifest; tried: {attempted}")


def verify_configuration(args: argparse.Namespace, payload: dict, probe_path: Path):
    metadata = payload["metadata"]

    expected = {
        "model_name_or_path": str(metadata["model_name_or_path"]),
        "layer_idx": int(payload["layer_idx"]),
        "loss_fn": str(payload["loss_fn"]),
        "batch_size": int(metadata["batch_size"]),
        "score_tokens_per_sample": None
        if metadata.get("score_tokens_per_sample") is None
        else int(metadata["score_tokens_per_sample"]),
        "score_token_budget": int(
            metadata.get("score_token_budget", payload["num_score_tokens"])
        ),
        "score_token_counts_variable": bool(
            metadata.get("score_token_counts_variable", False)
        ),
        "attn_implementation": str(
            metadata.get("attn_implementation", "flash_attention_2")
        ),
    }
    requested = {
        "model_name_or_path": args.model_name_or_path,
        "layer_idx": args.layer,
        "loss_fn": args.loss_fn,
        "batch_size": args.batch_size,
        "score_tokens_per_sample": args.score_tokens_per_sample,
        "score_token_budget": args.score_token_budget,
        "attn_implementation": args.attn_implementation,
    }
    for key, value in requested.items():
        if value is not None and value != expected[key]:
            raise ValueError(
                f"Requested {key}={value!r} does not match probe value {expected[key]!r}."
            )

    manifest_path = _resolve_manifest_path(
        str(metadata["selection_manifest"]), probe_path, args.selection_manifest
    )
    manifest, manifest_sha256 = _load_selection_manifest(str(manifest_path))
    expected_sha256 = str(metadata["selection_manifest_sha256"])
    if manifest_sha256 != expected_sha256:
        raise ValueError(
            "Selection manifest SHA256 mismatch: "
            f"probe={expected_sha256}, current={manifest_sha256}, path={manifest_path}."
        )
    if expected["score_token_counts_variable"]:
        manifest_budget = int(manifest.get("score_token_budget", 0))
        if manifest_budget != expected["score_token_budget"]:
            raise ValueError(
                "Manifest score_token_budget does not match the Hessian probe: "
                f"{manifest_budget} != {expected['score_token_budget']}."
            )
    else:
        manifest_score_tokens = int(manifest["score_tokens_per_sample"])
        if manifest_score_tokens != expected["score_tokens_per_sample"]:
            raise ValueError(
                "Manifest score_tokens_per_sample does not match the Hessian probe: "
                f"{manifest_score_tokens} != {expected['score_tokens_per_sample']}."
            )
    if len(manifest["samples"]) != int(metadata.get("selected_num_samples", len(manifest["samples"]))):
        raise ValueError("Manifest sample count does not match probe metadata.")
    expected_tokens = (
        sum(int(item["score_token_count"]) for item in manifest["samples"])
        if expected["score_token_counts_variable"]
        else len(manifest["samples"]) * expected["score_tokens_per_sample"]
    )
    if expected_tokens != int(payload["num_score_tokens"]):
        raise ValueError(
            f"Manifest implies {expected_tokens} score tokens, but probe stores "
            f"{int(payload['num_score_tokens'])}."
        )
    expected_batches = math.ceil(len(manifest["samples"]) / expected["batch_size"])
    if expected_batches != int(payload["num_batches"]):
        raise ValueError(
            f"Manifest/batch size imply {expected_batches} batches, but probe stores "
            f"{int(payload['num_batches'])}."
        )
    return expected, manifest_path, manifest, manifest_sha256


class RouteRecorder:
    """Assert that every beta forward receives identical router tensors."""

    def __init__(self, experts):
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            raise TypeError(
                "Route consistency recording currently requires the fused Qwen expert container."
            )
        self.experts = experts
        self.forward = experts.forward
        self.reference = None
        self.calls = 0

        def _recording_forward(
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
            else:
                same_indices = torch.equal(current[0], _recorder.reference[0])
                same_weights = torch.equal(current[1], _recorder.reference[1])
                if not same_indices or not same_weights:
                    raise RuntimeError(
                        "Router indices or gate weights changed between beta forwards."
                    )
            _recorder.calls += 1
            return _base_forward(hidden_states, router_indices, routing_weights)

        experts.forward = types.MethodType(_recording_forward, experts)

    def reset(self) -> None:
        self.reference = None
        self.calls = 0

    def restore(self) -> None:
        self.experts.forward = self.forward


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure P10/P50/P90 expert beta-loss curves from a Hessian probe."
    )
    parser.add_argument("--probe", required=True, help="Path to hessian_probe_L*.pt.")
    parser.add_argument("--output", default=None, help="Output .pt path beside the probe by default.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--selection_manifest", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--loss_fn", choices=["l2", "rel_l2"], default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--score_tokens_per_sample", type=int, default=None)
    parser.add_argument("--score_token_budget", type=int, default=None)
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    probe_path = Path(args.probe).expanduser().resolve()
    payload = _torch_load(probe_path)
    validate_probe(payload, probe_path)
    expected, manifest_path, manifest, manifest_sha256 = verify_configuration(
        args, payload, probe_path
    )
    selected = select_quantile_experts(payload)

    betas = [float(value) for value in args.betas]
    if len(betas) < 3 or len(set(betas)) != len(betas):
        raise ValueError("Provide at least three distinct beta values.")
    if not any(abs(value - 1.0) < 1e-12 for value in betas):
        raise ValueError("Beta values must include 1.0 for the measured baseline.")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else probe_path.with_name(f"hessian_beta_sweep_L{expected['layer_idx']}.pt")
    )
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to overwrite.")

    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    _assert_cuda_runtime_compat(device_map)
    attn_implementation = expected["attn_implementation"]
    print(
        "[beta-sweep] Verified probe + manifest: "
        f"layer={expected['layer_idx']}, loss={expected['loss_fn']}, "
        f"model={expected['model_name_or_path']}, sha256={manifest_sha256[:12]}...",
        flush=True,
    )
    print(
        "[beta-sweep] Selected experts: "
        + ", ".join(
            f"{item['label']}=E{item['expert_idx']} "
            f"(H/2={item['hessian_score_per_token']:.6g})"
            for item in selected
        ),
        flush=True,
    )

    bundle = load_model_bundle(
        expected["model_name_or_path"],
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    layer_to_num_experts, _ = discover_layer_structure(bundle)
    layer_idx = expected["layer_idx"]
    if layer_idx not in layer_to_num_experts:
        raise ValueError(
            f"Probe layer {layer_idx} is not an MoE layer in the loaded model; "
            f"available={sorted(layer_to_num_experts)}."
        )
    hessian_size = int(payload["hessian_per_token"].shape[0])
    if int(layer_to_num_experts[layer_idx]) != hessian_size:
        raise ValueError(
            "Loaded model expert count does not match Hessian: "
            f"{layer_to_num_experts[layer_idx]} != {hessian_size}."
        )
    stored_family = payload["metadata"].get("model_family")
    if stored_family is not None and stored_family != bundle.family:
        raise ValueError(
            f"Loaded model family {bundle.family!r} does not match probe {stored_family!r}."
        )

    dataset = ManifestRawDataset(
        manifest["samples"],
        num_video_frames=int(manifest.get("num_video_frames", 8)),
        video_max_long_side=int(manifest.get("video_max_long_side", 480)),
    )
    loader = DataLoader(
        dataset,
        batch_size=expected["batch_size"],
        shuffle=False,
        collate_fn=_identity_collate,
    )

    model = bundle.model
    model.eval()
    source_block = teacher_block(bundle, layer_idx)
    copied_block = copy.deepcopy(source_block)
    block_device = next(source_block.parameters()).device
    block_dtype = next(source_block.parameters()).dtype
    stored_dtype = payload["metadata"].get("block_dtype")
    if stored_dtype is not None and str(block_dtype) != stored_dtype:
        raise ValueError(
            f"Loaded block dtype {str(block_dtype)!r} does not match probe {stored_dtype!r}."
        )
    copied_block = copied_block.to(device=block_device, dtype=block_dtype).eval()
    experts = copied_block.mlp.experts
    alpha = torch.ones(hessian_size, device=block_device, dtype=torch.float32)
    alpha_state = patch_expert_output_alpha_vector(experts, alpha)
    route_recorder = RouteRecorder(experts)
    teacher_state: dict[str, Any] = {}
    teacher_handle = register_teacher_block_hook(source_block, teacher_state)
    measured_sums = {
        item["label"]: torch.zeros(len(betas), dtype=torch.float64) for item in selected
    }
    total_score_tokens = 0
    total_batches = 0
    route_checks = 0
    device_type = block_device.type
    autocast_enabled = device_type == "cuda" and block_dtype in (
        torch.float16,
        torch.bfloat16,
    )

    try:
        for batch in tqdm(loader, desc=f"Beta sweep L{layer_idx}"):
            teacher_state.clear()
            inputs = prepare_raw_batch_inputs(bundle, batch)
            inputs = move_inputs_to_model_device(model, inputs)
            batch_score_token_counts = (
                [int(sample["score_token_count"]) for sample in batch]
                if expected["score_token_counts_variable"]
                else expected["score_tokens_per_sample"]
            )
            score_mask = _build_fixed_score_mask(
                bundle, inputs, batch_score_token_counts
            ).to(block_device)
            with torch.no_grad():
                model(
                    **filter_model_forward_inputs(model, inputs),
                    use_cache=False,
                    return_dict=True,
                )
            if not teacher_state:
                raise RuntimeError(f"Teacher hook did not capture layer {layer_idx} inputs.")
            in_args = move_to_device_dtype(teacher_state["in_args"], block_device, block_dtype)
            in_kwargs = move_to_device_dtype(teacher_state["in_kwargs"], block_device, block_dtype)
            teacher_target = move_to_device_dtype(
                unwrap_output(teacher_state["output"]), block_device, block_dtype
            )
            route_recorder.reset()
            with torch.no_grad():
                for selection in selected:
                    expert_idx = int(selection["expert_idx"])
                    for beta_idx, beta in enumerate(betas):
                        alpha.fill_(1.0)
                        alpha[expert_idx] = beta
                        with torch.autocast(
                            device_type=device_type,
                            dtype=block_dtype,
                            enabled=autocast_enabled,
                        ):
                            pred = unwrap_output(copied_block(*in_args, **in_kwargs))
                            loss_sum, _ = compute_block_loss(
                                pred=pred,
                                teacher_target=teacher_target,
                                attn_mask=score_mask,
                                loss_fn=expected["loss_fn"],
                            )
                        measured_sums[selection["label"]][beta_idx] += float(
                            loss_sum.detach().float().item()
                        )
                alpha.fill_(1.0)
            expected_calls = len(selected) * len(betas)
            if route_recorder.calls != expected_calls:
                raise RuntimeError(
                    f"Expected {expected_calls} routed expert calls in one batch, got "
                    f"{route_recorder.calls}."
                )
            route_checks += route_recorder.calls - 1
            total_score_tokens += int(score_mask.sum().item())
            total_batches += 1
    finally:
        teacher_handle.remove()
        route_recorder.restore()
        _restore_patched_expert(alpha_state)

    if total_score_tokens != int(payload["num_score_tokens"]):
        raise RuntimeError(
            f"Sweep used {total_score_tokens} score tokens, probe used "
            f"{int(payload['num_score_tokens'])}."
        )
    if total_batches != int(payload["num_batches"]):
        raise RuntimeError(
            f"Sweep used {total_batches} batches, probe used {int(payload['num_batches'])}."
        )

    beta_tensor = torch.tensor(betas, dtype=torch.float64)
    baseline_idx = int(torch.argmin((beta_tensor - 1.0).abs()).item())
    hessian = payload["hessian_per_token"].detach().cpu().double()
    gradient = payload["gradient_per_token"].detach().cpu().double()
    curves = []
    for selection in selected:
        label = selection["label"]
        expert_idx = int(selection["expert_idx"])
        measured = measured_sums[label] / float(total_score_tokens)
        measured_delta = measured - measured[baseline_idx]
        delta_beta = beta_tensor - 1.0
        first_order = gradient[expert_idx] * delta_beta
        exact = first_order + 0.5 * hessian[expert_idx, expert_idx] * delta_beta.square()
        curves.append(
            {
                **selection,
                "gradient_per_token": float(gradient[expert_idx].item()),
                "hessian_diagonal_per_token": float(hessian[expert_idx, expert_idx].item()),
                "measured_loss_per_token": measured.float(),
                "measured_delta_per_token": measured_delta.float(),
                "first_order_delta_per_token": first_order.float(),
                "exact_quadratic_delta_per_token": exact.float(),
            }
        )

    output = {
        "schema_version": 1,
        "probe_path": str(probe_path),
        "probe_sha256": _sha256(probe_path),
        "layer_idx": layer_idx,
        "loss_fn": expected["loss_fn"],
        "beta_values": beta_tensor.float(),
        "num_batches": total_batches,
        "num_score_tokens": total_score_tokens,
        "route_consistency_verified": True,
        "route_comparisons": route_checks,
        "curves": curves,
        "metadata": {
            "model_name_or_path": expected["model_name_or_path"],
            "model_family": bundle.family,
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": manifest_sha256,
            "batch_size": expected["batch_size"],
            "score_tokens_per_sample": expected["score_tokens_per_sample"],
            "score_token_budget": expected["score_token_budget"],
            "score_token_counts_variable": expected["score_token_counts_variable"],
            "attn_implementation": attn_implementation,
            "block_dtype": str(block_dtype),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output, temporary_path)
    os.replace(temporary_path, output_path)
    print(
        f"[beta-sweep] Saved {len(curves)} real beta-loss curves to {output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
