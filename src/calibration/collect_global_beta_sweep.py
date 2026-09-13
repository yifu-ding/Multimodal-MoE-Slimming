"""Measure every expert partial derivative at synchronous all-expert beta values."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from observations.common import load_model_bundle
from scripts.analyze_beta095_counterexamples import write_csv
from src.calibration.collect_beta095_counterexamples import loss_sum, set_scale
from src.calibration.collect_method_validation_b import (
    ManifestRawDataset, RouteRecorder, _identity_collate, _load_selection_manifest,
    _prepare_block_inputs, _restore_patched_expert, _torch_load,
    patch_expert_output_alpha_vector, register_teacher_block_hook,
    suspend_tensor_saving, teacher_block, unwrap_output,
)


MAIN_BETAS = (0., .25, .5, .75, .95)
CHECK_BETAS = (1.,)
BETAS = MAIN_BETAS + CHECK_BETAS
REPEAT_BETAS = (0., .5, .95)


def measure_batch(block, alpha, inputs, kwargs, mask, loss_fn, check_experts=(0, 64, 127)):
    if not alpha.is_leaf or alpha.ndim != 1 or not alpha.requires_grad:
        raise ValueError("alpha must be an independent differentiable leaf vector.")
    with suspend_tensor_saving(block):
        set_scale(alpha, 1.)
        with torch.no_grad():
            teacher = unwrap_output(block(*inputs, **kwargs)).detach()

        def forward_loss():
            return loss_sum(unwrap_output(block(*inputs, **kwargs)), teacher, mask, loss_fn)

        measurements, repeats = [], []
        for beta in BETAS:
            set_scale(alpha, beta)
            loss = forward_loss()
            # One backward yields the partial derivative for EVERY independent coordinate.
            gradient = torch.autograd.grad(loss, alpha)[0]
            measurements.append({"beta_global": beta, "loss_sum": float(loss.detach()),
                                 "gradient_sum": gradient.detach().double().cpu()})
            if beta in REPEAT_BETAS:
                set_scale(alpha, beta)
                repeated_loss = forward_loss()
                repeated_gradient = torch.autograd.grad(repeated_loss, alpha)[0]
                repeats.append({"beta_global": beta, "loss_sum": float(repeated_loss.detach()),
                                "gradient_sum": repeated_gradient.detach().double().cpu()})

        midpoint_gradient = measurements[2]["gradient_sum"]
        differences = []
        # Individual perturbations are used only to check partial derivatives at beta=.5.
        for e in check_experts:
            for step in (.005, .01):
                with torch.no_grad():
                    set_scale(alpha, .5, e, .5 + step)
                    plus = float(forward_loss())
                    set_scale(alpha, .5, e, .5 - step)
                    minus = float(forward_loss())
                differences.append({"expert": e, "background_beta": .5, "step": step,
                                    "loss_plus_sum": plus, "loss_minus_sum": minus,
                                    "gradient_sum": float(midpoint_gradient[e]),
                                    "central_difference_sum": (plus - minus) / (2 * step)})
        set_scale(alpha, 1.)
    return {"num_score_tokens": int(mask.sum()), "measurements": measurements,
            "repeats": repeats, "finite_differences": differences}


def summarize(batches, reference_gradients, *, loss_fn, layer=0):
    total_tokens = sum(b["num_score_tokens"] for b in batches)
    gradients = torch.stack([sum(b["measurements"][i]["gradient_sum"] for b in batches) / total_tokens
                             for i in range(len(BETAS))]).numpy()
    losses = np.array([sum(b["measurements"][i]["loss_sum"] for b in batches) / total_tokens
                       for i in range(len(BETAS))])
    n = gradients.shape[1]
    if not np.isfinite(gradients).all() or not np.isfinite(losses).all():
        raise ValueError("Nonfinite real gradient/loss.")
    repeat_errors, repeat_loss_errors = [], []
    for beta in REPEAT_BETAS:
        index = BETAS.index(beta)
        errors, loss_errors = [], []
        for batch in batches:
            repeated = next(r for r in batch["repeats"] if r["beta_global"] == beta)
            main = batch["measurements"][index]
            errors.append((main["gradient_sum"] - repeated["gradient_sum"]).abs().numpy())
            loss_errors.append(abs(main["loss_sum"] - repeated["loss_sum"]))
        repeat_errors.append(np.sum(errors, axis=0) / total_tokens)
        repeat_loss_errors.append(sum(loss_errors) / total_tokens)
    repeat_errors = np.array(repeat_errors)
    repeat_max = float(repeat_errors.max())
    floor = 32 * np.finfo(np.float32).eps
    tau = max(8 * repeat_max, floor * float(np.abs(gradients[0]).max()))
    valid = np.abs(gradients[0]) > tau
    ratios = np.full_like(gradients, np.nan)
    ratios[:, valid] = 100 * np.abs(gradients[:, valid]) / np.abs(gradients[0, valid])
    theoretical = (1 - np.array(BETAS))[:, None] * gradients[0]
    residuals = gradients - theoretical
    fit_records, ratio_stats = [], []
    coordinates = np.array(BETAS)
    design = np.column_stack((coordinates, np.ones_like(coordinates)))
    coefficients = np.linalg.lstsq(design, gradients, rcond=None)[0]
    fit_residuals = gradients - design @ coefficients
    for e in range(n):
        span = float(np.ptp(gradients[:, e]))
        centered_sum = float(np.sum((gradients[:, e] - gradients[:, e].mean())**2))
        fit_records.append({"expert": e, "slope": float(coefficients[0, e]), "intercept": float(coefficients[1, e]),
                            "r_squared": None if centered_sum == 0 else 1 - float(np.sum(fit_residuals[:, e]**2)) / centered_sum,
                            "max_abs_residual": float(np.abs(fit_residuals[:, e]).max()),
                            "residual_over_gradient_range": None if span == 0 else float(np.abs(fit_residuals[:, e]).max()) / span})
    rows, residual_rows = [], []
    for i, beta in enumerate(BETAS):
        values = ratios[i, valid]
        ratio_stats.append({"beta_global": beta, "valid_experts": int(valid.sum()),
                            **{name: float(fn(values)) if values.size else None for name, fn in
                               (("min_pct", np.min), ("max_pct", np.max), ("mean_pct", np.mean), ("std_pct", np.std))}})
        for e in range(n):
            g = float(gradients[i, e])
            rows.append({"layer": layer, "expert": e, "loss_fn": loss_fn, "beta_global": beta,
                         "num_score_tokens": total_tokens, "gradient_signed": g, "gradient_abs": abs(g),
                         "gradient_at_global_zero": float(gradients[0, e]),
                         "ratio_to_global_zero_pct": float(ratios[i, e]) if valid[e] else "",
                         "ratio_valid": bool(valid[e]), "first_order_signed": -beta * g,
                         "first_order_abs": abs(beta * g)})
            residual_rows.append({"layer": layer, "expert": e, "beta_global": beta,
                                  "signed_residual": float(residuals[i, e]), "absolute_residual": abs(float(residuals[i, e])),
                                  "signed_residual_over_abs_g0_pct": 100 * float(residuals[i, e]) / abs(float(gradients[0, e])) if valid[e] else "",
                                  "ratio_deviation_percentage_points": float(ratios[i, e] - 100 * (1 - beta)) if valid[e] else ""})
    fd_errors = []
    for batch in batches:
        for item in batch["finite_differences"]:
            denom = max(abs(item["gradient_sum"]), tau * batch["num_score_tokens"])
            error = abs(item["gradient_sum"] - item["central_difference_sum"])
            fd_errors.append(error / denom if denom else 0.)
    reference_error = np.abs(gradients[BETAS.index(.95)] - np.array(reference_gradients))
    loss_tolerance = floor * float(np.abs(losses).max())
    validation = {
        "passed": bool(max(fd_errors) < .01 and np.abs(gradients[-1]).max() <= tau
                       and abs(losses[-1]) <= loss_tolerance and reference_error.max() <= tau),
        "num_experts": n, "num_score_tokens": total_tokens, "gradient_row_count": len(rows),
        "ratio_denominator_threshold": tau, "invalid_ratio_experts": np.flatnonzero(~valid).tolist(),
        "max_repeat_gradient_error_per_token": repeat_max,
        "repeat_gradient_abs_error_per_expert": repeat_errors.tolist(),
        "repeat_loss_abs_error_per_token": repeat_loss_errors,
        "max_finite_difference_relative_error": max(fd_errors),
        "identity_gradient_max_abs": float(np.abs(gradients[-1]).max()), "identity_loss": float(losses[-1]),
        "loss_tolerance": loss_tolerance, "beta095_reference_max_abs_error": float(reference_error.max()),
        "beta095_reference_abs_error_per_expert": reference_error.tolist(),
    }
    linearity = {
        "fit_betas": list(BETAS), "fitted_quantity": "signed partial gradient, using actual beta coordinates",
        "constant_curve_rule": "R squared and residual/range are null when their exact denominator is zero",
        "per_expert_fits": fit_records, "ratio_statistics": ratio_stats,
        "max_abs_endpoint_scaling_residual": float(np.abs(residuals).max()),
        "max_abs_ratio_deviation_percentage_points": float(np.abs(ratios[:, valid] - 100 * (1 - coordinates[:, None])).max()) if valid.any() else None,
        "numerical_residual_threshold": tau,
        "experts_with_fit_residual_above_threshold": np.flatnonzero(np.abs(fit_residuals).max(axis=0) > tau).tolist(),
        "experts_with_endpoint_scaling_residual_above_threshold": np.flatnonzero(np.abs(residuals).max(axis=0) > tau).tolist(),
    }
    loss_rows = [{"layer": layer, "loss_fn": loss_fn, "beta_global": beta,
                  "num_score_tokens": total_tokens, "loss": float(losses[i])} for i, beta in enumerate(BETAS)]
    return rows, loss_rows, residual_rows, validation, linearity


def collect(args):
    reference_dir = Path(args.reference_dir).resolve()
    refmeta = json.loads((reference_dir / "metadata.json").read_text())
    if refmeta["loss_fn"] != args.loss_fn or refmeta["layer"] != 0 or refmeta["num_samples"] != 32:
        raise ValueError("Require matching loss and the original layer-0 32-sample reference.")
    with (reference_dir / "scores.csv").open() as handle:
        reference_rows = {int(r["expert"]): r for r in csv.DictReader(handle)}
    if set(reference_rows) != set(range(128)):
        raise ValueError("Reference must include all 128 experts.")
    metadata = _torch_load(Path(args.scores))["metadata"]
    manifest_path = Path(args.selection_manifest or refmeta["selection_manifest"]).resolve()
    manifest, manifest_hash = _load_selection_manifest(str(manifest_path))
    if manifest_hash != refmeta["selection_manifest_sha256"] or manifest_hash != metadata["selection_manifest_sha256"]:
        raise ValueError("Frozen manifest hash mismatch.")
    if refmeta["model_name_or_path"] != metadata["model_name_or_path"]:
        raise ValueError("Calibration model differs from reference.")
    samples = manifest["samples"][:32]
    sample_ids = [s["sample_id"] for s in samples]
    batch_size = args.batch_size or int(refmeta["batch_size"])
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw.pt").exists():
        raise FileExistsError(output / "raw.pt")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    loader = DataLoader(ManifestRawDataset(samples, num_video_frames=int(manifest.get("num_video_frames", 8)),
                                          video_max_long_side=int(manifest.get("video_max_long_side", 480))),
                        batch_size=batch_size, shuffle=False, collate_fn=_identity_collate)
    attention = metadata.get("attn_implementation") or "sdpa"
    bundle = load_model_bundle(refmeta["model_name_or_path"], device_map=args.device_map, attn_implementation=attention)
    source = teacher_block(bundle, 0)
    block = copy.deepcopy(source).float().eval().requires_grad_(False)
    n = int(block.mlp.experts.num_experts)
    if n != 128:
        raise ValueError("Expected 128 experts.")
    alpha = torch.ones(n, device=next(block.parameters()).device, requires_grad=True)
    patch = patch_expert_output_alpha_vector(block.mlp.experts, alpha)
    recorder = RouteRecorder(block.mlp.experts)
    teacher_state = {}
    hook = register_teacher_block_hook(source, teacher_state)
    batches, position, route_comparisons = [], 0, 0
    try:
        for batch in tqdm(loader, desc=f"Synchronous beta: {args.loss_fn}"):
            inputs, kwargs, mask = _prepare_block_inputs(bundle, bundle.model, source, block, batch, metadata, teacher_state)
            recorder.reset()
            measured = measure_batch(block, alpha, inputs, kwargs, mask, args.loss_fn)
            if recorder.calls != 22:
                raise ValueError(f"Expected 22 forwards per batch including checks, got {recorder.calls}.")
            measured["sample_ids"] = sample_ids[position:position + len(batch)]
            measured["route_comparisons"] = recorder.calls - 1
            batches.append(measured)
            route_comparisons += recorder.calls - 1
            position += len(batch)
        tokens = sum(b["num_score_tokens"] for b in batches)
        if tokens != refmeta["num_score_tokens"] or position != 32:
            raise ValueError("Sample or score-token counts differ from beta095 reference.")
        source_hashes = {str(Path(path).resolve().relative_to(Path.cwd())): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for path in (Path(__file__), Path("src/calibration/collect_beta095_counterexamples.py"),
                                      Path("src/calibration/collector/loop_2_helpers.py"))}
        runmeta = {
            "schema_version": 1, "model_name_or_path": refmeta["model_name_or_path"], "layer": 0, "num_experts": n,
            "loss_fn": args.loss_fn, "normalization": refmeta["normalization"],
            "sweep_mode": "all_experts_synchronous", "main_betas": list(MAIN_BETAS), "check_betas": list(CHECK_BETAS),
            "repeat_betas": list(REPEAT_BETAS), "teacher_background": 1.,
            "scaling_location": "individual routed expert output contributions; routing weights and residual unchanged",
            "parameterization": "independent_alpha_vector_at_equal_values", "fixed_router": True,
            "route_comparisons": route_comparisons, "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": manifest_hash, "sample_ids": sample_ids, "num_samples": 32,
            "batch_size": batch_size, "reference_batch_size": refmeta["batch_size"], "num_batches": len(batches),
            "num_score_tokens": tokens, "score_tokens_per_sample": metadata["score_tokens_per_sample"],
            "score_token_sampling": manifest.get("score_token_sampling"), "block_dtype": "float32",
            "loss_dtype": "float64", "accumulation_dtype": "float64", "seed": 42,
            "attn_implementation": attention, "torch_version": str(torch.__version__),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": source_hashes, "reference_directory": str(reference_dir),
            "reference_scores_sha256": hashlib.sha256((reference_dir / "scores.csv").read_bytes()).hexdigest(),
            "complete": False,
        }
        reference_gradients = [float(reference_rows[e]["gradient_at_work"]) for e in range(n)]
        rows, losses, residuals, validation, linearity = summarize(batches, reference_gradients, loss_fn=args.loss_fn)
        write_csv(output / "gradients.csv", rows)
        write_csv(output / "losses.csv", losses)
        write_csv(output / "linearity_residuals.csv", residuals)
        runmeta["complete"] = validation["passed"]
        torch.save({"metadata": runmeta, "batches": batches, "reference_gradient_at_beta095": reference_gradients}, output / "raw.pt")
        for name, payload in (("metadata", runmeta), ("validation", validation), ("linearity", linearity)):
            (output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        if not validation["passed"]:
            raise RuntimeError(f"Measurements saved, but numerical checks failed: {output / 'validation.json'}")
        print(json.dumps({"output": str(output), "passed": True, "ratio_statistics": linearity["ratio_statistics"],
                          "max_ratio_deviation_pp": linearity["max_abs_ratio_deviation_percentage_points"],
                          "nonlinear_experts_above_threshold": len(linearity["experts_with_fit_residual_above_threshold"])}), flush=True)
    finally:
        hook.remove()
        recorder.restore()
        _restore_patched_expert(patch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loss-fn", choices=("l2", "kl_div"), required=True)
    parser.add_argument("--selection-manifest")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device-map", default="cuda:0")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
