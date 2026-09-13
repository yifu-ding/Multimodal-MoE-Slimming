"""Measure all-expert beta=.95 gradients against beta=1 removal/Hessian data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from observations.common import load_model_bundle
from scripts.analyze_beta095_counterexamples import export_analysis, write_csv
from src.calibration.collect_method_validation_b import (
    RouteRecorder, _identity_collate, _load_selection_manifest, _prepare_block_inputs,
    _restore_patched_expert, _torch_load, patch_expert_output_alpha_vector,
    register_teacher_block_hook, suspend_tensor_saving, teacher_block, unwrap_output,
    ManifestRawDataset,
)


BETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.25, 1.5)


def loss_sum(pred, target, mask, loss_fn):
    """Same mathematical block loss as compute_block_loss, with FP64 loss arithmetic."""
    pred, target = pred.double(), target.double()
    if loss_fn == "l2":
        tokens = (pred - target).square().mean(-1)
    elif loss_fn == "kl_div":
        tokens = F.kl_div(F.log_softmax(pred, -1), F.softmax(target, -1), reduction="none").sum(-1)
    else:
        raise ValueError(loss_fn)
    return (tokens * mask.double()).sum()


def set_scale(alpha, background, expert=None, beta=None):
    with torch.no_grad():
        alpha.fill_(background)
        if expert is not None:
            alpha[expert] = beta


def measure_batch(block, alpha, in_args, in_kwargs, mask, *, loss_fn, beta_work):
    """Independent forwards for both backgrounds; never infer deletion from curvature."""
    def forward_loss(target):
        return loss_sum(unwrap_output(block(*in_args, **in_kwargs)), target, mask, loss_fn)

    with suspend_tensor_saving(block):
        set_scale(alpha, 1.0)
        with torch.no_grad():
            target = unwrap_output(block(*in_args, **in_kwargs)).detach()
        identity_loss = forward_loss(target)
        identity_gradient = torch.autograd.grad(identity_loss, alpha)[0].detach().double().cpu()
        identity_base = float(identity_loss.detach())
        gradients, work_losses = [], []
        for _ in range(2):
            set_scale(alpha, beta_work)
            loss = forward_loss(target)
            gradients.append(torch.autograd.grad(loss, alpha)[0].detach().double().cpu())
            work_losses.append(float(loss.detach()))
        n = alpha.numel()
        removal = torch.zeros(n, dtype=torch.float64)
        removal_work = torch.zeros_like(removal)
        repeat_error = torch.zeros_like(removal)
        # Probe all experts, including ones with cancelling/zero gradient.
        with torch.no_grad():
            for e in range(n):
                set_scale(alpha, 1.0, e, 0.0)
                value = float(forward_loss(target))
                removal[e] = value - identity_base
                repeat_error[e] = abs(float(forward_loss(target)) - value)
                set_scale(alpha, beta_work, e, 0.0)
                removal_work[e] = float(forward_loss(target)) - work_losses[0]
        finite_differences = []
        for e in torch.argsort(gradients[0].abs(), descending=True)[:3].tolist():
            for step in (0.005, 0.01):
                with torch.no_grad():
                    set_scale(alpha, beta_work, e, beta_work + step)
                    plus = float(forward_loss(target))
                    set_scale(alpha, beta_work, e, beta_work - step)
                    minus = float(forward_loss(target))
                finite_differences.append({"expert": e, "step": step, "gradient_sum": float(gradients[0][e]),
                                           "central_difference_sum": (plus - minus) / (2 * step)})
        set_scale(alpha, beta_work)
    return {"gradient": gradients[0], "identity_gradient": identity_gradient,
            "removal": removal, "removal_work": removal_work,
            "repeat_loss_error": max(float(repeat_error.max()), abs(work_losses[1] - work_losses[0])),
            "repeat_gradient_error": float((gradients[1] - gradients[0]).abs().max()),
            "base_identity": identity_base, "base_work": work_losses[0],
            "num_score_tokens": int(mask.sum()), "finite_differences": finite_differences}


def collect(args):
    reference_path = Path(args.identity_input).resolve()
    reference = _torch_load(reference_path)
    refmeta = reference["metadata"]
    ref = reference["layers"][args.layer]
    scores = _torch_load(Path(args.scores))
    metadata = scores["metadata"]
    manifest_path = Path(refmeta["selection_manifest"])
    manifest, manifest_sha = _load_selection_manifest(str(manifest_path))
    if manifest_sha != refmeta["selection_manifest_sha256"] or manifest_sha != metadata["selection_manifest_sha256"]:
        raise ValueError("Frozen manifest differs from the identity Hessian/calibration scores.")
    if refmeta["loss_fn"] != args.loss_fn or refmeta["validation_dtype"] != "float32":
        raise ValueError("Reference loss/dtype does not match this experiment.")
    if refmeta["model_name_or_path"] != metadata["model_name_or_path"]:
        raise ValueError("Reference model differs from calibration model.")
    num_samples = 1 if args.smoke else int(refmeta["num_samples"])
    samples = manifest["samples"][:num_samples]
    loader = DataLoader(ManifestRawDataset(samples, num_video_frames=int(manifest.get("num_video_frames", 8)),
                                          video_max_long_side=int(manifest.get("video_max_long_side", 480))),
                        batch_size=int(refmeta["batch_size"]), shuffle=False, collate_fn=_identity_collate)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw.pt").exists():
        raise FileExistsError(f"Output already collected: {output}")
    bundle = load_model_bundle(refmeta["model_name_or_path"], device_map=args.device_map,
                               attn_implementation=metadata.get("attn_implementation") or "sdpa")
    source = teacher_block(bundle, args.layer)
    block = copy.deepcopy(source).float().eval().requires_grad_(False)
    device = next(block.parameters()).device
    alpha = torch.ones(int(block.mlp.experts.num_experts), device=device, requires_grad=True)
    patch = patch_expert_output_alpha_vector(block.mlp.experts, alpha)
    recorder = RouteRecorder(block.mlp.experts)
    teacher_state = {}
    hook = register_teacher_block_hook(source, teacher_state)
    batches, route_comparisons = [], 0
    try:
        for batch in tqdm(loader, desc=f"beta095 {args.loss_fn} all experts"):
            inputs, kwargs, mask = _prepare_block_inputs(bundle, bundle.model, source, block, batch, metadata, teacher_state)
            recorder.reset()
            batches.append(measure_batch(block, alpha, inputs, kwargs, mask, loss_fn=args.loss_fn, beta_work=args.beta_work))
            route_comparisons += recorder.calls - 1
            torch.save({"batches": batches, "complete": False}, output / "progress.pt")
        tokens = sum(b["num_score_tokens"] for b in batches)
        if not args.smoke and tokens != ref["num_score_tokens"]:
            raise ValueError("Score-token count differs from the identity Hessian reference.")
        runmeta = {**refmeta, "schema_version": 1, "num_samples": num_samples, "layer": args.layer,
                   "beta_work": args.beta_work, "teacher_background": 1.0, "gradient_background": args.beta_work,
                   "ground_truth_background": 1.0, "hessian_background": 1.0,
                   "parameterization": "absolute_expert_scale", "loss_arithmetic_dtype": "float64",
                   "normalization": ("sum of hidden-dimension MSE / score tokens" if args.loss_fn == "l2"
                                     else "sum of KL(softmax(teacher_hidden) || softmax(student_hidden)) / score tokens"),
                   "identity_input": str(reference_path), "identity_input_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                   "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "num_score_tokens": tokens, "smoke_only": args.smoke,
                   "route_consistency_verified": True, "route_comparisons": route_comparisons,
                   "removal_source": "independent forwards for every expert at backgrounds 1 and beta_work"}
        summed = {key: sum((b[key] for b in batches), torch.zeros(alpha.numel(), dtype=torch.float64)) / tokens
                  for key in ("gradient", "identity_gradient", "removal", "removal_work")}
        base_work = sum(b["base_work"] for b in batches) / tokens
        base_identity = sum(b["base_identity"] for b in batches) / tokens
        rows = []
        for e in range(alpha.numel()):
            if int(ref["active_batch_counts"][e]) <= 0:
                continue
            g = float(summed["gradient"][e])
            rows.append({"layer": args.layer, "expert": e, "loss_fn": args.loss_fn, "beta_work": args.beta_work,
                         "num_score_tokens": tokens, "reference_active_batch_count": int(ref["active_batch_counts"][e]),
                         "base_loss_at_identity": base_identity, "base_loss_at_work": base_work,
                         "gradient_at_identity": float(summed["identity_gradient"][e]), "gradient_at_work": g,
                         "first_order_signed": -args.beta_work * g, "first_order_abs": abs(args.beta_work * g),
                         "hessian_half_at_identity": float(ref["hessian_score_per_token"][e]),
                         "reference_removal_at_identity": float(ref["ablation_per_token"][e]),
                         "true_removal_at_identity": float(summed["removal"][e]),
                         "removal_loss_at_identity": base_identity + float(summed["removal"][e]),
                         "true_removal_delta_at_work": float(summed["removal_work"][e]),
                         "removal_loss_at_work": base_work + float(summed["removal_work"][e])})
        torch.save({"metadata": runmeta, "batches": batches, "measured": summed}, output / "raw.pt")
        (output / "metadata.json").write_text(json.dumps(runmeta, indent=2) + "\n")
        if args.smoke:
            # A one-sample gradient must never be joined to a 32-sample Hessian.
            print(f"Smoke measurements saved to {output}; no cross-sample ranking produced.", flush=True)
            return
        summary = export_analysis(output, rows,
                                  repeat_loss_error=sum(b["repeat_loss_error"] for b in batches) / tokens,
                                  repeat_gradient_error=args.beta_work * sum(b["repeat_gradient_error"] for b in batches) / tokens,
                                  hessian_rtol=args.hessian_match_rtol)
        selected = {item["expert"]: item["group"] for item in summary["representatives"]}
        for pair in summary["display_pairs"]:
            for key in ("expert_high", "expert_low"):
                selected.setdefault(pair[key], "counterexample")
        curves = {(e, background, beta): [0.0, 0.0, 0.0] for e in sorted(selected)
                  for background in (1.0, args.beta_work) for beta in BETAS}
        sweep_tokens = 0
        for batch in tqdm(loader, desc=f"beta095 {args.loss_fn} selected curves"):
            inputs, kwargs, mask = _prepare_block_inputs(bundle, bundle.model, source, block, batch, metadata, teacher_state)
            recorder.reset()
            set_scale(alpha, 1.0)
            with torch.no_grad(), suspend_tensor_saving(block):
                target = unwrap_output(block(*inputs, **kwargs)).detach()
            for background in (1.0, args.beta_work):
                set_scale(alpha, background)
                with torch.no_grad(), suspend_tensor_saving(block):
                    base = float(loss_sum(unwrap_output(block(*inputs, **kwargs)), target, mask, args.loss_fn))
                for e in sorted(selected):
                    for beta in BETAS:
                        set_scale(alpha, background, e, beta)
                        with suspend_tensor_saving(block):
                            loss = loss_sum(unwrap_output(block(*inputs, **kwargs)), target, mask, args.loss_fn)
                            grad = torch.autograd.grad(loss, alpha)[0]
                        item = curves[e, background, beta]
                        item[0] += float(loss.detach())
                        item[1] += float(loss.detach()) - base
                        item[2] += float(grad[e].detach())
            route_comparisons += recorder.calls - 1
            sweep_tokens += int(mask.sum())
        if sweep_tokens != tokens:
            raise ValueError("Sweep did not reuse the complete scoring set.")
        curve_rows = [{"layer": args.layer, "expert": e, "selection_group": selected[e], "background_beta": background,
                       "beta": beta, "measured_loss": values[0] / tokens, "measured_delta_loss": values[1] / tokens,
                       "local_gradient_at_beta": values[2] / tokens}
                      for (e, background, beta), values in curves.items()]
        by_expert = {row["expert"]: row for row in rows}
        for curve in curve_rows:
            row = by_expert[curve["expert"]]
            if curve["beta"] == 0:
                expected = row["true_removal_at_identity"] if curve["background_beta"] == 1 else row["true_removal_delta_at_work"]
                if abs(curve["measured_delta_loss"] - expected) > max(summary["truth_tolerance"], abs(expected) * 1e-5):
                    raise ValueError("Sweep beta=0 disagrees with independent deletion forward.")
            if curve["beta"] == curve["background_beta"] and abs(curve["measured_delta_loss"]) > summary["truth_tolerance"]:
                raise ValueError("Sweep baseline is not zero.")
        write_csv(output / "beta_curves.csv", curve_rows)
        runmeta.update(route_comparisons=route_comparisons, betas=list(BETAS), curve_backgrounds=[1.0, args.beta_work],
                       selected_experts=selected, complete=True)
        (output / "metadata.json").write_text(json.dumps(runmeta, indent=2) + "\n")
        print(json.dumps({"output": str(output), "counterexamples": summary["counterexample_counts"], "metrics": summary["metrics"]}), flush=True)
    finally:
        hook.remove()
        recorder.restore()
        _restore_patched_expert(patch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--identity-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loss-fn", choices=("l2", "kl_div"), required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--beta-work", type=float, default=0.95)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--hessian-match-rtol", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.beta_work != 0.95:
        parser.error("This experiment fixes all student experts at beta=0.95.")
    if not 0 < args.hessian_match_rtol < 1:
        parser.error("Hessian match rtol must lie between zero and one.")
    collect(args)


if __name__ == "__main__":
    main()
