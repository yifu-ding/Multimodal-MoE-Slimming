"""Threshold calibration: learn per-expert, per-modality pruning thresholds
that minimise blockwise reconstruction loss across a grid of pruning ratios.

Usage (standalone)::

    python -m src.calibration.threshold_calibration \
        --model_name_or_path moonshotai/Kimi-VL-A3B-Instruct \
        --scores_path storage/prune/scores/kimi_gqa/scores.pt \
        --output_dir storage/prune/thresholds/kimi_gqa \
        --num_samples 128

The output ``thresholds.pt`` maps each (layer, expert, pruning_ratio) to an
optimal ``(text_thresh, visual_thresh)`` pair.
"""

import argparse
import copy
import os
import random
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.calibration.common import (
    build_dataset,
    custom_collate_fn,
    discover_layer_structure,
    ensure_dir,
    load_model_bundle,
    move_inputs_to_model_device,
    prepare_inputs,
)
from src.calibration.forward import (
    _kimi_teacher_block,
    _move_to_device_dtype,
    _enable_input_grads,
    _register_teacher_block_hook,
    compute_block_loss,
    unwrap_output,
)

__all__ = [
    "compute_soft_union_mask",
    "threshold_calibration_forward",
    "run_threshold_calibration",
]

DEFAULT_PRUNING_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _fmt_tensor_decimals(t: torch.Tensor, ndigits: int = 4) -> str:
    """Pretty-print tensor as nested lists with uniform float formatting."""

    def _rec(x: Any) -> str:
        if isinstance(x, list):
            return "[" + ", ".join(_rec(v) for v in x) + "]"
        return f"{float(x):.{ndigits}f}"

    return _rec(t.detach().cpu().float().tolist())


# ---------------------------------------------------------------------------
# Soft-mask helpers
# ---------------------------------------------------------------------------

def compute_soft_union_mask(
    text_score: torch.Tensor,
    visual_score: torch.Tensor,
    text_thresh: torch.Tensor,
    visual_thresh: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Differentiable union mask via sigmoid relaxation.

    Args:
        text_score:    [E, I] pre-collected text channel scores (detached).
        visual_score:  [E, I] pre-collected visual channel scores (detached).
        text_thresh:   [E]    learnable text thresholds.
        visual_thresh: [E]    learnable visual thresholds.
        tau:           temperature (smaller → harder mask).

    Returns:
        soft_mask [E, I] in (0, 1).
    """
    m_text = torch.sigmoid((text_score - text_thresh[:, None]) / tau)
    m_visual = torch.sigmoid((visual_score - visual_thresh[:, None]) / tau)
    return 1.0 - (1.0 - m_text) * (1.0 - m_visual)


def _bisect_union_quantile(
    text_vec: torch.Tensor,
    visual_vec: torch.Tensor,
    target_keep: float,
    tol: float = 0.005,
    max_iter: int = 40,
) -> Tuple[float, float]:
    """Binary-search for a shared quantile ``q`` such that the union hard mask
    ``(text >= quantile(text, q)) | (visual >= quantile(visual, q))``
    achieves ``mean(keep) ≈ target_keep``.

    Returns ``(text_thresh, visual_thresh)``.
    """
    lo, hi = 0.0, 1.0
    tf = text_vec.float()
    vf = visual_vec.float()
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        t_th = torch.quantile(tf, mid)
        v_th = torch.quantile(vf, mid)
        keep = float(((tf >= t_th) | (vf >= v_th)).float().mean().item())
        if abs(keep - target_keep) < tol:
            return float(t_th.item()), float(v_th.item())
        if keep > target_keep:
            lo = mid
        else:
            hi = mid
    t_th = torch.quantile(tf, (lo + hi) / 2.0)
    v_th = torch.quantile(vf, (lo + hi) / 2.0)
    return float(t_th.item()), float(v_th.item())


def init_thresholds(
    text_score: torch.Tensor,
    visual_score: torch.Tensor,
    pruning_ratios: List[float],
    device: torch.device,
) -> Tuple[nn.Parameter, nn.Parameter]:
    """Initialise per-expert thresholds so that the union hard mask already
    approximates the target keep ratio for each pruning ratio.

    Uses binary search to compensate for the inflating effect of OR-union.

    Returns:
        text_thresh  [E, R]  (nn.Parameter)
        visual_thresh [E, R] (nn.Parameter)
    """
    E = text_score.shape[0]
    R = len(pruning_ratios)
    t_buf = torch.zeros(E, R, device=device)
    v_buf = torch.zeros(E, R, device=device)
    for k, ratio in enumerate(pruning_ratios):
        target_keep = 1.0 - min(max(ratio, 0.0), 1.0)
        for e in range(E):
            t_val, v_val = _bisect_union_quantile(
                text_score[e], visual_score[e], target_keep
            )
            t_buf[e, k] = t_val
            v_buf[e, k] = v_val
    return nn.Parameter(t_buf), nn.Parameter(v_buf)


# ---------------------------------------------------------------------------
# Expert-forward patching: inject per-expert soft channel mask
# ---------------------------------------------------------------------------

def _patch_masked_moe_infer(block: nn.Module, layer_idx: int):
    """Patch ``moe_infer`` so that each expert's intermediate activations are
    multiplied by ``self._current_soft_mask[eid]`` (set externally per ratio).
    """
    mlp = getattr(block, "mlp", None)
    if mlp is None or not hasattr(mlp, "moe_infer"):
        return None
    original = mlp.moe_infer

    def _masked_moe_infer(self, x, topk_ids, topk_weight, **kwargs):
        soft_mask = getattr(self, "_current_soft_mask", None)  # [E, I]

        if hasattr(self, "gate_dict") and self.gate_dict is not None:
            valid_mask = self.valid_expert_mask.to(topk_weight.device)
            topk_weight = topk_weight * valid_mask
            topk_ids = topk_ids.clone()
            topk_ids[~valid_mask] = len(self.experts)

        idxs = topk_ids.view(-1).argsort()
        flat_token_idx = idxs // topk_ids.shape[1]
        flat_routing_weight = topk_weight.reshape(-1)[idxs]
        sorted_tokens = x[flat_token_idx]
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts) + 1))
        src = torch.ones_like(topk_ids, dtype=cnts.dtype, device=cnts.device)
        cnts.scatter_add_(1, topk_ids, src)
        tokens_per_expert = cnts.sum(dim=0).tolist()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + int(num_tokens)
            if num_tokens == 0:
                continue
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            if i == len(self.experts):
                outputs.append(tokens_for_this_expert)
                break
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]

            gate_out = expert.gate_proj(tokens_for_this_expert)
            up_out = expert.up_proj(tokens_for_this_expert)
            h = expert.act_fn(gate_out) * up_out

            if soft_mask is not None and i < soft_mask.shape[0]:
                h = h * soft_mask[i].to(h.device, h.dtype)

            expert_out = expert.down_proj(h)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = (
            torch.cat(outputs, dim=0)
            if outputs
            else sorted_tokens.new_empty((0, x.shape[-1]))
        )
        new_x = torch.empty_like(outs)
        if idxs.numel() > 0:
            new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    mlp.moe_infer = types.MethodType(_masked_moe_infer, mlp)
    return mlp, original


# ---------------------------------------------------------------------------
# Single-layer calibration
# ---------------------------------------------------------------------------

def threshold_calibration_forward(
    bundle,
    layer_idx: int,
    text_score: torch.Tensor,
    visual_score: torch.Tensor,
    pruning_ratios: List[float],
    dataloader,
    dataset_name: str,
    *,
    num_epochs: int = 3,
    lr: float = 0.01,
    penalty_lambda: float = 100.0,
    tau_start: float = 0.1,
    tau_end: float = 0.01,
    anneal: bool = True,
    loss_fn: str = "rel_l2",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Calibrate per-expert pruning thresholds for one MoE layer.

    Returns dict with keys ``text_thresh``, ``visual_thresh`` (each ``[E, R]``),
    ``calibration_loss`` ``[R]``, ``actual_keep_ratio`` ``[R]``.
    """
    model = bundle.model
    model.eval()
    teacher_block = list(_kimi_teacher_block(model))[layer_idx]
    block_device = next(teacher_block.parameters()).device
    copied_block = copy.deepcopy(teacher_block)
    copied_block = copied_block.to(device=block_device, dtype=dtype)
    copied_block.eval()

    R = len(pruning_ratios)
    text_score_dev = text_score.to(device=block_device, dtype=torch.float32)
    visual_score_dev = visual_score.to(device=block_device, dtype=torch.float32)

    text_thresh, visual_thresh = init_thresholds(
        text_score_dev, visual_score_dev, pruning_ratios, block_device
    )
    print(
        f"[threshold_cal] Layer {layer_idx} init\n"
        f"  keep_ratio(target)={[1-x for x in pruning_ratios]}\n"
        f"  text_thresh={_fmt_tensor_decimals(text_thresh[0])}\n"
        f"  visual_thresh={_fmt_tensor_decimals(visual_thresh[0])}",
        flush=True,
    )

    optimizer = torch.optim.Adam([text_thresh, visual_thresh], lr=lr)

    teacher_state: Dict[str, Any] = {}
    teacher_handle = _register_teacher_block_hook(teacher_block, teacher_state)
    patch_state = _patch_masked_moe_infer(copied_block, layer_idx)

    device_type = block_device.type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    # Modality mask construction helpers
    special_ids = getattr(model, "special_token_id_tensor", None)
    media_token_id = getattr(model.config, "media_placeholder_token_id", None)

    total_step_count = 0

    try:
        for epoch in range(num_epochs):
            if anneal and num_epochs > 1:
                tau = tau_start * (tau_end / tau_start) ** (epoch / (num_epochs - 1))
            else:
                tau = tau_start

            ep_loss = torch.zeros(R, dtype=torch.float32)
            ep_keep = torch.zeros(R, dtype=torch.float32)
            ep_steps = 0

            pbar_desc = f"ThreshCal L{layer_idx} ep{epoch} tau={tau:.4f}"
            iterator = tqdm(
                dataloader, desc=pbar_desc, disable=not verbose, leave=False
            )
            for batch in iterator:
                teacher_state.clear()
                inputs = prepare_inputs(bundle, batch, dataset_name)
                inputs = move_inputs_to_model_device(model, inputs)
                attn_mask = inputs["attention_mask"].to(block_device)
                input_ids = inputs.get("input_ids", None)

                if input_ids is not None and hasattr(copied_block, "mlp"):
                    if special_ids is not None and media_token_id is not None:
                        sid = special_ids.to(input_ids.device)
                        copied_block.mlp.moe_text_mask = (
                            ~torch.isin(input_ids, sid).view(-1)
                        )[:, None]
                        copied_block.mlp.moe_media_mask = (
                            (input_ids == media_token_id).view(-1)
                        )[:, None]

                with torch.no_grad():
                    model(**inputs, use_cache=False, return_dict=True)

                if not teacher_state:
                    raise RuntimeError(
                        f"Teacher hook did not fire for layer {layer_idx}."
                    )

                in_args = _enable_input_grads(
                    _move_to_device_dtype(teacher_state["in_args"], block_device, dtype)
                )
                in_kwargs = _enable_input_grads(
                    _move_to_device_dtype(
                        teacher_state["in_kwargs"], block_device, dtype
                    )
                )
                teacher_target = unwrap_output(teacher_state["output"])
                teacher_target = _move_to_device_dtype(
                    teacher_target, block_device, dtype
                )

                for k in range(R):
                    soft_mask = compute_soft_union_mask(
                        text_score_dev,
                        visual_score_dev,
                        text_thresh[:, k],
                        visual_thresh[:, k],
                        tau,
                    )
                    copied_block.mlp._current_soft_mask = soft_mask

                    copied_block.zero_grad(set_to_none=True)
                    optimizer.zero_grad()
                    with torch.autocast(
                        device_type=device_type,
                        dtype=dtype,
                        enabled=autocast_enabled,
                    ):
                        pred = unwrap_output(copied_block(*in_args, **in_kwargs))
                        loss_recon, _ = compute_block_loss(
                            pred=pred,
                            teacher_target=teacher_target,
                            attn_mask=attn_mask,
                            loss_fn=loss_fn,
                        )

                    keep_ratio = soft_mask.mean()
                    target_keep = 1.0 - pruning_ratios[k]
                    loss_penalty = penalty_lambda * (keep_ratio - target_keep) ** 2
                    loss = loss_recon + loss_penalty
                    loss.backward()
                    optimizer.step()

                    ep_loss[k] += float(loss_recon.detach().item())
                    ep_keep[k] += float(keep_ratio.detach().item())

                ep_steps += 1
                total_step_count += 1


            ###############################################################
            # 换 hard mask 来验证，而不是 annealing 的 soft mask             #
            ###############################################################
            
            # -- hard-mask validation (no grad, detached) --
            hard_loss_accum = torch.zeros(R, dtype=torch.float32)
            hard_keep_accum = torch.zeros(R, dtype=torch.float32)
            hard_steps = 0
            with torch.no_grad():
                for batch in tqdm(
                    dataloader, desc=f"ThreshCal L{layer_idx} ep{epoch} hard-eval", leave=False
                ):
                    teacher_state.clear()
                    val_inputs = prepare_inputs(bundle, batch, dataset_name)
                    val_inputs = move_inputs_to_model_device(model, val_inputs)
                    val_attn = val_inputs["attention_mask"].to(block_device)
                    model(**val_inputs, use_cache=False, return_dict=True)
                    if not teacher_state:
                        continue
                    val_in_args = _move_to_device_dtype(
                        teacher_state["in_args"], block_device, dtype
                    )
                    val_in_kwargs = _move_to_device_dtype(
                        teacher_state["in_kwargs"], block_device, dtype
                    )
                    val_target = unwrap_output(teacher_state["output"])
                    val_target = _move_to_device_dtype(val_target, block_device, dtype)

                    for k in range(R):
                        hm = (
                            (text_score_dev >= text_thresh[:, k : k + 1])
                            | (visual_score_dev >= visual_thresh[:, k : k + 1])
                        ).float()
                        copied_block.mlp._current_soft_mask = hm
                        with torch.autocast(
                            device_type=device_type,
                            dtype=dtype,
                            enabled=autocast_enabled,
                        ):
                            hpred = unwrap_output(
                                copied_block(*val_in_args, **val_in_kwargs)
                            )
                            hloss, _ = compute_block_loss(
                                pred=hpred,
                                teacher_target=val_target,
                                attn_mask=val_attn,
                                loss_fn=loss_fn,
                            )
                        hard_loss_accum[k] += float(hloss.item())
                        hard_keep_accum[k] += float(hm.mean().item())
                    hard_steps += 1

            if verbose:
                avg_loss = ep_loss / max(ep_steps, 1)
                avg_keep = ep_keep / max(ep_steps, 1)
                avg_hard_loss = hard_loss_accum / max(hard_steps, 1)
                avg_hard_keep = hard_keep_accum / max(hard_steps, 1)
                headers = [f"r{r:.1f}" for r in pruning_ratios]
                print(f"[ThreshCal] L{layer_idx} epoch={epoch} tau={tau:.4f}")
                print(" ratio  " + "  ".join(f"{h:>8}" for h in headers))
                print("  keep  " + "  ".join(f"{float(v):>8.4f}" for v in avg_keep))
                print("  loss  " + "  ".join(f"{float(v):>8.4f}" for v in avg_loss))
                print(" hkeep  " + "  ".join(f"{float(v):>8.4f}" for v in avg_hard_keep))
                print(" hloss  " + "  ".join(f"{float(v):>8.4f}" for v in avg_hard_loss))
                print(
                    f"[threshold_cal] Layer {layer_idx} epoch end\n"
                    f"  text_thresh={_fmt_tensor_decimals(text_thresh[0])}\n"
                    f"  visual_thresh={_fmt_tensor_decimals(visual_thresh[0])}",
                    flush=True,
                )

    finally:
        teacher_handle.remove()
        if patch_state is not None:
            mlp, original_moe_infer = patch_state
            mlp.moe_infer = original_moe_infer
        if hasattr(copied_block.mlp, "_current_soft_mask"):
            del copied_block.mlp._current_soft_mask

    # Compute hard-threshold keep ratios for the final result
    with torch.no_grad():
        actual_keep = torch.zeros(R, dtype=torch.float32)
        for k in range(R):
            hard_mask = (
                (text_score_dev >= text_thresh[:, k : k + 1])
                | (visual_score_dev >= visual_thresh[:, k : k + 1])
            ).float()
            actual_keep[k] = hard_mask.mean().item()

    return {
        "text_thresh": text_thresh.detach().cpu(),      # [E, R]
        "visual_thresh": visual_thresh.detach().cpu(),   # [E, R]
        "calibration_loss": ep_loss / max(ep_steps, 1), # [R] (last epoch avg)
        "actual_keep_ratio": actual_keep,                # [R]
    }


# ---------------------------------------------------------------------------
# Full calibration across all layers
# ---------------------------------------------------------------------------

def _load_modality_scores(
    scores_path: str,
    device: str = "cpu",
    intra_expert_metric: str = "activation",
):
    """Load per-modality channel scores from a scores payload.

    The field names are ``{intra_expert_metric}_text`` and
    ``{intra_expert_metric}_visual`` inside ``expert_scores``.

    Returns:
        text_scores  {layer_idx: Tensor[E, I]}
        visual_scores {layer_idx: Tensor[E, I]}
        layers       list[int]
    """
    payload = torch.load(scores_path, map_location=device, weights_only=False)
    es = payload.get("expert_scores", {})

    def _to_layer_tensors(nested):
        out = {}
        for lid in sorted(nested.keys()):
            eids = sorted(nested[lid].keys())
            out[lid] = torch.stack(
                [nested[lid][eid].detach().float() for eid in eids], dim=0
            )
        return out

    text_key = f"{intra_expert_metric}_text"
    visual_key = f"{intra_expert_metric}_visual"
    text_nested = es.get(text_key)
    visual_nested = es.get(visual_key)
    if text_nested is None or visual_nested is None:
        available = sorted(es.keys())
        raise ValueError(
            f"scores.pt must contain expert_scores.{text_key} and "
            f"expert_scores.{visual_key}. "
            f"Available keys: {available}"
        )
    text_scores = _to_layer_tensors(text_nested)
    visual_scores = _to_layer_tensors(visual_nested)
    layers = payload.get("layers", sorted(text_scores.keys()))
    return text_scores, visual_scores, layers


def run_threshold_calibration(args) -> None:
    """Main entry point: load model + scores, calibrate all layers, save."""
    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, "thresholds.pt")
    if os.path.exists(out_path) and not args.force:
        print(
            f"[threshold_cal] Found existing thresholds at {out_path}. "
            "Pass --force to overwrite."
        )
        return

    pruning_ratios = [float(x) for x in args.pruning_ratios.split(",")]

    text_scores, visual_scores, layers = _load_modality_scores(
        args.scores_path, device="cpu",
        intra_expert_metric=args.intra_expert_metric,
    )

    bundle = load_model_bundle(args.model_name_or_path)
    if bundle.family != "kimi":
        raise NotImplementedError("Threshold calibration currently only supports Kimi-VL.")

    dataset = build_dataset(args.dataset, bundle.family)
    pool = list(range(args.start_idx, len(dataset)))
    if args.subset_seed is not None and args.subset_seed >= 0:
        rng = random.Random(args.subset_seed)
        indices = rng.sample(pool, min(args.num_samples, len(pool)))
    else:
        indices = pool[: args.num_samples]

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    block_dtype = next(bundle.model.parameters()).dtype

    result_text_thresh: Dict[int, torch.Tensor] = {}
    result_visual_thresh: Dict[int, torch.Tensor] = {}
    result_loss: Dict[int, torch.Tensor] = {}
    result_keep: Dict[int, torch.Tensor] = {}

    for layer_idx in layers:
        if layer_idx not in text_scores or layer_idx not in visual_scores:
            print(f"[threshold_cal] Skipping layer {layer_idx}: no modality scores.")
            continue

        layer_result = threshold_calibration_forward(
            bundle=bundle,
            layer_idx=layer_idx,
            text_score=text_scores[layer_idx],
            visual_score=visual_scores[layer_idx],
            pruning_ratios=pruning_ratios,
            dataloader=loader,
            dataset_name=args.dataset,
            num_epochs=args.num_epochs,
            lr=args.lr,
            penalty_lambda=args.penalty_lambda,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            anneal=not args.no_anneal,
            loss_fn=args.loss_fn,
            dtype=block_dtype,
            verbose=True,
        )

        result_text_thresh[layer_idx] = layer_result["text_thresh"]
        result_visual_thresh[layer_idx] = layer_result["visual_thresh"]
        result_loss[layer_idx] = layer_result["calibration_loss"]
        result_keep[layer_idx] = layer_result["actual_keep_ratio"]

        print(
            f"[threshold_cal] Layer {layer_idx} done\n"
            f"  keep_ratio={_fmt_tensor_decimals(layer_result['actual_keep_ratio'])}\n"
            f"  text_thresh={_fmt_tensor_decimals(layer_result['text_thresh'][0])}\n"
            f"  visual_thresh={_fmt_tensor_decimals(layer_result['visual_thresh'][0])}",
            flush=True,
        )

        # Incremental save
        _save_thresholds(
            out_path,
            result_text_thresh,
            result_visual_thresh,
            result_loss,
            result_keep,
            pruning_ratios,
            args,
        )

    print(f"[threshold_cal] All layers complete. Saved to {out_path}")


def _save_thresholds(
    path: str,
    text_thresh: Dict[int, torch.Tensor],
    visual_thresh: Dict[int, torch.Tensor],
    cal_loss: Dict[int, torch.Tensor],
    keep_ratio: Dict[int, torch.Tensor],
    pruning_ratios: List[float],
    args,
) -> None:
    payload = {
        "text_thresh": text_thresh,
        "visual_thresh": visual_thresh,
        "pruning_ratios": torch.tensor(pruning_ratios, dtype=torch.float32),
        "calibration_loss": cal_loss,
        "actual_keep_ratio": keep_ratio,
        "tau_start": args.tau_start,
        "tau_end": args.tau_end,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "penalty_lambda": args.penalty_lambda,
        "metadata": {
            "model_name_or_path": args.model_name_or_path,
            "scores_path": args.scores_path,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "loss_fn": args.loss_fn,
            "intra_expert_metric": args.intra_expert_metric,
        },
    }
    torch.save(payload, path)
    print(f"[threshold_cal] Saved to {path}")


# ---------------------------------------------------------------------------
# Mask generation from calibrated thresholds
# ---------------------------------------------------------------------------

def generate_mask_from_thresholds(
    thresholds_path: str,
    scores_path: str,
    target_ratio: float,
    device: str = "cpu",
    intra_expert_metric: str = "activation",
) -> Dict[str, Any]:
    """Build binary pruning masks from a calibrated ``thresholds.pt``.

    Returns a dict compatible with ``generate_masks`` output:
    ``intermediate_masks`` ``[L, E, I]`` bool, ``K_E_inter``, ``layers``.
    """
    thresh = torch.load(thresholds_path, map_location=device, weights_only=False)
    ratios = thresh["pruning_ratios"]
    k = int((ratios - target_ratio).abs().argmin().item())

    text_scores, visual_scores, layers = _load_modality_scores(
        scores_path, device, intra_expert_metric=intra_expert_metric
    )

    L = len(layers)
    first_layer = layers[0]
    E, I = text_scores[first_layer].shape
    masks = torch.zeros(L, E, I, dtype=torch.bool, device=device)

    for lid_pos, layer_idx in enumerate(layers):
        t_th = thresh["text_thresh"][layer_idx][:, k]   # [E]
        v_th = thresh["visual_thresh"][layer_idx][:, k]  # [E]
        ts = text_scores[layer_idx].to(device)
        vs = visual_scores[layer_idx].to(device)
        masks[lid_pos] = (ts >= t_th[:, None]) | (vs >= v_th[:, None])

    return {
        "intermediate_masks": masks,
        "K_E_inter": masks.sum(dim=-1),
        "layers": layers,
        "layerwise_keep_plan": masks.float().mean(dim=(1, 2)),
        "selected_ratio_idx": k,
        "selected_ratio": float(ratios[k].item()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calibrate per-expert pruning thresholds via blockwise loss."
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--scores_path", type=str, required=True,
                    help="Path to scores.pt with modality-split channel scores.")
    p.add_argument("--intra_expert_metric", type=str, default="activation",
                    help="Base metric name in expert_scores (fields: {metric}_text, {metric}_visual).",
                    choices=["activation", "down_second_order"],
                    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="gqa", choices=["gqa", "coco"])
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--pruning_ratios", type=str,
                    default=",".join(str(r) for r in DEFAULT_PRUNING_RATIOS),
                    help="Comma-separated pruning ratios.")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--penalty_lambda", type=float, default=100.0)
    p.add_argument("--tau_start", type=float, default=0.1)
    p.add_argument("--tau_end", type=float, default=0.01)
    p.add_argument("--no_anneal", action="store_true")
    p.add_argument("--loss_fn", type=str, default="rel_l2",
                    choices=["l2", "rel_l2", "cosine"])
    p.add_argument("--force", "-f", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_threshold_calibration(args)


if __name__ == "__main__":
    main()
