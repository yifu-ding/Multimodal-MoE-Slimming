import argparse
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from tqdm import trange

from src.calibration.helpers.helpers import compute_block_loss
from src.calibration.representation_distill.common import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    ensure_dir,
    get_decoder_layer,
    seed_everything,
    sort_sequence_by_position_ids,
    utc_now_iso,
)
from src.calibration.representation_distill.runtime.forward_from_hidden import forward_from_hidden


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_rows(tensor: torch.Tensor, count: int) -> torch.Tensor:
    if count >= tensor.shape[0]:
        return tensor
    indices = torch.randint(0, tensor.shape[0], (count,), device=tensor.device)
    return tensor.index_select(0, indices)


def _subsample_flat(tensor: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Subsample rows for efficient kernel computation."""
    if tensor.shape[0] <= max_tokens:
        return tensor
    indices = torch.randperm(tensor.shape[0], device=tensor.device)[:max_tokens]
    return tensor[indices]


def _build_synthetic_banks(
    init_hidden: torch.Tensor,
    init_labels: torch.Tensor | None,
    init_std: float,
) -> tuple[torch.nn.ParameterDict, torch.Tensor]:
    flat_hidden = init_hidden.reshape(-1, init_hidden.shape[-1])
    if init_labels is None:
        flat_labels = torch.full(
            (flat_hidden.shape[0],),
            MODALITY_TEXT,
            dtype=torch.int8,
            device=flat_hidden.device,
        )
    else:
        flat_labels = init_labels.reshape(-1).to(torch.int8)

    bank_params: dict[str, torch.nn.Parameter] = {}
    for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
        mod_mask = flat_labels == mod_id
        if int(mod_mask.sum()) == 0:
            continue
        bank_value = flat_hidden[mod_mask].clone()
        if init_std > 0:
            bank_value.add_(torch.randn_like(bank_value) * init_std)
        bank_params[str(int(mod_id))] = torch.nn.Parameter(bank_value)
    return torch.nn.ParameterDict(bank_params), flat_labels


def _assemble_synthetic_hidden(
    bank_params: torch.nn.ParameterDict,
    template_labels: torch.Tensor,
    synthetic_size: int,
    compressed_length: int,
    hidden_size: int,
) -> torch.Tensor:
    flat_labels = template_labels.reshape(-1)
    flat_hidden = torch.empty(
        flat_labels.shape[0],
        hidden_size,
        device=flat_labels.device,
        dtype=next(iter(bank_params.values())).dtype,
    )
    for mod_id_str, bank in bank_params.items():
        flat_hidden[flat_labels == int(mod_id_str)] = bank
    return flat_hidden.view(synthetic_size, compressed_length, hidden_size)


def _compute_next_block_rel_l2(
    *,
    bundle,
    layer_idx: int,
    teacher_hidden: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_position_ids: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    synthetic_attention_mask: torch.Tensor,
    synthetic_position_ids: torch.Tensor,
) -> torch.Tensor:
    block_layer = get_decoder_layer(bundle, layer_idx)
    block_dtype = next(block_layer.parameters()).dtype
    teacher_hidden = teacher_hidden.to(dtype=block_dtype)
    synthetic_hidden = synthetic_hidden.to(dtype=block_dtype)

    with torch.no_grad():
        teacher_target = forward_from_hidden(
            bundle=bundle,
            hidden_states=teacher_hidden,
            attention_mask=teacher_attention_mask,
            start_layer=layer_idx,
            end_layer=layer_idx,
            position_ids=teacher_position_ids,
            apply_final_norm=False,
        )
    synth_pred = forward_from_hidden(
        bundle=bundle,
        hidden_states=synthetic_hidden,
        attention_mask=synthetic_attention_mask,
        start_layer=layer_idx,
        end_layer=layer_idx,
        position_ids=synthetic_position_ids,
        apply_final_norm=False,
    )
    loss_sum, _ = compute_block_loss(
        pred=synth_pred,
        teacher_target=teacher_target,
        attn_mask=synthetic_attention_mask,
        loss_fn="rel_l2",
    )
    denom = synthetic_attention_mask.float().sum().clamp_min(1.0)
    return loss_sum / denom


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidth_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> torch.Tensor:
    """Unbiased MMD^2 estimate with multi-bandwidth RBF kernel.

    Parameters
    ----------
    X, Y : (N, D) and (M, D) token-level feature matrices.
    bandwidth_multipliers : scales applied to the median pairwise distance.
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.tensor(0.0, device=X.device)

    XX = torch.cdist(X, X).pow(2)
    YY = torch.cdist(Y, Y).pow(2)
    XY = torch.cdist(X, Y).pow(2)

    # Median heuristic for bandwidth
    with torch.no_grad():
        all_dists = torch.cat([XX.view(-1), YY.view(-1), XY.view(-1)])
        median_dist = all_dists.median().clamp(min=1e-6)

    mmd = torch.tensor(0.0, device=X.device)
    for mult in bandwidth_multipliers:
        bw_sq = 2.0 * (mult * median_dist)
        k_XX = (-XX / bw_sq).exp().mean()
        k_YY = (-YY / bw_sq).exp().mean()
        k_XY = (-XY / bw_sq).exp().mean()
        mmd = mmd + k_XX + k_YY - 2.0 * k_XY
    return mmd


def _cov_loss(teacher_flat: torch.Tensor, synth_flat: torch.Tensor) -> torch.Tensor:
    """MSE between cross-channel covariance matrices."""
    t_centered = teacher_flat - teacher_flat.mean(dim=0, keepdim=True)
    s_centered = synth_flat - synth_flat.mean(dim=0, keepdim=True)
    t_cov = (t_centered.T @ t_centered) / max(teacher_flat.shape[0] - 1, 1)
    s_cov = (s_centered.T @ s_centered) / max(synth_flat.shape[0] - 1, 1)
    return (t_cov - s_cov).pow(2).mean()


def _mean_var_loss(
    teacher_flat: torch.Tensor,
    synth_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel 1st/2nd moment MSE, returned as (mean_loss, var_loss)."""
    t_mean = teacher_flat.mean(dim=0)
    s_mean = synth_flat.mean(dim=0)
    t_var = teacher_flat.var(dim=0, unbiased=False)
    s_var = synth_flat.var(dim=0, unbiased=False)
    return (t_mean - s_mean).pow(2).mean(), (t_var - s_var).pow(2).mean()


def _diversity_loss(synthetic_flat: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity (lower = more diverse)."""
    if synthetic_flat.shape[0] <= 1:
        return torch.tensor(0.0, device=synthetic_flat.device)
    normed = F.normalize(synthetic_flat, dim=-1)
    sim = normed @ normed.T
    n = sim.shape[0]
    # Exclude diagonal
    off_diag = sim.masked_select(~torch.eye(n, dtype=torch.bool, device=sim.device))
    return off_diag.mean()


def _iter_modality_groups(
    teacher_flat: torch.Tensor,
    synth_flat: torch.Tensor,
    teacher_labels_flat: torch.Tensor | None,
    synth_labels_flat: torch.Tensor | None,
    min_tokens: int = 2,
):
    """Yield (modality_id, teacher_sub, synth_sub) token groups.

    When labels are missing, yields a single ``(None, teacher_flat, synth_flat)`` pair
    so that the caller can fall back to a global (pooled) computation.
    """
    if teacher_labels_flat is None or synth_labels_flat is None:
        yield None, teacher_flat, synth_flat
        return

    for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
        t_mask = teacher_labels_flat == mod_id
        s_mask = synth_labels_flat == mod_id
        if int(t_mask.sum()) < min_tokens or int(s_mask.sum()) < min_tokens:
            continue
        yield mod_id, teacher_flat[t_mask], synth_flat[s_mask]


def _compute_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    teacher_labels: torch.Tensor | None,
    synth_labels: torch.Tensor | None,
    mmd_subsample: int,
) -> dict[str, torch.Tensor]:
    """All distillation losses, computed per-modality and averaged across modalities.

    Groups covered (MMD / cov / mean / var): text, image, video — whichever are
    present in *both* teacher and synthetic batches. When no modality labels are
    available the computation gracefully degrades to a single pooled pair.
    ``div`` stays pooled on the synthetic tokens (regardless of modality) because
    diversity should be measured on the whole synthetic set.
    """
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    teacher_labels_flat = teacher_labels.reshape(-1) if teacher_labels is not None else None
    synth_labels_flat = synth_labels.reshape(-1) if synth_labels is not None else None

    device = teacher_flat.device
    zero = torch.tensor(0.0, device=device)

    mmd_sum = zero.clone()
    cov_sum = zero.clone()
    mean_sum = zero.clone()
    var_sum = zero.clone()
    n_groups = 0
    per_group: dict[str, torch.Tensor] = {}

    for mod_id, t_group, s_group in _iter_modality_groups(
        teacher_flat, synth_flat, teacher_labels_flat, synth_labels_flat
    ):
        t_sub = _subsample_flat(t_group, mmd_subsample)
        s_sub = _subsample_flat(s_group, mmd_subsample)

        mmd_g = _mmd_rbf(t_sub, s_sub)
        cov_g = _cov_loss(t_sub, s_sub)
        mean_g, var_g = _mean_var_loss(t_group, s_group)

        mmd_sum = mmd_sum + mmd_g
        cov_sum = cov_sum + cov_g
        mean_sum = mean_sum + mean_g
        var_sum = var_sum + var_g
        n_groups += 1

        if mod_id is not None:
            per_group[f"mmd/mod{int(mod_id)}"] = mmd_g.detach()
            per_group[f"cov/mod{int(mod_id)}"] = cov_g.detach()
            per_group[f"mean/mod{int(mod_id)}"] = mean_g.detach()
            per_group[f"var/mod{int(mod_id)}"] = var_g.detach()

    if n_groups == 0:
        # Nothing matched (e.g. degenerate batch); fall back to pooled.
        t_sub = _subsample_flat(teacher_flat, mmd_subsample)
        s_sub = _subsample_flat(synth_flat, mmd_subsample)
        mmd_sum = _mmd_rbf(t_sub, s_sub)
        cov_sum = _cov_loss(t_sub, s_sub)
        mean_sum, var_sum = _mean_var_loss(teacher_flat, synth_flat)
        n_groups = 1

    div_loss = _diversity_loss(_subsample_flat(synth_flat, mmd_subsample))

    losses: dict[str, torch.Tensor] = {
        "mmd": mmd_sum / n_groups,
        "cov": cov_sum / n_groups,
        "mean": mean_sum / n_groups,
        "var": var_sum / n_groups,
        "div": div_loss,
    }
    losses.update(per_group)
    return losses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill a small synthetic hidden calibration set from teacher hidden cache."
    )
    parser.add_argument("--teacher_cache_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--synthetic_size", type=int, default=256)
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_std", type=float, default=0.0)
    # Distribution-aware loss weights
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_cov", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    # Legacy moment-matching (auxiliary, reduced weight)
    parser.add_argument("--lambda_mean", type=float, default=0.5)
    parser.add_argument("--lambda_var", type=float, default=0.5)
    parser.add_argument("--lambda_block", type=float, default=0.0)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    # Diversity warmup: avoids ``div`` dominating the first few hundred steps
    # when the other losses are still on the order of 1e-5.
    parser.add_argument("--div_warmup_steps", type=int, default=200,
                        help="Linearly ramp ``lambda_div`` from 0 to its full value over "
                             "this many steps. Set to 0 to disable.")
    # MMD efficiency
    parser.add_argument("--mmd_subsample", type=int, default=2048,
                        help="Max tokens for kernel matrix computation.")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    ensure_dir(os.path.dirname(args.output_path))

    cache_payload = torch.load(args.teacher_cache_path, map_location="cpu")
    teacher_cache = cache_payload["teacher_cache"].float()
    teacher_meta = cache_payload["metadata"]
    teacher_labels = cache_payload.get("modality_labels", None)
    teacher_position_ids = cache_payload.get("position_ids", None)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_cache = teacher_cache.to(device)
    if teacher_labels is not None:
        teacher_labels = teacher_labels.to(device)
    if teacher_position_ids is None:
        teacher_position_ids = torch.arange(
            teacher_cache.shape[1], device=device, dtype=torch.long
        ).unsqueeze(0).expand(teacher_cache.shape[0], -1)
    else:
        teacher_position_ids = teacher_position_ids.to(device=device, dtype=torch.long)
        teacher_cache, teacher_position_ids, teacher_labels = sort_sequence_by_position_ids(
            teacher_cache,
            teacher_position_ids,
            teacher_labels,
        )

    # Initialize from random teacher samples
    init_indices = torch.randint(0, teacher_cache.shape[0], (args.synthetic_size,), device=device)
    init_hidden = teacher_cache.index_select(0, init_indices).clone()
    init_position_ids = teacher_position_ids.index_select(0, init_indices).clone()

    # Modality labels: frozen, copied from init samples
    synth_labels = None
    if teacher_labels is not None:
        synth_labels = teacher_labels.index_select(0, init_indices).clone()  # not a Parameter

    bank_params, synth_template_labels_flat = _build_synthetic_banks(
        init_hidden=init_hidden,
        init_labels=synth_labels,
        init_std=args.init_std,
    )
    synth_template_labels = (
        synth_labels
        if synth_labels is not None
        else synth_template_labels_flat.view(init_hidden.shape[0], init_hidden.shape[1])
    )
    optimizer = torch.optim.Adam(list(bank_params.parameters()), lr=args.lr)
    history = []
    final_losses = None

    block_bundle = None
    block_constraint_layer = None
    teacher_anchor_hidden = init_hidden
    teacher_anchor_attention_mask = torch.ones(init_hidden.shape[:2], dtype=torch.long, device=device)
    teacher_anchor_position_ids = init_position_ids
    synth_attention_mask = torch.ones(init_hidden.shape[:2], dtype=torch.long, device=device)
    synth_position_ids = init_position_ids.clone()

    if args.lambda_block > 0:
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required when --lambda_block > 0.")
        from observations.common import load_model_bundle

        block_bundle = load_model_bundle(
            args.model_name_or_path,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
        block_constraint_layer = int(teacher_meta["teacher_layer"]) + 1
        block_layer = get_decoder_layer(block_bundle, block_constraint_layer)
        block_device = next(block_layer.parameters()).device
        teacher_anchor_hidden = teacher_anchor_hidden.to(block_device)
        teacher_anchor_attention_mask = teacher_anchor_attention_mask.to(block_device)
        teacher_anchor_position_ids = teacher_anchor_position_ids.to(block_device)
        synth_attention_mask = synth_attention_mask.to(block_device)
        synth_position_ids = synth_position_ids.to(block_device)
        if device != str(block_device):
            teacher_cache = teacher_cache.to(block_device)
            if teacher_labels is not None:
                teacher_labels = teacher_labels.to(block_device)
            if synth_labels is not None:
                synth_labels = synth_labels.to(block_device)
                synth_template_labels = synth_labels
            teacher_position_ids = teacher_position_ids.to(block_device)
            for _, bank in bank_params.items():
                bank.data = bank.data.to(block_device)
            synth_template_labels_flat = synth_template_labels_flat.to(block_device)
            device = str(block_device)

    for step in trange(args.train_steps, desc="Distilling compact hidden", leave=False):
        synthetic_hidden = _assemble_synthetic_hidden(
            bank_params=bank_params,
            template_labels=synth_template_labels,
            synthetic_size=args.synthetic_size,
            compressed_length=init_hidden.shape[1],
            hidden_size=init_hidden.shape[2],
        )
        # Sample a teacher batch
        batch_indices = torch.randint(0, teacher_cache.shape[0], (args.teacher_batch_size,), device=device)
        teacher_batch = teacher_cache.index_select(0, batch_indices)
        teacher_batch_labels = teacher_labels.index_select(0, batch_indices) if teacher_labels is not None else None

        # Per-modality grouped losses (primary). ``div`` stays pooled.
        losses = _compute_losses(
            teacher_batch, synthetic_hidden,
            teacher_batch_labels, synth_labels,
            mmd_subsample=args.mmd_subsample,
        )

        # Warmup: ramp ``div`` in linearly so it doesn't dominate at step 0 when
        # the other losses are tiny (and would be destabilized by a large cosine
        # regularizer). After ``div_warmup_steps`` the full weight is applied.
        div_scale = 1.0
        if args.div_warmup_steps > 0:
            div_scale = min(1.0, (step + 1) / float(args.div_warmup_steps))

        total_loss = (
            args.lambda_mmd * losses["mmd"]
            + args.lambda_cov * losses["cov"]
            + args.lambda_mean * losses["mean"]
            + args.lambda_var * losses["var"]
            + div_scale * args.lambda_div * losses["div"]
        )
        if args.lambda_block > 0:
            block_rel_l2 = _compute_next_block_rel_l2(
                bundle=block_bundle,
                layer_idx=block_constraint_layer,
                teacher_hidden=teacher_anchor_hidden,
                teacher_attention_mask=teacher_anchor_attention_mask,
                teacher_position_ids=teacher_anchor_position_ids,
                synthetic_hidden=synthetic_hidden,
                synthetic_attention_mask=synth_attention_mask,
                synthetic_position_ids=synth_position_ids,
            )
            total_loss = total_loss + args.lambda_block * block_rel_l2
            losses["block_rel_l2"] = block_rel_l2

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_losses = {
            "total": float(total_loss.detach().cpu().item()),
            "mmd": float(losses["mmd"].detach().cpu().item()),
            "cov": float(losses["cov"].detach().cpu().item()),
            "div": float(losses["div"].detach().cpu().item()),
            "mean": float(losses["mean"].detach().cpu().item()),
            "var": float(losses["var"].detach().cpu().item()),
            "div_scale": div_scale,
        }
        if "block_rel_l2" in losses:
            final_losses["block_rel_l2"] = float(losses["block_rel_l2"].detach().cpu().item())
        for k, v in losses.items():
            if "/" in k:
                final_losses[k] = float(v.detach().cpu().item())
        if step % args.log_interval == 0 or step == args.train_steps - 1:
            history.append({"step": step, **final_losses})

    synthetic_hidden = _assemble_synthetic_hidden(
        bank_params=bank_params,
        template_labels=synth_template_labels,
        synthetic_size=args.synthetic_size,
        compressed_length=init_hidden.shape[1],
        hidden_size=init_hidden.shape[2],
    )
    synthetic_hidden_cpu = synthetic_hidden.detach().cpu().float()
    attention_mask = torch.ones(
        synthetic_hidden_cpu.shape[0],
        synthetic_hidden_cpu.shape[1],
        dtype=torch.long,
    )
    payload = {
        "synthetic_hidden": synthetic_hidden_cpu,
        "attention_mask": attention_mask,
        "position_ids": synth_position_ids.detach().cpu().long(),
        "metadata": {
            "method": "multimodal_representation_level_calibration_distillation",
            "teacher_cache_path": args.teacher_cache_path,
            "teacher_metadata": teacher_meta,
            "synthetic_size": args.synthetic_size,
            "compressed_length": int(synthetic_hidden_cpu.shape[1]),
            "hidden_size": int(synthetic_hidden_cpu.shape[2]),
            "dtype": "float32",
            "position_ids_strategy": "frozen_from_init_teacher_samples",
            "train_steps": args.train_steps,
            "teacher_batch_size": args.teacher_batch_size,
            "lr": args.lr,
            "loss_weights": {
                "mmd": args.lambda_mmd,
                "cov": args.lambda_cov,
                "div": args.lambda_div,
                "mean": args.lambda_mean,
                "var": args.lambda_var,
                "block_rel_l2": args.lambda_block,
            },
            "modality_grouped_losses": ["mmd", "cov", "mean", "var"],
            "div_warmup_steps": args.div_warmup_steps,
            "mmd_subsample": args.mmd_subsample,
            "synthetic_bank_mode": "independent_per_modality",
            "block_constraint_layer": block_constraint_layer,
            "final_losses": final_losses,
            "history": history,
            "created_at": utc_now_iso(),
        },
    }
    if synth_labels is not None:
        payload["modality_labels"] = synth_labels.cpu()

    torch.save(payload, args.output_path)
    print(f"[representation_distill] Saved synthetic calibration hidden to {args.output_path}")


if __name__ == "__main__":
    main()
