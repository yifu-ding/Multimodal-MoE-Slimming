import argparse
import math
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

from src.calibration.representation_distill.common import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    ensure_dir,
    seed_everything,
    utc_now_iso,
)


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


def _compute_stat_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Legacy moment-matching losses (kept as auxiliaries)."""
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    teacher_mean = teacher_flat.mean(dim=0)
    synth_mean = synth_flat.mean(dim=0)
    teacher_var = teacher_flat.var(dim=0, unbiased=False)
    synth_var = synth_flat.var(dim=0, unbiased=False)

    return {
        "mean": (teacher_mean - synth_mean).pow(2).mean(),
        "var": (teacher_var - synth_var).pow(2).mean(),
    }


def _compute_distribution_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    teacher_labels: torch.Tensor | None,
    synth_labels: torch.Tensor | None,
    mmd_subsample: int,
) -> dict[str, torch.Tensor]:
    """Distribution-aware losses: MMD (modality-conditioned), covariance, diversity."""
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    losses: dict[str, torch.Tensor] = {}

    # --- Modality-conditioned MMD ---
    if teacher_labels is not None and synth_labels is not None:
        teacher_labels_flat = teacher_labels.reshape(-1) if teacher_labels is not None else None
        synth_labels_flat = synth_labels.reshape(-1) if synth_labels is not None else None

        mmd_total = torch.tensor(0.0, device=teacher_flat.device)
        n_modalities = 0
        for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
            t_mask = teacher_labels_flat == mod_id
            s_mask = synth_labels_flat == mod_id
            if t_mask.any() and s_mask.any():
                t_sub = _subsample_flat(teacher_flat[t_mask], mmd_subsample)
                s_sub = _subsample_flat(synth_flat[s_mask], mmd_subsample)
                mmd_total = mmd_total + _mmd_rbf(t_sub, s_sub)
                n_modalities += 1
        if n_modalities > 0:
            losses["mmd"] = mmd_total / n_modalities
        else:
            losses["mmd"] = _mmd_rbf(
                _subsample_flat(teacher_flat, mmd_subsample),
                _subsample_flat(synth_flat, mmd_subsample),
            )
    else:
        losses["mmd"] = _mmd_rbf(
            _subsample_flat(teacher_flat, mmd_subsample),
            _subsample_flat(synth_flat, mmd_subsample),
        )

    # --- Covariance matching ---
    losses["cov"] = _cov_loss(
        _subsample_flat(teacher_flat, mmd_subsample),
        synth_flat,
    )

    # --- Diversity regularization ---
    losses["div"] = _diversity_loss(_subsample_flat(synth_flat, mmd_subsample))

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
    parser.add_argument("--init_std", type=float, default=1e-3)
    # Distribution-aware loss weights
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_cov", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    # Legacy moment-matching (auxiliary, reduced weight)
    parser.add_argument("--lambda_mean", type=float, default=0.5)
    parser.add_argument("--lambda_var", type=float, default=0.5)
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

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_cache = teacher_cache.to(device)
    if teacher_labels is not None:
        teacher_labels = teacher_labels.to(device)

    # Initialize from random teacher samples
    init_indices = torch.randint(0, teacher_cache.shape[0], (args.synthetic_size,), device=device)
    synthetic_hidden = torch.nn.Parameter(teacher_cache.index_select(0, init_indices).clone())
    if args.init_std > 0:
        synthetic_hidden.data.add_(torch.randn_like(synthetic_hidden) * args.init_std)

    # Modality labels: frozen, copied from init samples
    synth_labels = None
    if teacher_labels is not None:
        synth_labels = teacher_labels.index_select(0, init_indices).clone()  # not a Parameter

    optimizer = torch.optim.Adam([synthetic_hidden], lr=args.lr)
    history = []
    final_losses = None

    for step in trange(args.train_steps, desc="Distilling synthetic hidden", leave=False):
        # Sample a teacher batch
        batch_indices = torch.randint(0, teacher_cache.shape[0], (args.teacher_batch_size,), device=device)
        teacher_batch = teacher_cache.index_select(0, batch_indices)
        teacher_batch_labels = teacher_labels.index_select(0, batch_indices) if teacher_labels is not None else None

        # Moment-matching losses (auxiliary)
        stat_losses = _compute_stat_losses(teacher_batch, synthetic_hidden)

        # Distribution-aware losses (primary)
        dist_losses = _compute_distribution_losses(
            teacher_batch, synthetic_hidden,
            teacher_batch_labels, synth_labels,
            mmd_subsample=args.mmd_subsample,
        )

        total_loss = (
            args.lambda_mmd * dist_losses["mmd"]
            + args.lambda_cov * dist_losses["cov"]
            + args.lambda_div * dist_losses["div"]
            + args.lambda_mean * stat_losses["mean"]
            + args.lambda_var * stat_losses["var"]
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_losses = {
            "total": float(total_loss.detach().cpu().item()),
            "mmd": float(dist_losses["mmd"].detach().cpu().item()),
            "cov": float(dist_losses["cov"].detach().cpu().item()),
            "div": float(dist_losses["div"].detach().cpu().item()),
            "mean": float(stat_losses["mean"].detach().cpu().item()),
            "var": float(stat_losses["var"].detach().cpu().item()),
        }
        if step % args.log_interval == 0 or step == args.train_steps - 1:
            history.append({"step": step, **final_losses})

    synthetic_hidden_cpu = synthetic_hidden.detach().cpu().float()
    attention_mask = torch.ones(
        synthetic_hidden_cpu.shape[0],
        synthetic_hidden_cpu.shape[1],
        dtype=torch.long,
    )
    payload = {
        "synthetic_hidden": synthetic_hidden_cpu,
        "attention_mask": attention_mask,
        "metadata": {
            "method": "multimodal_representation_level_calibration_distillation",
            "teacher_cache_path": args.teacher_cache_path,
            "teacher_metadata": teacher_meta,
            "synthetic_size": args.synthetic_size,
            "compressed_length": int(synthetic_hidden_cpu.shape[1]),
            "hidden_size": int(synthetic_hidden_cpu.shape[2]),
            "dtype": "float32",
            "position_ids_strategy": "sequential_from_attention_mask",
            "train_steps": args.train_steps,
            "teacher_batch_size": args.teacher_batch_size,
            "lr": args.lr,
            "loss_weights": {
                "mmd": args.lambda_mmd,
                "cov": args.lambda_cov,
                "div": args.lambda_div,
                "mean": args.lambda_mean,
                "var": args.lambda_var,
            },
            "mmd_subsample": args.mmd_subsample,
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
