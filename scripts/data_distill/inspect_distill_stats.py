"""Compare the statistics of a teacher hidden cache and its distilled counterpart.

Usage
-----
python scripts/data_distill/inspect_distill_stats.py \
    --teacher storage/data_distill/teacher_cache-attn_weighted/teacher_hidden_cache.pt \
    --distilled storage/data_distill/teacher_cache-attn_weighted/distilled/distilled_hidden.pt

The script inspects whether the distilled synthetic hidden states still resemble the
teacher cache in terms of first/second order moments, per-channel behaviour, sample
norms and (optionally) modality-conditional distributions. It is deliberately
read-only and does not depend on any model weights.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_payload(path: str) -> Dict:
    pth = Path(path)
    if not pth.exists():
        raise FileNotFoundError(f"Cache file not found: {pth}")
    try:
        payload = torch.load(pth, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(pth, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type in {pth}: {type(payload)}")
    return payload


def _pick_hidden(payload: Dict, label: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (hidden[N, L, D], attn_mask[N, L] or None, modality_labels[N, L] or None)."""
    if "teacher_cache" in payload:
        hidden = payload["teacher_cache"]
    elif "synthetic_hidden" in payload:
        hidden = payload["synthetic_hidden"]
    elif "hidden" in payload:
        hidden = payload["hidden"]
    else:
        raise KeyError(
            f"[{label}] Payload does not contain 'teacher_cache', 'synthetic_hidden' or 'hidden'. "
            f"Available keys: {list(payload.keys())}"
        )

    if hidden.dim() != 3:
        raise ValueError(
            f"[{label}] Expected hidden of shape [N, L, D], got {tuple(hidden.shape)}"
        )

    attn_mask = payload.get("attention_mask")
    if attn_mask is not None:
        if attn_mask.dim() != 2 or attn_mask.shape[:2] != hidden.shape[:2]:
            attn_mask = None

    labels = payload.get("modality_labels")
    if labels is not None and (labels.dim() != 2 or labels.shape[:2] != hidden.shape[:2]):
        labels = None

    return hidden, attn_mask, labels


def _flatten_valid(
    hidden: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return [T, D] tensor of valid tokens (mask>0) plus aligned modality labels."""
    flat = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
    flat_labels = labels.reshape(-1) if labels is not None else None
    if attn_mask is not None:
        keep = attn_mask.reshape(-1).to(torch.bool)
        flat = flat[keep]
        if flat_labels is not None:
            flat_labels = flat_labels[keep]
    return flat, flat_labels


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------


def _basic_stats(flat: torch.Tensor) -> Dict[str, float]:
    return {
        "num_tokens": int(flat.shape[0]),
        "hidden_size": int(flat.shape[1]),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "l2_per_token_mean": float(flat.norm(dim=-1).mean().item()),
        "l2_per_token_std": float(flat.norm(dim=-1).std(unbiased=False).item()),
    }


def _channel_stats(flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_c = flat.mean(dim=0)
    std_c = flat.std(dim=0, unbiased=False)
    return mean_c, std_c


def _covariance(flat: torch.Tensor, subsample: int = 4096, seed: int = 0) -> torch.Tensor:
    n = flat.shape[0]
    if n > subsample:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:subsample]
        flat = flat[idx]
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    return cov


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((a * b).sum().item() / denom.item())


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item())
    if denom == 0.0:
        return float("inf")
    return float((a - b).norm().item() / denom)


def _mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    max_samples: int = 2048,
    seed: int = 0,
) -> float:
    """Biased-ish MMD^2 under an RBF kernel with median-heuristic bandwidth."""
    g = torch.Generator().manual_seed(seed)
    if x.shape[0] > max_samples:
        x = x[torch.randperm(x.shape[0], generator=g)[:max_samples]]
    if y.shape[0] > max_samples:
        y = y[torch.randperm(y.shape[0], generator=g)[:max_samples]]
    xy = torch.cat([x, y], dim=0)
    with torch.no_grad():
        dists = torch.cdist(xy, xy, p=2.0)
        tri = dists[torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)]
        sigma = float(tri.median().item())
        sigma = max(sigma, 1e-6)
        gamma = 1.0 / (2.0 * sigma * sigma)

        def _k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.exp(-gamma * torch.cdist(a, b, p=2.0) ** 2)

        kxx = _k(x, x).mean()
        kyy = _k(y, y).mean()
        kxy = _k(x, y).mean()
    return float((kxx + kyy - 2.0 * kxy).item())


def _pca_subspace_similarity(
    cov_a: torch.Tensor,
    cov_b: torch.Tensor,
    k: int = 32,
) -> Dict[str, float]:
    k = min(k, cov_a.shape[0])
    eigvals_a, eigvecs_a = torch.linalg.eigh(cov_a)
    eigvals_b, eigvecs_b = torch.linalg.eigh(cov_b)
    top_a = eigvecs_a[:, -k:]
    top_b = eigvecs_b[:, -k:]
    m = top_a.T @ top_b
    sigvals = torch.linalg.svdvals(m)
    grassmann = float(sigvals.mean().item())
    expl_a = float((eigvals_a[-k:].clamp(min=0.0).sum() / eigvals_a.clamp(min=0.0).sum()).item())
    expl_b = float((eigvals_b[-k:].clamp(min=0.0).sum() / eigvals_b.clamp(min=0.0).sum()).item())
    return {
        "top_k": k,
        "grassmann_cosine_mean": grassmann,
        "teacher_topk_var_ratio": expl_a,
        "distilled_topk_var_ratio": expl_b,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _kv(label: str, value, fmt: str = ".6f") -> None:
    if isinstance(value, (int,)):
        print(f"  {label:<36s} {value}")
    elif isinstance(value, float):
        print(f"  {label:<36s} {value:{fmt}}")
    else:
        print(f"  {label:<36s} {value}")


def _describe_tensor(name: str, flat: torch.Tensor) -> Dict[str, float]:
    stats = _basic_stats(flat)
    print(f"[{name}] tokens={stats['num_tokens']}  hidden={stats['hidden_size']}")
    _kv("mean", stats["mean"])
    _kv("std", stats["std"])
    _kv("abs_mean", stats["abs_mean"])
    _kv("min", stats["min"])
    _kv("max", stats["max"])
    _kv("per-token L2 mean", stats["l2_per_token_mean"])
    _kv("per-token L2 std", stats["l2_per_token_std"])
    return stats


def _summarise_metadata(payload: Dict, label: str) -> None:
    meta = payload.get("metadata", {})
    if not isinstance(meta, dict):
        return
    print(f"[{label}] metadata highlights:")
    for key in (
        "method",
        "model_name_or_path",
        "model_family",
        "teacher_layer",
        "compressed_length",
        "compression_mode",
        "hidden_size",
        "teacher_cache_dtype",
        "teacher_datasets",
        "total_samples",
        "synthetic_size",
        "train_steps",
        "lr",
        "loss_weights",
        "mmd_subsample",
        "final_losses",
        "created_at",
    ):
        if key in meta:
            _kv(key, meta[key], fmt=".4g")
    history = meta.get("history")
    if isinstance(history, list) and history:
        first = history[0]
        last = history[-1]
        print("  loss history (first -> last):")
        for k in ("total", "mmd", "cov", "div", "mean", "var"):
            if k in first and k in last:
                fv = float(first[k])
                lv = float(last[k])
                delta = lv - fv
                print(f"    {k:<6s} {fv: .4g}  ->  {lv: .4g}   (delta={delta:+.4g})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        default="storage/data_distill/teacher_cache-attn_weighted/teacher_hidden_cache.pt",
        help="Path to teacher_hidden_cache.pt",
    )
    parser.add_argument(
        "--distilled",
        default="storage/data_distill/teacher_cache-attn_weighted/distilled/distilled_hidden.pt",
        help="Path to distilled_hidden.pt",
    )
    parser.add_argument(
        "--topk_channels",
        type=int,
        default=10,
        help="How many channels to report in the largest-deviation tables.",
    )
    parser.add_argument(
        "--pca_k",
        type=int,
        default=32,
        help="Top-k principal components to compare between teacher and distilled caches.",
    )
    parser.add_argument(
        "--mmd_max_samples",
        type=int,
        default=2048,
        help="Max tokens (per side) used when estimating MMD with an RBF kernel.",
    )
    parser.add_argument(
        "--cov_subsample",
        type=int,
        default=8192,
        help="Number of tokens sampled to estimate the covariance matrix per side.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for subsampling.",
    )
    args = parser.parse_args()

    _section("Loading caches")
    teacher_payload = _load_payload(args.teacher)
    distilled_payload = _load_payload(args.distilled)
    print(f"  teacher  : {args.teacher}")
    print(f"  distilled: {args.distilled}")

    teacher_hidden, teacher_mask, teacher_labels = _pick_hidden(teacher_payload, "teacher")
    distilled_hidden, distilled_mask, distilled_labels = _pick_hidden(distilled_payload, "distilled")

    _kv("teacher shape", tuple(teacher_hidden.shape))
    _kv("teacher dtype", str(teacher_hidden.dtype))
    _kv("distilled shape", tuple(distilled_hidden.shape))
    _kv("distilled dtype", str(distilled_hidden.dtype))

    if teacher_hidden.shape[-1] != distilled_hidden.shape[-1]:
        raise ValueError(
            f"Hidden size mismatch: teacher D={teacher_hidden.shape[-1]} vs distilled D={distilled_hidden.shape[-1]}"
        )

    _section("Metadata")
    _summarise_metadata(teacher_payload, "teacher")
    print()
    _summarise_metadata(distilled_payload, "distilled")

    teacher_flat, teacher_flat_labels = _flatten_valid(teacher_hidden, teacher_mask, teacher_labels)
    distilled_flat, distilled_flat_labels = _flatten_valid(distilled_hidden, distilled_mask, distilled_labels)

    _section("Global marginal statistics (valid tokens)")
    teacher_stats = _describe_tensor("teacher", teacher_flat)
    print()
    distilled_stats = _describe_tensor("distilled", distilled_flat)

    _section("First / second moment deltas")
    teacher_mean_c, teacher_std_c = _channel_stats(teacher_flat)
    distilled_mean_c, distilled_std_c = _channel_stats(distilled_flat)

    mean_l2_rel = _relative_l2(teacher_mean_c, distilled_mean_c)
    std_l2_rel = _relative_l2(teacher_std_c, distilled_std_c)
    mean_cos = _cosine(teacher_mean_c, distilled_mean_c)
    std_cos = _cosine(teacher_std_c, distilled_std_c)
    _kv("||mu_t - mu_s||2 / ||mu_t||2", mean_l2_rel)
    _kv("cos(mu_t, mu_s)", mean_cos)
    _kv("||sigma_t - sigma_s||2 / ||sigma_t||2", std_l2_rel)
    _kv("cos(sigma_t, sigma_s)", std_cos)
    _kv("max |delta mean per channel|", float((teacher_mean_c - distilled_mean_c).abs().max().item()))
    _kv("max |delta std per channel|", float((teacher_std_c - distilled_std_c).abs().max().item()))
    _kv("delta global mean", teacher_stats["mean"] - distilled_stats["mean"])
    _kv("delta global std", teacher_stats["std"] - distilled_stats["std"])
    _kv("delta per-token L2 mean", teacher_stats["l2_per_token_mean"] - distilled_stats["l2_per_token_mean"])

    topk = max(1, args.topk_channels)
    _section(f"Top-{topk} channels with largest |delta mean| (teacher - distilled)")
    diff_mean = (teacher_mean_c - distilled_mean_c).abs()
    top_mean_idx = torch.topk(diff_mean, topk).indices.tolist()
    print(f"  {'channel':>8s}  {'teacher_mean':>12s}  {'distil_mean':>12s}  {'abs_delta':>10s}")
    for idx in top_mean_idx:
        print(
            f"  {idx:>8d}  {float(teacher_mean_c[idx]):>12.4g}  "
            f"{float(distilled_mean_c[idx]):>12.4g}  {float(diff_mean[idx]):>10.4g}"
        )

    _section(f"Top-{topk} channels with largest |delta std|")
    diff_std = (teacher_std_c - distilled_std_c).abs()
    top_std_idx = torch.topk(diff_std, topk).indices.tolist()
    print(f"  {'channel':>8s}  {'teacher_std':>12s}  {'distil_std':>12s}  {'abs_delta':>10s}")
    for idx in top_std_idx:
        print(
            f"  {idx:>8d}  {float(teacher_std_c[idx]):>12.4g}  "
            f"{float(distilled_std_c[idx]):>12.4g}  {float(diff_std[idx]):>10.4g}"
        )

    _section("Second-order structure (covariance)")
    cov_teacher = _covariance(teacher_flat, subsample=args.cov_subsample, seed=args.seed)
    cov_distilled = _covariance(distilled_flat, subsample=args.cov_subsample, seed=args.seed)
    cov_diff = cov_teacher - cov_distilled
    cov_rel_l2 = _relative_l2(cov_teacher, cov_distilled)
    cov_cos = _cosine(cov_teacher.flatten(), cov_distilled.flatten())
    _kv("||C_t - C_s||_F / ||C_t||_F", cov_rel_l2)
    _kv("cosine(C_t, C_s) (flattened)", cov_cos)
    _kv("tr(C_t)", float(torch.diagonal(cov_teacher).sum().item()))
    _kv("tr(C_s)", float(torch.diagonal(cov_distilled).sum().item()))
    _kv("max |delta cov entry|", float(cov_diff.abs().max().item()))

    pca_info = _pca_subspace_similarity(cov_teacher, cov_distilled, k=args.pca_k)
    _kv(
        f"top-{pca_info['top_k']} PCA Grassmann mean cos",
        pca_info["grassmann_cosine_mean"],
    )
    _kv(
        f"teacher top-{pca_info['top_k']} var ratio",
        pca_info["teacher_topk_var_ratio"],
    )
    _kv(
        f"distilled top-{pca_info['top_k']} var ratio",
        pca_info["distilled_topk_var_ratio"],
    )

    _section("Distribution-level distance (RBF-MMD^2 on subsamples)")
    mmd_val = _mmd_rbf(
        teacher_flat,
        distilled_flat,
        max_samples=args.mmd_max_samples,
        seed=args.seed,
    )
    _kv("MMD^2 (RBF, median-sigma)", mmd_val)

    if teacher_flat_labels is not None and distilled_flat_labels is not None:
        modalities = sorted(set(int(x) for x in teacher_flat_labels.unique().tolist()))
        _section("Per-modality moment comparison (where labels agree)")
        for m in modalities:
            t_sub = teacher_flat[teacher_flat_labels == m]
            d_sub = distilled_flat[distilled_flat_labels == m]
            if t_sub.numel() == 0 or d_sub.numel() == 0:
                continue
            t_mean_c, t_std_c = _channel_stats(t_sub)
            d_mean_c, d_std_c = _channel_stats(d_sub)
            print(f"  modality={m}  teacher_tokens={t_sub.shape[0]}  distilled_tokens={d_sub.shape[0]}")
            _kv("  mean rel L2", _relative_l2(t_mean_c, d_mean_c))
            _kv("  std rel L2", _relative_l2(t_std_c, d_std_c))
            _kv("  per-token L2 (teacher/distilled)",
                f"{float(t_sub.norm(dim=-1).mean().item()):.4f} / {float(d_sub.norm(dim=-1).mean().item()):.4f}")

    _section("Sanity checks")
    checks: List[Tuple[str, bool]] = [
        ("teacher hidden contains no NaN", not bool(torch.isnan(teacher_hidden).any().item())),
        ("distilled hidden contains no NaN", not bool(torch.isnan(distilled_hidden).any().item())),
        ("teacher hidden contains no Inf", not bool(torch.isinf(teacher_hidden).any().item())),
        ("distilled hidden contains no Inf", not bool(torch.isinf(distilled_hidden).any().item())),
        (
            "hidden sizes match",
            teacher_hidden.shape[-1] == distilled_hidden.shape[-1],
        ),
    ]
    for msg, ok in checks:
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {msg}")

    _section("Interpretation hints")
    print(
        "  * If relative-L2 on (mean, std, cov) is ~<= 0.1-0.2, the synthetic distribution\n"
        "    closely reproduces the teacher's first/second-order statistics.\n"
        "  * cos(sigma_t, sigma_s) close to 1.0 means per-channel variance profile is preserved.\n"
        "  * Grassmann mean-cosine of top-k PCA close to 1.0 means the principal subspace is\n"
        "    aligned; values below ~0.7 indicate the dominant directions drifted.\n"
        "  * MMD^2 values are scale-dependent; compare runs against the same bandwidth, or use\n"
        "    the value as a ranking between different distillation checkpoints."
    )


if __name__ == "__main__":
    main()
