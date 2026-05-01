#!/usr/bin/env python3
"""A3: split teacher hidden into core/outlier subsets and plot density contours."""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_maes")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from observations.a3.plot_hidden_tsne import (  # noqa: E402
    COLOR_TEXT,
    COLOR_VISUAL,
    _hidden_key,
    _load_payload,
    _sanitize_stem,
    _valid_mask,
)
from observations.common import ensure_dir, print_saved_artifact_message  # noqa: E402


DEFAULT_TEACHER_PATH = (
    "/home/dyf/code/distill/MAES/storage/data_distill_kimi/"
    "gqa-num_342-token_2048-sample_at1.0-0429161205/teacher_hidden.pt"
)
DEFAULT_OUTPUT_DIR = "/home/dyf/code/distill/MAES/observations/a3/split-hidden"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a teacher hidden cache into core/outlier subsets and visualize "
            "them with density contours instead of highlighted scatter points."
        )
    )
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--min_samples", type=int, default=100)
    parser.add_argument("--sample_core_ratio", type=float, default=0.35)
    parser.add_argument("--sample_outlier_ratio", type=float, default=0.30)
    parser.add_argument("--modality_core_ratio", type=float, default=0.35)
    parser.add_argument("--modality_outlier_ratio", type=float, default=0.30)
    parser.add_argument("--min_modality_tokens", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _derive_stem(teacher_path: str, output_name: str) -> str:
    if output_name.strip():
        return _sanitize_stem(output_name)
    teacher_tag = _sanitize_stem(os.path.basename(os.path.dirname(teacher_path)))
    return f"a3_density_core_outlier_{teacher_tag}"


def _sample_mean_features(payload: Dict[str, torch.Tensor]) -> np.ndarray:
    hidden = payload[_hidden_key(payload)].to(torch.float32)
    valid = _valid_mask(payload)
    rows = [hidden[idx, valid[idx]].mean(dim=0) for idx in range(hidden.shape[0])]
    return torch.stack(rows, dim=0).numpy()


def _modality_centroids(
    payload: Dict[str, torch.Tensor],
    min_modality_tokens: int,
) -> Tuple[np.ndarray, np.ndarray]:
    hidden = payload[_hidden_key(payload)].to(torch.float32)
    modality = payload["modality_labels"]
    valid = _valid_mask(payload)
    text_rows = []
    visual_rows = []
    for idx in range(hidden.shape[0]):
        sample_mask = valid[idx]
        text_mask = sample_mask & (modality[idx] == 0)
        visual_mask = sample_mask & (modality[idx] == 1)
        if int(text_mask.sum().item()) < min_modality_tokens:
            raise ValueError(f"Sample {idx} has too few text tokens for modality centroid.")
        if int(visual_mask.sum().item()) < min_modality_tokens:
            raise ValueError(f"Sample {idx} has too few visual tokens for modality centroid.")
        text_rows.append(hidden[idx, text_mask].mean(dim=0))
        visual_rows.append(hidden[idx, visual_mask].mean(dim=0))
    return torch.stack(text_rows, dim=0).numpy(), torch.stack(visual_rows, dim=0).numpy()


def _standardize_fit(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = np.clip(features.std(axis=0, keepdims=True), 1e-6, None)
    return (features - mean) / std


def _rank_mask(values: np.ndarray, count: int, smallest: bool) -> np.ndarray:
    order = np.argsort(values)
    selected = order[:count] if smallest else order[-count:]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def _counts_from_ratio(n: int, ratio: float, min_samples: int) -> int:
    return min(n, max(min_samples, int(round(n * ratio))))


def _adaptive_core_mask(
    sample_radius: np.ndarray,
    text_radius: np.ndarray,
    visual_radius: np.ndarray,
    core_count: int,
    min_samples: int,
) -> np.ndarray:
    for scale in (1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4):
        k = min(sample_radius.shape[0], max(min_samples, int(round(core_count * scale))))
        mask = (
            _rank_mask(sample_radius, k, smallest=True)
            & _rank_mask(text_radius, k, smallest=True)
            & _rank_mask(visual_radius, k, smallest=True)
        )
        if int(mask.sum()) >= min_samples:
            return mask
    votes = (
        _rank_mask(sample_radius, core_count, smallest=True).astype(np.int32)
        + _rank_mask(text_radius, core_count, smallest=True).astype(np.int32)
        + _rank_mask(visual_radius, core_count, smallest=True).astype(np.int32)
    )
    mask = votes >= 2
    if int(mask.sum()) >= min_samples:
        return mask
    return votes >= 1


def _adaptive_outlier_mask(
    sample_radius: np.ndarray,
    text_radius: np.ndarray,
    visual_radius: np.ndarray,
    outlier_count: int,
    min_samples: int,
) -> np.ndarray:
    for scale in (1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4):
        k = min(sample_radius.shape[0], max(min_samples, int(round(outlier_count * scale))))
        votes = (
            _rank_mask(sample_radius, k, smallest=False).astype(np.int32)
            + _rank_mask(text_radius, k, smallest=False).astype(np.int32)
            + _rank_mask(visual_radius, k, smallest=False).astype(np.int32)
        )
        mask = votes >= 2
        if int(mask.sum()) >= min_samples:
            return mask
    votes = (
        _rank_mask(sample_radius, outlier_count, smallest=False).astype(np.int32)
        + _rank_mask(text_radius, outlier_count, smallest=False).astype(np.int32)
        + _rank_mask(visual_radius, outlier_count, smallest=False).astype(np.int32)
    )
    return votes >= 1


def _subset_payload(
    payload: Dict[str, torch.Tensor],
    indices: np.ndarray,
    subset_name: str,
    base_stem: str,
) -> Dict[str, object]:
    idx_tensor = torch.as_tensor(indices, dtype=torch.long)
    n_samples = payload[_hidden_key(payload)].shape[0]
    out: Dict[str, object] = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.shape[:1] == (n_samples,):
            out[key] = value.index_select(0, idx_tensor)
        elif isinstance(value, list) and len(value) == n_samples:
            out[key] = [value[i] for i in indices.tolist()]
        else:
            out[key] = value

    metadata = dict(payload.get("metadata", {}))
    metadata["derived_from_teacher_path"] = payload.get("__source_path__")
    metadata["subset_name"] = subset_name
    metadata["subset_num_samples"] = int(indices.shape[0])
    metadata["subset_indices"] = indices.tolist()
    metadata["subset_stem"] = base_stem
    out["metadata"] = metadata
    out["num_samples"] = int(indices.shape[0])
    return out


def _draw_density_contours(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    label: str,
    alpha_fill: float,
    linewidth: float,
) -> None:
    if points.shape[0] < 16:
        return
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    pad_x = max((xmax - xmin) * 0.10, 1e-3)
    pad_y = max((ymax - ymin) * 0.10, 1e-3)
    xx, yy = np.mgrid[
        (xmin - pad_x):(xmax + pad_x):160j,
        (ymin - pad_y):(ymax + pad_y):160j,
    ]
    kde = gaussian_kde(points.T)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.quantile(zz, [0.70, 0.85, 0.94])
    ax.contourf(
        xx,
        yy,
        zz,
        levels=np.concatenate([[zz.min()], levels]),
        colors=[color, color, color],
        alpha=alpha_fill,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        colors=color,
        linewidths=linewidth,
        alpha=min(alpha_fill + 0.35, 0.85),
    )


def _plot_density_panels(
    out_path: str,
    text_tsne: np.ndarray,
    visual_tsne: np.ndarray,
    core_mask: np.ndarray,
    outlier_mask: np.ndarray,
    title: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4), constrained_layout=True)
    panel_specs = [
        (axes[0], core_mask, "Core Hidden Density"),
        (axes[1], outlier_mask, "Outlier Hidden Density"),
    ]
    for ax, selected_mask, subtitle in panel_specs:
        ax.scatter(
            text_tsne[:, 0],
            text_tsne[:, 1],
            s=10,
            c=COLOR_TEXT,
            alpha=0.08,
            linewidths=0.0,
            label="Teacher text (all)",
            zorder=1,
        )
        ax.scatter(
            visual_tsne[:, 0],
            visual_tsne[:, 1],
            s=10,
            c=COLOR_VISUAL,
            alpha=0.08,
            linewidths=0.0,
            label="Teacher visual (all)",
            zorder=1,
        )
        _draw_density_contours(
            ax,
            text_tsne[selected_mask],
            color=COLOR_TEXT,
            label=f"{subtitle} text density",
            alpha_fill=0.22,
            linewidth=1.3,
        )
        _draw_density_contours(
            ax,
            visual_tsne[selected_mask],
            color=COLOR_VISUAL,
            label=f"{subtitle} visual density",
            alpha_fill=0.22,
            linewidth=1.3,
        )
        ax.set_title(subtitle, fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=COLOR_TEXT, markeredgewidth=0, alpha=0.35, label="Teacher text (all)"),
            Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=COLOR_VISUAL, markeredgewidth=0, alpha=0.35, label="Teacher visual (all)"),
            Line2D([0], [0], color=COLOR_TEXT, linewidth=1.8, label=f"{subtitle} text density"),
            Line2D([0], [0], color=COLOR_VISUAL, linewidth=1.8, label=f"{subtitle} visual density"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(title, fontsize=15)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    teacher_payload = _load_payload(args.teacher_path)
    teacher_payload["__source_path__"] = args.teacher_path
    n_samples = teacher_payload[_hidden_key(teacher_payload)].shape[0]

    sample_features = _sample_mean_features(teacher_payload)
    sample_pca = PCA(
        n_components=min(args.pca_dim, sample_features.shape[0], sample_features.shape[1]),
        random_state=args.random_state,
    ).fit_transform(_standardize_fit(sample_features))
    sample_center = sample_pca.mean(axis=0, keepdims=True)
    sample_radius = np.linalg.norm(sample_pca - sample_center, axis=1)

    text_features, visual_features = _modality_centroids(
        teacher_payload, min_modality_tokens=args.min_modality_tokens
    )
    text_pca = PCA(
        n_components=min(args.pca_dim, text_features.shape[0], text_features.shape[1]),
        random_state=args.random_state,
    ).fit_transform(_standardize_fit(text_features))
    visual_pca = PCA(
        n_components=min(args.pca_dim, visual_features.shape[0], visual_features.shape[1]),
        random_state=args.random_state,
    ).fit_transform(_standardize_fit(visual_features))
    text_center = text_pca.mean(axis=0, keepdims=True)
    visual_center = visual_pca.mean(axis=0, keepdims=True)
    text_radius = np.linalg.norm(text_pca - text_center, axis=1)
    visual_radius = np.linalg.norm(visual_pca - visual_center, axis=1)

    core_count = max(
        _counts_from_ratio(n_samples, args.sample_core_ratio, args.min_samples),
        _counts_from_ratio(n_samples, args.modality_core_ratio, args.min_samples),
    )
    outlier_count = max(
        _counts_from_ratio(n_samples, args.sample_outlier_ratio, args.min_samples),
        _counts_from_ratio(n_samples, args.modality_outlier_ratio, args.min_samples),
    )

    core_mask = _adaptive_core_mask(
        sample_radius=sample_radius,
        text_radius=text_radius,
        visual_radius=visual_radius,
        core_count=core_count,
        min_samples=args.min_samples,
    )
    outlier_mask = _adaptive_outlier_mask(
        sample_radius=sample_radius,
        text_radius=text_radius,
        visual_radius=visual_radius,
        outlier_count=outlier_count,
        min_samples=args.min_samples,
    )

    core_indices = np.flatnonzero(core_mask)
    outlier_indices = np.flatnonzero(outlier_mask)
    overlap_indices = np.flatnonzero(core_mask & outlier_mask)

    stem = _derive_stem(args.teacher_path, args.output_name)
    core_pt_path = os.path.join(args.output_dir, f"{stem}_core-hidden.pt")
    outlier_pt_path = os.path.join(args.output_dir, f"{stem}_outlier-hidden.pt")
    plot_path = os.path.join(args.output_dir, f"{stem}_density_tsne.png")
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.json")

    torch.save(_subset_payload(teacher_payload, core_indices, "core-hidden", stem), core_pt_path)
    torch.save(_subset_payload(teacher_payload, outlier_indices, "outlier-hidden", stem), outlier_pt_path)

    text_perplexity = min(args.perplexity, max(5.0, (text_pca.shape[0] - 1) / 3.0))
    visual_perplexity = min(args.perplexity, max(5.0, (visual_pca.shape[0] - 1) / 3.0))
    text_tsne = TSNE(
        n_components=2,
        perplexity=text_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    ).fit_transform(text_pca)
    visual_tsne = TSNE(
        n_components=2,
        perplexity=visual_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    ).fit_transform(visual_pca)

    _plot_density_panels(
        out_path=plot_path,
        text_tsne=text_tsne,
        visual_tsne=visual_tsne,
        core_mask=core_mask,
        outlier_mask=outlier_mask,
        title=(
            "A3 Teacher Split by Calibration Geometry\n"
            "Density contours emphasize the central manifold captured by core vs the diffuse tails kept by outlier"
        ),
        dpi=args.dpi,
    )

    summary = {
        "teacher_path": args.teacher_path,
        "num_teacher_samples": int(n_samples),
        "core_num_samples": int(core_mask.sum()),
        "outlier_num_samples": int(outlier_mask.sum()),
        "overlap_num_samples": int(overlap_indices.shape[0]),
        "sample_radius_p50": float(np.percentile(sample_radius, 50)),
        "sample_radius_p90": float(np.percentile(sample_radius, 90)),
        "text_radius_p50": float(np.percentile(text_radius, 50)),
        "text_radius_p90": float(np.percentile(text_radius, 90)),
        "visual_radius_p50": float(np.percentile(visual_radius, 50)),
        "visual_radius_p90": float(np.percentile(visual_radius, 90)),
        "core_indices": core_indices.tolist(),
        "outlier_indices": outlier_indices.tolist(),
        "overlap_indices": overlap_indices.tolist(),
        "artifacts": {
            "core_hidden_pt": core_pt_path,
            "outlier_hidden_pt": outlier_pt_path,
            "density_tsne_plot": plot_path,
            "summary_json": summary_path,
        },
    }
    _save_json(summary_path, summary)

    print_saved_artifact_message(core_pt_path, "A3 density core hidden subset")
    print_saved_artifact_message(outlier_pt_path, "A3 density outlier hidden subset")
    print_saved_artifact_message(plot_path, "A3 density t-SNE plot")
    print_saved_artifact_message(summary_path, "A3 density split summary JSON")


if __name__ == "__main__":
    main()
