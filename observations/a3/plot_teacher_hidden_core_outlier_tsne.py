#!/usr/bin/env python3
"""A3: teacher-only t-SNE with core/outlier panels."""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_maes")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FONT_FAMILY = ["Times New Roman", "Liberation Serif", "serif"]
# PANEL_TITLE_FONT_SIZE = 18
TICK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
DENSITY_GRID_SIZE = 120
DENSITY_FILLED_LEVELS = 8
DENSITY_LINE_LEVELS = 10
DENSITY_ALPHA = 0.4
DEFAULT_TEACHER_PATH = (
    "/home/dyf/code/distill/MAES/storage/data_distill_kimi/"
    "mixed-num_1024-token_2048-sample_at1.0-0421180651/teacher_hidden.pt"
)
DEFAULT_VIDEO_MMMU_PATH = (
    "/home/dyf/code/distill/MAES/storage/data_distill_/"
    "video_mmmu-num_342-token_2048-sample_at1.0-0506110803/teacher_hidden.pt"
)
DEFAULT_OUTPUT_DIR = "./results"  # do not change 
DEFAULT_OUTPUT_NAME = "a3_teacher_core_outlier_tsne"
SOURCE_ORDER = ("gqa", "coco", "m4-instruct", "video-mmmu")
SOURCE_COLORS = {
    "gqa": "#1cb3b0",
    "coco": "#fcd924",
    "m4-instruct": "#f1895e",
    "video-mmmu": "#3e96d1",
}
MODALITY_MARKERS = {
    "visual": "o",
    "text": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot teacher hidden states in a shared t-SNE with core/outlier panels."
    )
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--video_mmmu_path", type=str, default=DEFAULT_VIDEO_MMMU_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--summary_path", type=str, default="")
    parser.add_argument("--min_modality_tokens", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--core_ratio", type=float, default=0.20)
    parser.add_argument("--outlier_ratio", type=float, default=0.20)
    parser.add_argument("--min_samples", type=int, default=200)
    parser.add_argument(
        "--refresh_from_pt",
        action="store_true",
        help="Ignore cached summary and rebuild from teacher pt.",
    )
    return parser.parse_args()


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_payload(path: str) -> Dict[str, object]:
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {path}, got {type(payload)}")
    payload = _materialize_sharded_payload(payload, path)
    payload["__source_path__"] = path
    return payload


def _materialize_sharded_payload(payload: Dict[str, object], payload_path: str) -> Dict[str, object]:
    shards = payload.get("shards")
    if not isinstance(shards, list):
        return payload

    payload_dir = os.path.dirname(payload_path)
    merged: Dict[str, List[torch.Tensor]] = {}
    passthrough: Dict[str, object] = {}
    total_samples = 0
    for shard_meta in shards:
        shard_path = os.path.join(payload_dir, shard_meta["path"])
        shard_payload = _torch_load(shard_path)
        total_samples += int(shard_payload.get("num_samples", shard_meta.get("num_samples", 0)))
        for key, value in shard_payload.items():
            if torch.is_tensor(value):
                merged.setdefault(key, []).append(value)
            elif key not in passthrough:
                passthrough[key] = value

    out: Dict[str, object] = {
        key: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        for key, parts in merged.items()
    }
    for key, value in payload.items():
        if key == "shards":
            continue
        if key not in out:
            out[key] = value
    for key, value in passthrough.items():
        if key not in out:
            out[key] = value
    out["num_samples"] = total_samples
    out["metadata"] = payload.get("metadata", {})
    return out


def _hidden_key(payload: Dict[str, object]) -> str:
    if "teacher_cache" in payload:
        return "teacher_cache"
    raise KeyError("Expected `teacher_cache` in payload.")


def _valid_mask(payload: Dict[str, object]) -> torch.Tensor:
    if "attention_mask" in payload:
        return payload["attention_mask"] > 0
    if "modality_labels" in payload:
        return payload["modality_labels"] >= 0
    raise KeyError("Cannot infer valid token mask from payload.")


def _normalize_source_name(name: str) -> str:
    return name.replace("_", "-")


def _build_teacher_points(
    payload: Dict[str, object],
    min_modality_tokens: int,
) -> Tuple[np.ndarray, List[str], List[str]]:
    hidden = payload[_hidden_key(payload)]
    modality = payload["modality_labels"]
    valid = _valid_mask(payload)
    dataset_ids = payload.get("dataset_ids")
    metadata = payload.get("metadata", {})
    dataset_id_to_name = metadata.get("dataset_id_to_name", {})
    if dataset_ids is None:
        raise KeyError("Expected `dataset_ids` in teacher payload.")

    features: List[np.ndarray] = []
    source_labels: List[str] = []
    modality_labels: List[str] = []
    for idx in range(hidden.shape[0]):
        source_id = int(dataset_ids[idx].item())
        source_name = _normalize_source_name(dataset_id_to_name.get(source_id, str(source_id)))
        sample_mask = valid[idx]
        for mod_id, mod_name in ((1, "visual"), (0, "text")):
            mod_mask = sample_mask & (modality[idx] == mod_id)
            if int(mod_mask.sum().item()) < min_modality_tokens:
                continue
            centroid = hidden[idx, mod_mask].to(torch.float32).mean(dim=0).cpu().numpy()
            features.append(centroid)
            source_labels.append(source_name)
            modality_labels.append(mod_name)
    return np.stack(features, axis=0), source_labels, modality_labels


def _concat_point_sets(
    point_sets: List[Tuple[np.ndarray, List[str], List[str]]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    features = np.concatenate([item[0] for item in point_sets], axis=0)
    source_labels: List[str] = []
    modality_labels: List[str] = []
    for _, src, mod in point_sets:
        source_labels.extend(src)
        modality_labels.extend(mod)
    return features, source_labels, modality_labels


def _fit_tsne(
    features: np.ndarray,
    pca_dim: int,
    perplexity: float,
    random_state: int,
) -> Tuple[np.ndarray, float]:
    mean = features.mean(axis=0, keepdims=True)
    std = np.clip(features.std(axis=0, keepdims=True), 1e-6, None)
    standardized = (features - mean) / std

    effective_dim = min(pca_dim, standardized.shape[0], standardized.shape[1])
    pca = PCA(n_components=effective_dim, random_state=random_state)
    pca_features = pca.fit_transform(standardized)

    effective_perplexity = min(perplexity, max(5.0, (pca_features.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(pca_features), float(np.sum(pca.explained_variance_ratio_))


def _split_modal_embeddings(
    embedding: np.ndarray,
    modality_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    modality_arr = np.asarray(modality_labels)
    text = embedding[modality_arr == "text"]
    visual = embedding[modality_arr == "visual"]
    if text.shape[0] != visual.shape[0]:
        raise ValueError("Expected equal counts for text and visual points.")
    return text, visual


def _counts(n: int, ratio: float, min_samples: int) -> int:
    return min(n, max(min_samples, int(round(n * ratio))))


def _kde_log_density(points: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(points.T)
    density = kde(points.T)
    return np.log(np.clip(density, 1e-12, None))


def _radius(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    return np.linalg.norm(points - center, axis=1)


def _select_sample_masks(
    text_tsne: np.ndarray,
    visual_tsne: np.ndarray,
    core_ratio: float,
    outlier_ratio: float,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    core_count = _counts(text_tsne.shape[0], core_ratio, min_samples)
    outlier_count = _counts(text_tsne.shape[0], outlier_ratio, min_samples)
    sample_centers = 0.5 * (text_tsne + visual_tsne)
    sample_radius = _radius(sample_centers)

    core_idx = np.argsort(sample_radius)[:core_count]
    outlier_idx = np.argsort(sample_radius)[-outlier_count:]
    core_mask = np.zeros(text_tsne.shape[0], dtype=bool)
    outlier_mask = np.zeros(text_tsne.shape[0], dtype=bool)
    core_mask[core_idx] = True
    outlier_mask[outlier_idx] = True
    return core_mask, outlier_mask


def _sample_masks_to_point_masks(modality_labels: List[str], sample_mask: np.ndarray) -> np.ndarray:
    point_mask = np.zeros(len(modality_labels), dtype=bool)
    text_idx = 0
    visual_idx = 0
    for idx, modality in enumerate(modality_labels):
        if modality == "visual":
            point_mask[idx] = sample_mask[visual_idx]
            visual_idx += 1
        elif modality == "text":
            point_mask[idx] = sample_mask[text_idx]
            text_idx += 1
        else:
            raise ValueError(f"Unknown modality label: {modality}")
    if text_idx != sample_mask.shape[0] or visual_idx != sample_mask.shape[0]:
        raise ValueError("Point labels are not aligned one visual/text pair per sample.")
    return point_mask


def _select_uniform_sample_mask(
    text_tsne: np.ndarray,
    visual_tsne: np.ndarray,
    count: int,
) -> np.ndarray:
    sample_centers = 0.5 * (text_tsne + visual_tsne)
    n_samples = sample_centers.shape[0]
    if count >= n_samples:
        return np.ones(n_samples, dtype=bool)

    global_center = sample_centers.mean(axis=0, keepdims=True)
    start_idx = int(np.argmin(np.linalg.norm(sample_centers - global_center, axis=1)))
    selected = [start_idx]
    min_dist = np.linalg.norm(sample_centers - sample_centers[start_idx], axis=1)
    min_dist[start_idx] = -1.0

    while len(selected) < count:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        next_dist = np.linalg.norm(sample_centers - sample_centers[next_idx], axis=1)
        min_dist = np.minimum(min_dist, next_dist)
        min_dist[selected] = -1.0

    mask = np.zeros(n_samples, dtype=bool)
    mask[selected] = True
    return mask


def _legend_handles() -> Tuple[List[Line2D], List[str]]:
    handles: List[Line2D] = []
    labels: List[str] = []
    for source_name in SOURCE_ORDER:
        for mod_name in ("visual", "text"):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=MODALITY_MARKERS[mod_name],
                    color="none",
                    markerfacecolor=SOURCE_COLORS[source_name],
                    markeredgecolor="black",
                    markeredgewidth=0.65,
                    markersize=7,
                    linestyle="none",
                )
            )
            labels.append(f"{source_name} {mod_name}")
    return handles, labels


def _plot_density_background(ax, points: np.ndarray) -> None:
    if points.shape[0] < 4:
        return
    try:
        kde = gaussian_kde(points.T)
    except np.linalg.LinAlgError:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_grid = np.linspace(x_min, x_max, DENSITY_GRID_SIZE)
    y_grid = np.linspace(y_min, y_max, DENSITY_GRID_SIZE)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    if not np.any(density > 0):
        return

    levels = np.linspace(float(density.min()), float(density.max()), DENSITY_FILLED_LEVELS)
    ax.contourf(
        xx,
        yy,
        density,
        levels=levels,
        cmap="PuBu",
        alpha=DENSITY_ALPHA,
        antialiased=True,
        zorder=0,
    )
    ax.contour(
        xx,
        yy,
        density,
        levels=DENSITY_LINE_LEVELS,
        colors="#cbdfdf", # cbdfdf
        alpha=0.22,
        linewidths=0.55,
        zorder=0.5,
    )


def _plot(
    output_path: str,
    embedding: np.ndarray,
    source_labels: List[str],
    modality_labels: List[str],
    core_mask: np.ndarray,
    outlier_mask: np.ndarray,
    uniform_mask: np.ndarray,
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    source_arr = np.asarray(source_labels)
    modality_arr = np.asarray(modality_labels)
    x_pad = 0.05 * (float(np.max(embedding[:, 0])) - float(np.min(embedding[:, 0])))
    y_pad = 0.05 * (float(np.max(embedding[:, 1])) - float(np.min(embedding[:, 1])))
    x_lim = (float(np.min(embedding[:, 0])) - x_pad, float(np.max(embedding[:, 0])) + x_pad)
    y_lim = (float(np.min(embedding[:, 1])) - y_pad, float(np.max(embedding[:, 1])) + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    legend_handles, legend_labels = _legend_handles()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        frameon=True,
        ncol=len(legend_labels),
        fontsize=LEGEND_FONT_SIZE,
    )
    panel_specs = [
        (axes[0], "Core Hidden", core_mask),
        (axes[1], "Outlier Hidden", outlier_mask),
        (axes[2], "Uniform Hidden", uniform_mask),
    ]
    for ax, title, active_mask in panel_specs:
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        _plot_density_background(ax, embedding[active_mask])

        for source_name in SOURCE_ORDER:
            for mod_name, marker, size, alpha in (
                ("visual", MODALITY_MARKERS["visual"], 12, 0.1),
                ("text", MODALITY_MARKERS["text"], 16, 0.1),
            ):
                bg_mask = (source_arr == source_name) & (modality_arr == mod_name)
                if not np.any(bg_mask):
                    continue
                points = embedding[bg_mask]
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=size,
                    c=SOURCE_COLORS[source_name],
                    marker=marker,
                    alpha=alpha,
                    edgecolors="none",
                    linewidths=0.0,
                    zorder=1,
                )

        for source_name in SOURCE_ORDER:
            for mod_name, marker, size, alpha in (
                ("visual", MODALITY_MARKERS["visual"], 24, 0.8),
                ("text", MODALITY_MARKERS["text"], 34, 0.8),
            ):
                fg_mask = active_mask & (source_arr == source_name) & (modality_arr == mod_name)
                if not np.any(fg_mask):
                    continue
                points = embedding[fg_mask]
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=size,
                    c=SOURCE_COLORS[source_name],
                    marker=marker,
                    alpha=alpha,
                    edgecolors="black",
                    linewidths=0.65,
                    zorder=3,
                )

        # ax.set_title(title, fontsize=PANEL_TITLE_FONT_SIZE, pad=10)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.25, linestyle="--", linewidth=0.8)
        ax.grid(which="minor", alpha=0.14, linestyle="--", linewidth=0.45)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_cached_plot_data(summary_path: str) -> Optional[Dict[str, object]]:
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    required = [
        "embedding",
        "source_labels",
        "modality_labels",
        "core_mask",
        "outlier_mask",
        "uniform_mask",
        "sources",
    ]
    if any(key not in summary for key in required):
        return None
    if tuple(summary["sources"]) != SOURCE_ORDER:
        return None
    return summary


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_stem = args.output_name.strip() or DEFAULT_OUTPUT_NAME
    figure_path = os.path.join(args.output_dir, f"{output_stem}.png")
    summary_path = args.summary_path.strip() or os.path.join(args.output_dir, f"{output_stem}_summary.json")

    summary = None if args.refresh_from_pt else _load_cached_plot_data(summary_path)
    if summary is None:
        base_payload = _load_payload(args.teacher_path)
        video_payload = _load_payload(args.video_mmmu_path)
        features, source_labels, modality_labels = _concat_point_sets(
            [
                _build_teacher_points(base_payload, min_modality_tokens=args.min_modality_tokens),
                _build_teacher_points(video_payload, min_modality_tokens=args.min_modality_tokens),
            ]
        )
        embedding, explained_var_sum = _fit_tsne(
            features,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
        text_tsne, visual_tsne = _split_modal_embeddings(embedding, modality_labels)
        core_samples, outlier_samples = _select_sample_masks(
            text_tsne,
            visual_tsne,
            core_ratio=args.core_ratio,
            outlier_ratio=args.outlier_ratio,
            min_samples=args.min_samples,
        )
        core_mask = _sample_masks_to_point_masks(modality_labels, core_samples)
        outlier_mask = _sample_masks_to_point_masks(modality_labels, outlier_samples)
        uniform_samples = _select_uniform_sample_mask(
            text_tsne,
            visual_tsne,
            count=int(core_samples.sum()),
        )
        uniform_mask = _sample_masks_to_point_masks(modality_labels, uniform_samples)

        summary = {
            "teacher_path": args.teacher_path,
            "video_mmmu_path": args.video_mmmu_path,
            "output_figure": figure_path,
            "num_points": int(embedding.shape[0]),
            "sources": list(SOURCE_ORDER),
            "modalities": ["text", "visual"],
            "core_ratio": args.core_ratio,
            "outlier_ratio": args.outlier_ratio,
            "min_samples": args.min_samples,
            "pca_dim": int(min(args.pca_dim, features.shape[0], features.shape[1])),
            "perplexity": float(min(args.perplexity, max(5.0, (embedding.shape[0] - 1) / 3.0))),
            "explained_variance_ratio_sum": explained_var_sum,
            "embedding": embedding.tolist(),
            "source_labels": source_labels,
            "modality_labels": modality_labels,
            "core_mask": core_mask.tolist(),
            "outlier_mask": outlier_mask.tolist(),
            "uniform_mask": uniform_mask.tolist(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot(
        figure_path,
        embedding=np.asarray(summary["embedding"], dtype=np.float32),
        source_labels=list(summary["source_labels"]),
        modality_labels=list(summary["modality_labels"]),
        core_mask=np.asarray(summary["core_mask"], dtype=bool),
        outlier_mask=np.asarray(summary["outlier_mask"], dtype=bool),
        uniform_mask=np.asarray(summary["uniform_mask"], dtype=bool),
        dpi=args.dpi,
    )

    print(f"Saved figure to {figure_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
