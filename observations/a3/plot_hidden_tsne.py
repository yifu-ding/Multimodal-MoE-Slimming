#!/usr/bin/env python3
"""A3: t-SNE visualization for teacher vs distilled hidden states."""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_maes")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from observations.common import ensure_dir, print_saved_artifact_message


DEFAULT_TEACHER_PATH = ("/home/dyf/code/distill/MAES/storage/data_distill_kimi/mixed-num_342-token_2048-sample_at1.0-0423143643/teacher_hidden.pt")
# DEFAULT_TEACHER_PATH=("/home/dyf/code/distill/MAES/storage/data_distill_kimi/mixed-num_1024-token_2048-sample_at1.0-0421175408/teacher_hidden.pt")

DEFAULT_DISTILLED_PATH = ("/home/dyf/code/distill/MAES/storage/data_distill_kimi/online-distilled-0429022932-div_no_div-dist_full/distilled_hidden-step5000.pt")

# 只用gqa的teacher
# DEFAULT_TEACHER_PATH=("/home/dyf/code/distill/MAES/storage/data_distill_kimi/gqa-num_342-token_2048-sample_at1.0-0429161205/teacher_hidden.pt")
# 只用gqa的distilled
# DEFAULT_DISTILLED_PATH=("/home/dyf/code/distill/MAES/storage/data_distill_kimi/online-distilled-0429003527-div_full-dist_full/distilled_hidden-step5000.pt")

DEFAULT_OUTPUT_DIR = "/home/dyf/code/distill/MAES/observations/a3/results"
COLOR_VISUAL = "#3E96D1"
COLOR_TEXT = "#FACD56"
COLOR_TEACHER_EDGE = "#8D99A8"
COLOR_DISTILLED_EDGE = "#5C2C00"


@dataclass
class PointSet:
    features: np.ndarray
    text_ratio: np.ndarray
    token_lengths: np.ndarray
    labels: List[str]
    source: str
    per_sample_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the hidden-state manifold difference between a teacher cache "
            "and a distilled synthetic hidden set."
        )
    )
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--distilled_path", type=str, default=DEFAULT_DISTILLED_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help=(
            "Optional output stem without extension. When omitted, a unique name is "
            "derived from teacher/distilled parent directories."
        ),
    )
    parser.add_argument(
        "--point_mode",
        type=str,
        default="modality_centroid",
        choices=("sample_mean", "modality_centroid"),
        help=(
            "sample_mean: one point per sample; modality_centroid: up to two points "
            "per sample, split into text/visual centroids."
        ),
    )
    parser.add_argument("--min_modality_tokens", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _load_payload(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {path}, got {type(payload)}")
    return _materialize_sharded_payload(payload, path)


def _materialize_sharded_payload(
    payload: Dict[str, torch.Tensor],
    payload_path: str,
) -> Dict[str, torch.Tensor]:
    shards = payload.get("shards")
    if not isinstance(shards, list):
        return payload

    payload_dir = os.path.dirname(payload_path)
    merged: Dict[str, List[torch.Tensor]] = {}
    passthrough: Dict[str, object] = {}
    total_samples = 0
    for shard_meta in shards:
        shard_rel_path = shard_meta["path"]
        shard_path = os.path.join(payload_dir, shard_rel_path)
        shard_payload = torch.load(shard_path, map_location="cpu")
        total_samples += int(shard_payload.get("num_samples", shard_meta.get("num_samples", 0)))
        for key, value in shard_payload.items():
            if torch.is_tensor(value):
                merged.setdefault(key, []).append(value)
            elif key not in passthrough:
                passthrough[key] = value

    out: Dict[str, torch.Tensor] = {
        key: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        for key, parts in merged.items()
    }
    for key, value in payload.items():
        if key in {"shards"}:
            continue
        if key not in out:
            out[key] = value
    for key, value in passthrough.items():
        if key not in out:
            out[key] = value
    out["num_samples"] = total_samples
    out["metadata"] = payload.get("metadata", {})
    return out


def _valid_mask(payload: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "attention_mask" in payload:
        return payload["attention_mask"] > 0
    if "modality_labels" in payload:
        return payload["modality_labels"] >= 0
    raise KeyError("Cannot infer valid token mask from payload.")


def _hidden_key(payload: Dict[str, torch.Tensor]) -> str:
    if "teacher_cache" in payload:
        return "teacher_cache"
    if "synthetic_hidden" in payload:
        return "synthetic_hidden"
    raise KeyError("Unknown hidden tensor key in payload.")


def _build_sample_mean_points(payload: Dict[str, torch.Tensor], source: str) -> PointSet:
    hidden = payload[_hidden_key(payload)].to(torch.float32)
    modality = payload["modality_labels"]
    valid = _valid_mask(payload)

    features = []
    text_ratio = []
    token_lengths = []
    labels = []
    for idx in range(hidden.shape[0]):
        sample_mask = valid[idx]
        sample_hidden = hidden[idx, sample_mask]
        features.append(sample_hidden.mean(dim=0))
        n_tokens = int(sample_mask.sum().item())
        n_text = int(((modality[idx] == 0) & sample_mask).sum().item())
        text_ratio.append(n_text / max(n_tokens, 1))
        token_lengths.append(n_tokens)
        labels.append(f"{source}_sample")

    metadata = payload.get("metadata", {})
    per_sample_tokens = int(metadata.get("compressed_length", hidden.shape[1]))
    return PointSet(
        features=torch.stack(features, dim=0).numpy(),
        text_ratio=np.asarray(text_ratio, dtype=np.float32),
        token_lengths=np.asarray(token_lengths, dtype=np.int32),
        labels=labels,
        source=source,
        per_sample_tokens=per_sample_tokens,
    )


def _build_modality_centroid_points(
    payload: Dict[str, torch.Tensor],
    source: str,
    min_modality_tokens: int,
) -> PointSet:
    hidden = payload[_hidden_key(payload)].to(torch.float32)
    modality = payload["modality_labels"]
    valid = _valid_mask(payload)

    features = []
    text_ratio = []
    token_lengths = []
    labels = []
    for idx in range(hidden.shape[0]):
        sample_mask = valid[idx]
        for mod_id, mod_name in ((0, "text"), (1, "visual")):
            mod_mask = sample_mask & (modality[idx] == mod_id)
            n_tokens = int(mod_mask.sum().item())
            if n_tokens < min_modality_tokens:
                continue
            features.append(hidden[idx, mod_mask].mean(dim=0))
            token_lengths.append(n_tokens)
            text_ratio.append(1.0 if mod_id == 0 else 0.0)
            labels.append(f"{source}_{mod_name}")

    metadata = payload.get("metadata", {})
    per_sample_tokens = int(metadata.get("compressed_length", hidden.shape[1]))
    return PointSet(
        features=torch.stack(features, dim=0).numpy(),
        text_ratio=np.asarray(text_ratio, dtype=np.float32),
        token_lengths=np.asarray(token_lengths, dtype=np.int32),
        labels=labels,
        source=source,
        per_sample_tokens=per_sample_tokens,
    )


def build_points(
    payload: Dict[str, torch.Tensor],
    source: str,
    point_mode: str,
    min_modality_tokens: int,
) -> PointSet:
    if point_mode == "sample_mean":
        return _build_sample_mean_points(payload, source)
    if point_mode == "modality_centroid":
        return _build_modality_centroid_points(payload, source, min_modality_tokens)
    raise ValueError(f"Unsupported point_mode: {point_mode}")


def _fit_embedding(
    teacher_points: PointSet,
    distilled_points: PointSet,
    pca_dim: int,
    perplexity: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_features = np.concatenate([teacher_points.features, distilled_points.features], axis=0)
    feat_mean = all_features.mean(axis=0, keepdims=True)
    feat_std = all_features.std(axis=0, keepdims=True)
    standardized = (all_features - feat_mean) / np.clip(feat_std, 1e-6, None)

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
    embedding_2d = tsne.fit_transform(pca_features)
    return embedding_2d, pca_features, pca.explained_variance_ratio_


def _covariance_ellipse(points_2d: np.ndarray, n_std: float) -> Ellipse:
    center = points_2d.mean(axis=0)
    cov = np.cov(points_2d.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 1e-12))
    return Ellipse(xy=center, width=width, height=height, angle=angle)


def _draw_teacher_density(ax: plt.Axes, points_2d: np.ndarray) -> None:
    if points_2d.shape[0] < 32:
        return
    xmin, ymin = points_2d.min(axis=0)
    xmax, ymax = points_2d.max(axis=0)
    pad_x = max((xmax - xmin) * 0.08, 1e-3)
    pad_y = max((ymax - ymin) * 0.08, 1e-3)
    xx, yy = np.mgrid[
        (xmin - pad_x):(xmax + pad_x):180j,
        (ymin - pad_y):(ymax + pad_y):180j,
    ]
    kde = gaussian_kde(points_2d.T)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.quantile(zz, [0.70, 0.85, 0.94])
    ax.contourf(
        xx,
        yy,
        zz,
        levels=np.concatenate([[zz.min()], levels]),
        colors=["#d9dde3", "#b9c2cf", "#95a4b8"],
        alpha=0.30,
        antialiased=True,
    )
    ax.contour(xx, yy, zz, levels=levels, colors="#6f7f93", linewidths=0.8, alpha=0.7)


def _draw_density(ax: plt.Axes, points_2d: np.ndarray, color: str, alpha: float = 0.18) -> None:
    if points_2d.shape[0] < 32:
        return
    xmin, ymin = points_2d.min(axis=0)
    xmax, ymax = points_2d.max(axis=0)
    pad_x = max((xmax - xmin) * 0.08, 1e-3)
    pad_y = max((ymax - ymin) * 0.08, 1e-3)
    xx, yy = np.mgrid[
        (xmin - pad_x):(xmax + pad_x):180j,
        (ymin - pad_y):(ymax + pad_y):180j,
    ]
    kde = gaussian_kde(points_2d.T)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.quantile(zz, [0.70, 0.85, 0.94])
    ax.contourf(
        xx,
        yy,
        zz,
        levels=np.concatenate([[zz.min()], levels]),
        colors=[color, color, color],
        alpha=alpha,
        antialiased=True,
    )
    ax.contour(xx, yy, zz, levels=levels, colors=color, linewidths=0.9, alpha=min(alpha + 0.18, 0.55))


def _summarize_geometry(teacher_pca: np.ndarray, distilled_pca: np.ndarray) -> Dict[str, float]:
    teacher_center = teacher_pca.mean(axis=0)
    distilled_center = distilled_pca.mean(axis=0)
    teacher_radius = np.linalg.norm(teacher_pca - teacher_center, axis=1)
    distilled_radius = np.linalg.norm(distilled_pca - distilled_center, axis=1)
    teacher_r90 = float(np.percentile(teacher_radius, 90))
    distilled_r90 = float(np.percentile(distilled_radius, 90))
    return {
        "teacher_radius_p50": float(np.percentile(teacher_radius, 50)),
        "teacher_radius_p90": teacher_r90,
        "distilled_radius_p50": float(np.percentile(distilled_radius, 50)),
        "distilled_radius_p90": distilled_r90,
        "radius_shrink_pct": float(100.0 * (1.0 - distilled_r90 / max(teacher_r90, 1e-8))),
        "center_gap_l2": float(np.linalg.norm(teacher_center - distilled_center)),
        "center_gap_over_teacher_r90": float(
            np.linalg.norm(teacher_center - distilled_center) / max(teacher_r90, 1e-8)
        ),
    }


def _save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _sanitize_stem(text: str) -> str:
    text = text.strip().replace(os.sep, "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-") or "unnamed"


def _derive_output_stem(point_mode: str, teacher_path: str, distilled_path: str, output_name: str) -> str:
    if output_name.strip():
        return _sanitize_stem(output_name)
    teacher_tag = _sanitize_stem(os.path.basename(os.path.dirname(teacher_path)))
    distilled_tag = _sanitize_stem(os.path.basename(os.path.dirname(distilled_path)))
    return f"a3_hidden_tsne_{point_mode}_{teacher_tag}__{distilled_tag}"


def plot_embedding(
    output_path: str,
    teacher_embed: np.ndarray,
    distilled_embed: np.ndarray,
    teacher_points: PointSet,
    distilled_points: PointSet,
    summary: Dict[str, float],
    point_mode: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 7.6), constrained_layout=True)
    if point_mode == "modality_centroid":
        teacher_labels = np.asarray(teacher_points.labels)
        distilled_labels = np.asarray(distilled_points.labels)
        group_specs = [
            ("teacher_text", teacher_embed, teacher_labels, COLOR_TEXT, 18, 0.22, COLOR_TEACHER_EDGE, "Teacher text"),
            ("teacher_visual", teacher_embed, teacher_labels, COLOR_VISUAL, 18, 0.22, COLOR_TEACHER_EDGE, "Teacher visual"),
            ("distilled_text", distilled_embed, distilled_labels, COLOR_TEXT, 28, 0.92, COLOR_DISTILLED_EDGE, "Distilled text"),
            ("distilled_visual", distilled_embed, distilled_labels, COLOR_VISUAL, 28, 0.92, COLOR_DISTILLED_EDGE, "Distilled visual"),
        ]
        for label_key, embed, labels, color, size, alpha, edgecolor, legend_name in group_specs:
            mask = labels == label_key
            if not np.any(mask):
                continue
            group_points = embed[mask]
            if label_key.startswith("teacher_"):
                _draw_density(ax, group_points, color=color, alpha=0.15)
            ax.scatter(
                group_points[:, 0],
                group_points[:, 1],
                s=size,
                c=color,
                alpha=alpha,
                edgecolors=edgecolor if label_key.startswith("distilled_") else "none",
                linewidths=0.25 if label_key.startswith("distilled_") else 0.0,
                label=f"{legend_name} ({group_points.shape[0]})",
                zorder=3 if label_key.startswith("distilled_") else 2,
            )
    else:
        _draw_teacher_density(ax, teacher_embed)
        ax.scatter(
            teacher_embed[:, 0],
            teacher_embed[:, 1],
            s=16,
            c="#93a1b2",
            alpha=0.28,
            linewidths=0.0,
            label=f"Teacher ({teacher_embed.shape[0]} points)",
            zorder=2,
        )
        ax.scatter(
            distilled_embed[:, 0],
            distilled_embed[:, 1],
            s=26,
            c="#d95f02",
            alpha=0.88,
            edgecolors="#4b1d00",
            linewidths=0.25,
            label=f"Distilled ({distilled_embed.shape[0]} points)",
            zorder=3,
        )

        teacher_ellipse = _covariance_ellipse(teacher_embed, n_std=2.0)
        teacher_ellipse.set_facecolor("none")
        teacher_ellipse.set_edgecolor("#6f7f93")
        teacher_ellipse.set_linewidth(1.2)
        teacher_ellipse.set_linestyle("--")
        teacher_ellipse.set_alpha(0.85)
        ax.add_patch(teacher_ellipse)

        distilled_ellipse = _covariance_ellipse(distilled_embed, n_std=2.0)
        distilled_ellipse.set_facecolor("none")
        distilled_ellipse.set_edgecolor("#a63c00")
        distilled_ellipse.set_linewidth(1.5)
        distilled_ellipse.set_alpha(0.95)
        ax.add_patch(distilled_ellipse)

    teacher_center = teacher_embed.mean(axis=0)
    distilled_center = distilled_embed.mean(axis=0)
    ax.scatter(
        teacher_center[0],
        teacher_center[1],
        marker="X",
        s=110,
        c="#44556b",
        linewidths=0.0,
        zorder=4,
    )
    ax.scatter(
        distilled_center[0],
        distilled_center[1],
        marker="X",
        s=125,
        c="#7f2704",
        linewidths=0.0,
        zorder=5,
    )

    ax.set_title(
        "A3 t-SNE of Hidden Calibration Manifold\n"
        "Distilled synthetic hidden states stay near the teacher core while contracting diffuse tails",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper right", frameon=True, ncol=2 if point_mode == "modality_centroid" else 1)

    teacher_text_ratio = float(teacher_points.text_ratio.mean())
    distilled_text_ratio = float(distilled_points.text_ratio.mean())
    note_lines = [
        f"Point mode: {point_mode}",
        f"Teacher budget: {teacher_points.per_sample_tokens} tokens/sample",
        f"Distilled budget: {distilled_points.per_sample_tokens} tokens/sample",
        f"PCA-space r90: {summary['teacher_radius_p90']:.2f} -> {summary['distilled_radius_p90']:.2f}",
        f"Radius shrinkage: {summary['radius_shrink_pct']:.1f}%",
        f"Center gap / teacher r90: {summary['center_gap_over_teacher_r90'] * 100:.1f}%",
        f"Mean text ratio: {teacher_text_ratio:.3f} -> {distilled_text_ratio:.3f}",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(note_lines),
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#c7ccd4"},
    )

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    teacher_payload = _load_payload(args.teacher_path)
    distilled_payload = _load_payload(args.distilled_path)

    teacher_points = build_points(
        teacher_payload,
        source="teacher",
        point_mode=args.point_mode,
        min_modality_tokens=args.min_modality_tokens,
    )
    distilled_points = build_points(
        distilled_payload,
        source="distilled",
        point_mode=args.point_mode,
        min_modality_tokens=args.min_modality_tokens,
    )

    embedding_2d, pca_features, pca_var_ratio = _fit_embedding(
        teacher_points=teacher_points,
        distilled_points=distilled_points,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        random_state=args.random_state,
    )
    split = teacher_points.features.shape[0]
    teacher_embed = embedding_2d[:split]
    distilled_embed = embedding_2d[split:]
    teacher_pca = pca_features[:split]
    distilled_pca = pca_features[split:]

    summary = {
        "point_mode": args.point_mode,
        "teacher_path": args.teacher_path,
        "distilled_path": args.distilled_path,
        "teacher_num_points": int(teacher_points.features.shape[0]),
        "distilled_num_points": int(distilled_points.features.shape[0]),
        "teacher_tokens_per_point_mean": float(np.mean(teacher_points.token_lengths)),
        "distilled_tokens_per_point_mean": float(np.mean(distilled_points.token_lengths)),
        "teacher_text_ratio_mean": float(np.mean(teacher_points.text_ratio)),
        "distilled_text_ratio_mean": float(np.mean(distilled_points.text_ratio)),
        "teacher_text_ratio_std": float(np.std(teacher_points.text_ratio)),
        "distilled_text_ratio_std": float(np.std(distilled_points.text_ratio)),
        "pca_explained_variance_top10_sum": float(np.sum(pca_var_ratio[:10])),
    }
    summary.update(_summarize_geometry(teacher_pca=teacher_pca, distilled_pca=distilled_pca))

    stem = _derive_output_stem(
        point_mode=args.point_mode,
        teacher_path=args.teacher_path,
        distilled_path=args.distilled_path,
        output_name=args.output_name,
    )
    plot_path = os.path.join(args.output_dir, f"{stem}.png")
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.json")
    embedding_path = os.path.join(args.output_dir, f"{stem}_embedding.pt")

    plot_embedding(
        output_path=plot_path,
        teacher_embed=teacher_embed,
        distilled_embed=distilled_embed,
        teacher_points=teacher_points,
        distilled_points=distilled_points,
        summary=summary,
        point_mode=args.point_mode,
        dpi=args.dpi,
    )

    _save_json(summary_path, summary)
    torch.save(
        {
            "teacher_embedding_2d": torch.from_numpy(teacher_embed),
            "distilled_embedding_2d": torch.from_numpy(distilled_embed),
            "teacher_text_ratio": torch.from_numpy(teacher_points.text_ratio),
            "distilled_text_ratio": torch.from_numpy(distilled_points.text_ratio),
            "teacher_token_lengths": torch.from_numpy(teacher_points.token_lengths),
            "distilled_token_lengths": torch.from_numpy(distilled_points.token_lengths),
            "teacher_labels": teacher_points.labels,
            "distilled_labels": distilled_points.labels,
            "summary": summary,
        },
        embedding_path,
    )

    print_saved_artifact_message(plot_path, "A3 t-SNE visualization")
    print_saved_artifact_message(summary_path, "A3 summary JSON")
    print_saved_artifact_message(embedding_path, "A3 embedding tensors")


if __name__ == "__main__":
    main()
