#!/usr/bin/env python3
"""A3: split a teacher hidden cache into core / outlier subsets and visualize them."""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_maes")

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


# DEFAULT_TEACHER_PATH = (
#     "/home/dyf/code/distill/MAES/storage/data_distill_kimi/"
#     "gqa-num_342-token_2048-sample_at1.0-0429161205/teacher_hidden.pt"
# )
DEFAULT_TEACHER_PATH=("/home/dyf/code/distill/MAES/storage/data_distill_kimi/gqa-num_1024-token_2048-sample_at1.0-0429164415/teacher_hidden.pt")
DEFAULT_OUTPUT_DIR = "/home/dyf/code/distill/MAES/observations/a3/split-hidden"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a teacher hidden cache into core and outlier subsets based on "
            "sample-level compactness and modality-level deviation."
        )
    )
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--min_samples", type=int, default=342)
    parser.add_argument("--sample_core_ratio", type=float, default=0.20)
    parser.add_argument("--sample_outlier_ratio", type=float, default=0.20)
    parser.add_argument("--modality_core_ratio", type=float, default=0.20)
    parser.add_argument("--modality_outlier_ratio", type=float, default=0.20)
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
    return f"a3_core_outlier_{teacher_tag}"


def _sample_mean_features(payload: Dict[str, torch.Tensor]) -> np.ndarray:
    hidden = payload[_hidden_key(payload)].to(torch.float32)
    valid = _valid_mask(payload)
    rows = []
    for idx in range(hidden.shape[0]):
        rows.append(hidden[idx, valid[idx]].mean(dim=0))
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


def _standardize_fit(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (features - mean) / std, mean, std


def _select_topk_smallest(values: np.ndarray, count: int) -> np.ndarray:
    idx = np.argsort(values)[:count]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[idx] = True
    return mask


def _select_topk_largest(values: np.ndarray, count: int) -> np.ndarray:
    idx = np.argsort(values)[-count:]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[idx] = True
    return mask


def _counts_from_ratio(n: int, ratio: float, min_samples: int) -> int:
    return min(n, max(min_samples, int(round(n * ratio))))


def _rank_mask(values: np.ndarray, count: int, smallest: bool) -> np.ndarray:
    order = np.argsort(values)
    if smallest:
        selected = order[:count]
    else:
        selected = order[-count:]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def _adaptive_core_mask(
    sample_radius: np.ndarray,
    text_radius: np.ndarray,
    visual_radius: np.ndarray,
    core_count: int,
    min_samples: int,
) -> np.ndarray:
    # Prefer the intersection so "core" really means jointly central across views.
    for scale in (1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4):
        k = min(sample_radius.shape[0], max(min_samples, int(round(core_count * scale))))
        sample_mask = _rank_mask(sample_radius, k, smallest=True)
        text_mask = _rank_mask(text_radius, k, smallest=True)
        visual_mask = _rank_mask(visual_radius, k, smallest=True)
        mask = sample_mask & text_mask & visual_mask
        if int(mask.sum()) >= min_samples:
            return mask
    # Fallback: majority vote still keeps the set central without exploding overlap.
    sample_mask = _rank_mask(sample_radius, core_count, smallest=True)
    text_mask = _rank_mask(text_radius, core_count, smallest=True)
    visual_mask = _rank_mask(visual_radius, core_count, smallest=True)
    votes = sample_mask.astype(np.int32) + text_mask.astype(np.int32) + visual_mask.astype(np.int32)
    mask = votes >= 2
    if int(mask.sum()) >= min_samples:
        return mask
    return sample_mask | text_mask | visual_mask


def _adaptive_outlier_mask(
    sample_radius: np.ndarray,
    text_radius: np.ndarray,
    visual_radius: np.ndarray,
    outlier_count: int,
    min_samples: int,
) -> np.ndarray:
    # Prefer samples that are extreme in at least one view and not central in the others.
    for scale in (1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4):
        k = min(sample_radius.shape[0], max(min_samples, int(round(outlier_count * scale))))
        sample_mask = _rank_mask(sample_radius, k, smallest=False)
        text_mask = _rank_mask(text_radius, k, smallest=False)
        visual_mask = _rank_mask(visual_radius, k, smallest=False)
        votes = sample_mask.astype(np.int32) + text_mask.astype(np.int32) + visual_mask.astype(np.int32)
        mask = votes >= 2
        if int(mask.sum()) >= min_samples:
            return mask
    sample_mask = _rank_mask(sample_radius, outlier_count, smallest=False)
    text_mask = _rank_mask(text_radius, outlier_count, smallest=False)
    visual_mask = _rank_mask(visual_radius, outlier_count, smallest=False)
    return sample_mask | text_mask | visual_mask


def _subset_payload(
    payload: Dict[str, torch.Tensor],
    indices: np.ndarray,
    subset_name: str,
    base_stem: str,
) -> Dict[str, object]:
    idx_tensor = torch.as_tensor(indices, dtype=torch.long)
    out: Dict[str, object] = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.shape[:1] == (payload[_hidden_key(payload)].shape[0],):
            out[key] = value.index_select(0, idx_tensor)
        elif isinstance(value, list) and len(value) == payload[_hidden_key(payload)].shape[0]:
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


def _plot_core_outlier_tsne(
    out_path: str,
    text_tsne: np.ndarray,
    visual_tsne: np.ndarray,
    core_mask: np.ndarray,
    outlier_mask: np.ndarray,
    title: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)
    panel_specs = [
        (axes[0], core_mask, "Core Hidden"),
        (axes[1], outlier_mask, "Outlier Hidden"),
    ]
    for ax, selected_mask, subtitle in panel_specs:
        ax.scatter(
            text_tsne[:, 0],
            text_tsne[:, 1],
            s=18,
            c=COLOR_TEXT,
            alpha=0.12,
            linewidths=0.0,
            label="Teacher text (all)",
            zorder=1,
        )
        ax.scatter(
            visual_tsne[:, 0],
            visual_tsne[:, 1],
            s=18,
            c=COLOR_VISUAL,
            alpha=0.12,
            linewidths=0.0,
            label="Teacher visual (all)",
            zorder=1,
        )
        ax.scatter(
            text_tsne[selected_mask, 0],
            text_tsne[selected_mask, 1],
            s=34,
            c=COLOR_TEXT,
            alpha=0.92,
            edgecolors="#6d5800",
            linewidths=0.35,
            label=f"{subtitle} text",
            zorder=2,
        )
        ax.scatter(
            visual_tsne[selected_mask, 0],
            visual_tsne[selected_mask, 1],
            s=34,
            c=COLOR_VISUAL,
            alpha=0.92,
            edgecolors="#1d5377",
            linewidths=0.35,
            label=f"{subtitle} visual",
            zorder=2,
        )
        ax.set_title(subtitle, fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=9, frameon=True)

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
    sample_std, sample_mean, sample_scale = _standardize_fit(sample_features)
    sample_pca_dim = min(args.pca_dim, sample_std.shape[0], sample_std.shape[1])
    sample_pca = PCA(n_components=sample_pca_dim, random_state=args.random_state)
    sample_pca_feat = sample_pca.fit_transform(sample_std)
    sample_center = sample_pca_feat.mean(axis=0, keepdims=True)
    sample_radius = np.linalg.norm(sample_pca_feat - sample_center, axis=1)

    text_features, visual_features = _modality_centroids(
        teacher_payload, min_modality_tokens=args.min_modality_tokens
    )
    text_std, _, _ = _standardize_fit(text_features)
    visual_std, _, _ = _standardize_fit(visual_features)
    text_pca_dim = min(args.pca_dim, text_std.shape[0], text_std.shape[1])
    visual_pca_dim = min(args.pca_dim, visual_std.shape[0], visual_std.shape[1])
    text_pca = PCA(n_components=text_pca_dim, random_state=args.random_state).fit_transform(text_std)
    visual_pca = PCA(n_components=visual_pca_dim, random_state=args.random_state).fit_transform(visual_std)
    text_center = text_pca.mean(axis=0, keepdims=True)
    visual_center = visual_pca.mean(axis=0, keepdims=True)
    text_radius = np.linalg.norm(text_pca - text_center, axis=1)
    visual_radius = np.linalg.norm(visual_pca - visual_center, axis=1)
    modality_radius = np.maximum(text_radius, visual_radius)

    core_count = _counts_from_ratio(n_samples, args.sample_core_ratio, args.min_samples)
    outlier_count = _counts_from_ratio(n_samples, args.sample_outlier_ratio, args.min_samples)
    modality_core_count = _counts_from_ratio(n_samples, args.modality_core_ratio, args.min_samples)
    modality_outlier_count = _counts_from_ratio(n_samples, args.modality_outlier_ratio, args.min_samples)

    target_core = max(core_count, modality_core_count)
    target_outlier = max(outlier_count, modality_outlier_count)
    core_mask = _adaptive_core_mask(
        sample_radius=sample_radius,
        text_radius=text_radius,
        visual_radius=visual_radius,
        core_count=target_core,
        min_samples=args.min_samples,
    )
    outlier_mask = _adaptive_outlier_mask(
        sample_radius=sample_radius,
        text_radius=text_radius,
        visual_radius=visual_radius,
        outlier_count=target_outlier,
        min_samples=args.min_samples,
    )

    if int(core_mask.sum()) < args.min_samples or int(outlier_mask.sum()) < args.min_samples:
        raise RuntimeError(
            f"Subset size too small: core={int(core_mask.sum())}, outlier={int(outlier_mask.sum())}, "
            f"min_samples={args.min_samples}"
        )

    overlap_mask = core_mask & outlier_mask
    core_indices = np.flatnonzero(core_mask)
    outlier_indices = np.flatnonzero(outlier_mask)
    overlap_indices = np.flatnonzero(overlap_mask)

    stem = _derive_stem(args.teacher_path, args.output_name)
    core_pt_path = os.path.join(args.output_dir, f"{stem}_core-hidden.pt")
    outlier_pt_path = os.path.join(args.output_dir, f"{stem}_outlier-hidden.pt")
    plot_path = os.path.join(args.output_dir, f"{stem}_core_outlier_tsne.png")
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.json")

    core_payload = _subset_payload(teacher_payload, core_indices, "core-hidden", stem)
    outlier_payload = _subset_payload(teacher_payload, outlier_indices, "outlier-hidden", stem)
    # 暂时无需存pt，先画图看看
    # torch.save(core_payload, core_pt_path)
    # torch.save(outlier_payload, outlier_pt_path)

    # text / visual must share one embedding space for overlay; independent t-SNE runs
    # produce unrelated coordinate systems and visually mislead the split.
    modality_all = np.concatenate([text_pca, visual_pca], axis=0)
    shared_perplexity = min(args.perplexity, max(5.0, (modality_all.shape[0] - 1) / 3.0))
    modality_tsne = TSNE(
        n_components=2,
        perplexity=shared_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    ).fit_transform(modality_all)
    text_tsne = modality_tsne[:n_samples]
    visual_tsne = modality_tsne[n_samples:]

    title = (
        "A3 Teacher Split by Calibration Geometry\n"
        "Core keeps dense manifold centers; Outlier keeps diffuse or modality-deviant samples"
    )
    _plot_core_outlier_tsne(
        out_path=plot_path,
        text_tsne=text_tsne,
        visual_tsne=visual_tsne,
        core_mask=core_mask,
        outlier_mask=outlier_mask,
        title=title,
        dpi=args.dpi,
    )

    summary = {
        "teacher_path": args.teacher_path,
        "num_teacher_samples": int(n_samples),
        "core_num_samples": int(core_mask.sum()),
        "outlier_num_samples": int(outlier_mask.sum()),
        "overlap_num_samples": int(overlap_mask.sum()),
        "core_overlap_ratio": float(overlap_mask.sum() / max(core_mask.sum(), 1)),
        "outlier_overlap_ratio": float(overlap_mask.sum() / max(outlier_mask.sum(), 1)),
        "sample_core_count": int(core_count),
        "sample_outlier_count": int(outlier_count),
        "modality_core_count": int(modality_core_count),
        "modality_outlier_count": int(modality_outlier_count),
        "sample_radius_p50": float(np.percentile(sample_radius, 50)),
        "sample_radius_p90": float(np.percentile(sample_radius, 90)),
        "text_radius_p50": float(np.percentile(text_radius, 50)),
        "text_radius_p90": float(np.percentile(text_radius, 90)),
        "visual_radius_p50": float(np.percentile(visual_radius, 50)),
        "visual_radius_p90": float(np.percentile(visual_radius, 90)),
        "modality_radius_p50": float(np.percentile(modality_radius, 50)),
        "modality_radius_p90": float(np.percentile(modality_radius, 90)),
        "core_indices": core_indices.tolist(),
        "outlier_indices": outlier_indices.tolist(),
        "overlap_indices": overlap_indices.tolist(),
        "artifacts": {
            "core_hidden_pt": core_pt_path,
            "outlier_hidden_pt": outlier_pt_path,
            "tsne_plot": plot_path,
            "summary_json": summary_path,
        },
    }
    _save_json(summary_path, summary)

    print_saved_artifact_message(core_pt_path, "A3 core hidden subset")
    print_saved_artifact_message(outlier_pt_path, "A3 outlier hidden subset")
    print_saved_artifact_message(plot_path, "A3 core/outlier t-SNE plot")
    print_saved_artifact_message(summary_path, "A3 split summary JSON")


if __name__ == "__main__":
    main()
