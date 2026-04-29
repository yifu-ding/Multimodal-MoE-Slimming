#!/usr/bin/env python3
"""A3: select core/outlier teacher subsets directly in shared t-SNE 2D space."""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_maes")

import matplotlib.pyplot as plt
import numpy as np
import torch
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


# DEFAULT_TEACHER_PATH = (
#     "/home/dyf/code/distill/MAES/storage/data_distill_kimi/"
#     "gqa-num_1024-token_2048-sample_at1.0-0429164415/teacher_hidden.pt"
# )

DEFAULT_TEACHER_PATH=("/home/dyf/code/distill/MAES/storage/data_distill_kimi/gqa-num_3072-token_2048-sample_at1.0-0429182412/teacher_hidden.pt")

DEFAULT_OUTPUT_DIR = "/home/dyf/code/distill/MAES/observations/a3/split-hidden"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a teacher hidden cache into core/outlier subsets by selecting "
            "samples directly in the shared t-SNE 2D space."
        )
    )
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--min_samples", type=int, default=1024)
    parser.add_argument("--core_ratio", type=float, default=0.20)
    parser.add_argument("--outlier_ratio", type=float, default=0.20)
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="joint",
        choices=("joint", "per_modality"),
        help=(
            "joint: one sample-level core/outlier set using both modalities together; "
            "per_modality: select text-core/text-outlier/visual-core/visual-outlier separately."
        ),
    )
    parser.add_argument(
        "--count_mode",
        type=str,
        default="fixed",
        choices=("fixed", "ratio"),
        help=(
            "fixed: use --min_samples as the exact number selected for each split; "
            "ratio: use ratios with --min_samples as a lower bound."
        ),
    )
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
    return f"a3_tsne2d_select_{teacher_tag}"


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = np.clip(features.std(axis=0, keepdims=True), 1e-6, None)
    return (features - mean) / std


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
            raise ValueError(f"Sample {idx} has too few text tokens.")
        if int(visual_mask.sum().item()) < min_modality_tokens:
            raise ValueError(f"Sample {idx} has too few visual tokens.")
        text_rows.append(hidden[idx, text_mask].mean(dim=0))
        visual_rows.append(hidden[idx, visual_mask].mean(dim=0))
    return torch.stack(text_rows, dim=0).numpy(), torch.stack(visual_rows, dim=0).numpy()


def _shared_tsne_embedding(
    text_features: np.ndarray,
    visual_features: np.ndarray,
    pca_dim: int,
    perplexity: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_features = np.concatenate([text_features, visual_features], axis=0)
    all_features = _standardize(all_features)
    pca_features = PCA(
        n_components=min(pca_dim, all_features.shape[0], all_features.shape[1]),
        random_state=random_state,
    ).fit_transform(all_features)
    tsne_perplexity = min(perplexity, max(5.0, (pca_features.shape[0] - 1) / 3.0))
    embed = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(pca_features)
    n = text_features.shape[0]
    return embed[:n], embed[n:]


def _kde_log_density(points: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(points.T)
    density = kde(points.T)
    return np.log(np.clip(density, 1e-12, None))


def _radius(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    return np.linalg.norm(points - center, axis=1)


def _counts(n: int, ratio: float, min_samples: int) -> int:
    return min(n, max(min_samples, int(round(n * ratio))))


def _resolve_count(n: int, ratio: float, min_samples: int, count_mode: str) -> int:
    if count_mode == "fixed":
        return min(n, int(min_samples))
    if count_mode == "ratio":
        return _counts(n, ratio, min_samples)
    raise ValueError(f"Unsupported count_mode: {count_mode}")


def _subset_payload(
    payload: Dict[str, torch.Tensor],
    indices: np.ndarray,
    subset_name: str,
    base_stem: str,
    extra_metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    n_samples = payload[_hidden_key(payload)].shape[0]
    idx_tensor = torch.as_tensor(indices, dtype=torch.long)
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
    metadata["selection_space"] = "shared_tsne_2d"
    if extra_metadata:
        metadata.update(extra_metadata)
    out["metadata"] = metadata
    out["num_samples"] = int(indices.shape[0])
    return out


def _select_masks(
    text_tsne: np.ndarray,
    visual_tsne: np.ndarray,
    core_count: int,
    outlier_count: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    text_log_kde = _kde_log_density(text_tsne)
    visual_log_kde = _kde_log_density(visual_tsne)
    text_r = _radius(text_tsne)
    visual_r = _radius(visual_tsne)

    # Core: both modalities dense and close to their local centers.
    core_score = (
        1.0 * text_log_kde
        + 1.0 * visual_log_kde
        - 0.35 * text_r
        - 0.35 * visual_r
    )
    # Outlier: at least one modality sparse/far. max() makes tail behavior dominate.
    outlier_score = np.maximum(-text_log_kde + 0.45 * text_r, -visual_log_kde + 0.45 * visual_r)

    core_idx = np.argsort(core_score)[-core_count:]
    outlier_idx = np.argsort(outlier_score)[-outlier_count:]

    core_mask = np.zeros(text_tsne.shape[0], dtype=bool)
    outlier_mask = np.zeros(text_tsne.shape[0], dtype=bool)
    core_mask[core_idx] = True
    outlier_mask[outlier_idx] = True
    metrics = {
        "text_log_kde": text_log_kde,
        "visual_log_kde": visual_log_kde,
        "text_radius": text_r,
        "visual_radius": visual_r,
        "core_score": core_score,
        "outlier_score": outlier_score,
    }
    return core_mask, outlier_mask, metrics


def _plot_panels(
    out_path: str,
    text_tsne: np.ndarray,
    visual_tsne: np.ndarray,
    core_mask: np.ndarray,
    outlier_mask: np.ndarray,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4), constrained_layout=True)
    for ax, mask, title in (
        (axes[0], core_mask, "Core Hidden"),
        (axes[1], outlier_mask, "Outlier Hidden"),
    ):
        ax.scatter(
            text_tsne[:, 0],
            text_tsne[:, 1],
            s=11,
            c=COLOR_TEXT,
            alpha=0.08,
            linewidths=0.0,
            label="Teacher text (all)",
            zorder=1,
        )
        ax.scatter(
            visual_tsne[:, 0],
            visual_tsne[:, 1],
            s=11,
            c=COLOR_VISUAL,
            alpha=0.08,
            linewidths=0.0,
            label="Teacher visual (all)",
            zorder=1,
        )
        ax.scatter(
            text_tsne[mask, 0],
            text_tsne[mask, 1],
            s=34,
            c=COLOR_TEXT,
            alpha=0.95,
            edgecolors="#7b6508",
            linewidths=0.35,
            label=f"{title} text",
            zorder=3,
        )
        ax.scatter(
            visual_tsne[mask, 0],
            visual_tsne[mask, 1],
            s=34,
            c=COLOR_VISUAL,
            alpha=0.95,
            edgecolors="#1f587d",
            linewidths=0.35,
            label=f"{title} visual",
            zorder=3,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        "A3 Teacher Split Directly in Shared t-SNE Space\n"
        "Selection and visualization now use the same 2D geometry, making core/outlier visually explicit",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_modality_panels(
    out_path: str,
    text_tsne: np.ndarray,
    visual_tsne: np.ndarray,
    text_core_mask: np.ndarray,
    text_outlier_mask: np.ndarray,
    visual_core_mask: np.ndarray,
    visual_outlier_mask: np.ndarray,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.6), constrained_layout=True)
    panel_specs = [
        (
            axes[0],
            "Core Hidden",
            [
                (text_tsne, text_core_mask, COLOR_TEXT, "#7b6508", "Teacher text (all)", "Text core"),
                (visual_tsne, visual_core_mask, COLOR_VISUAL, "#1f587d", "Teacher visual (all)", "Visual core"),
            ],
        ),
        (
            axes[1],
            "Outlier Hidden",
            [
                (text_tsne, text_outlier_mask, COLOR_TEXT, "#7b6508", "Teacher text (all)", "Text outlier"),
                (visual_tsne, visual_outlier_mask, COLOR_VISUAL, "#1f587d", "Teacher visual (all)", "Visual outlier"),
            ],
        ),
    ]
    for ax, title, items in panel_specs:
        for points, mask, color, edge, bg_label, fg_label in items:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=11,
                c=color,
                alpha=0.08,
                linewidths=0.0,
                label=bg_label,
                zorder=1,
            )
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=34,
                c=color,
                alpha=0.95,
                edgecolors=edge,
                linewidths=0.35,
                label=fg_label,
                zorder=3,
            )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.suptitle(
        "A3 Teacher Split in Shared t-SNE Space by Modality\n"
        "Text and visual now select their own core/outlier samples directly in 2D, then are overlaid by split type",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_plot_data(path: str, payload: Dict[str, object]) -> None:
    torch.save(payload, path)


def _save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    payload = _load_payload(args.teacher_path)
    payload["__source_path__"] = args.teacher_path
    n_samples = payload[_hidden_key(payload)].shape[0]

    text_features, visual_features = _modality_centroids(
        payload, min_modality_tokens=args.min_modality_tokens
    )
    text_tsne, visual_tsne = _shared_tsne_embedding(
        text_features=text_features,
        visual_features=visual_features,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        random_state=args.random_state,
    )

    stem = _derive_stem(args.teacher_path, args.output_name)
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.json")
    plot_data_path = os.path.join(args.output_dir, f"{stem}_plot_data.pt")

    core_count = _resolve_count(n_samples, args.core_ratio, args.min_samples, args.count_mode)
    outlier_count = _resolve_count(n_samples, args.outlier_ratio, args.min_samples, args.count_mode)

    if args.selection_mode == "joint":
        core_mask, outlier_mask, metrics = _select_masks(
            text_tsne=text_tsne,
            visual_tsne=visual_tsne,
            core_count=core_count,
            outlier_count=outlier_count,
        )

        core_indices = np.flatnonzero(core_mask)
        outlier_indices = np.flatnonzero(outlier_mask)
        overlap_indices = np.flatnonzero(core_mask & outlier_mask)

        core_pt_path = os.path.join(args.output_dir, f"{stem}_core-hidden.pt")
        outlier_pt_path = os.path.join(args.output_dir, f"{stem}_outlier-hidden.pt")
        plot_path = os.path.join(args.output_dir, f"{stem}_core_outlier_tsne.png")

        torch.save(_subset_payload(payload, core_indices, "core-hidden", stem), core_pt_path)
        torch.save(_subset_payload(payload, outlier_indices, "outlier-hidden", stem), outlier_pt_path)
        _plot_panels(plot_path, text_tsne, visual_tsne, core_mask, outlier_mask, args.dpi)

        summary = {
            "teacher_path": args.teacher_path,
            "selection_space": "shared_tsne_2d",
            "selection_mode": "joint",
            "count_mode": args.count_mode,
            "num_teacher_samples": int(n_samples),
            "core_num_samples": int(core_mask.sum()),
            "outlier_num_samples": int(outlier_mask.sum()),
            "overlap_num_samples": int(overlap_indices.shape[0]),
            "core_ratio": args.core_ratio,
            "outlier_ratio": args.outlier_ratio,
            "min_samples": args.min_samples,
            "core_indices": core_indices.tolist(),
            "outlier_indices": outlier_indices.tolist(),
            "overlap_indices": overlap_indices.tolist(),
            "metric_summary": {
                "text_log_kde_p10": float(np.percentile(metrics["text_log_kde"], 10)),
                "text_log_kde_p90": float(np.percentile(metrics["text_log_kde"], 90)),
                "visual_log_kde_p10": float(np.percentile(metrics["visual_log_kde"], 10)),
                "visual_log_kde_p90": float(np.percentile(metrics["visual_log_kde"], 90)),
                "text_radius_p10": float(np.percentile(metrics["text_radius"], 10)),
                "text_radius_p90": float(np.percentile(metrics["text_radius"], 90)),
                "visual_radius_p10": float(np.percentile(metrics["visual_radius"], 10)),
                "visual_radius_p90": float(np.percentile(metrics["visual_radius"], 90)),
            },
            "artifacts": {
                "core_hidden_pt": core_pt_path,
                "outlier_hidden_pt": outlier_pt_path,
                "tsne_plot": plot_path,
                "plot_data_pt": plot_data_path,
                "summary_json": summary_path,
            },
        }
        _save_plot_data(
            plot_data_path,
            {
                "selection_space": "shared_tsne_2d",
                "selection_mode": "joint",
                "teacher_path": args.teacher_path,
                "text_tsne": torch.from_numpy(text_tsne).to(torch.float32),
                "visual_tsne": torch.from_numpy(visual_tsne).to(torch.float32),
                "core_mask": torch.from_numpy(core_mask),
                "outlier_mask": torch.from_numpy(outlier_mask),
                "core_indices": torch.from_numpy(core_indices),
                "outlier_indices": torch.from_numpy(outlier_indices),
                "metrics": {k: torch.from_numpy(v).to(torch.float32) for k, v in metrics.items()},
                "style": {"color_text": COLOR_TEXT, "color_visual": COLOR_VISUAL},
            },
        )
        print_saved_artifact_message(core_pt_path, "A3 shared-tSNE core hidden subset")
        print_saved_artifact_message(outlier_pt_path, "A3 shared-tSNE outlier hidden subset")
        print_saved_artifact_message(plot_path, "A3 shared-tSNE core/outlier plot")
        print_saved_artifact_message(plot_data_path, "A3 shared-tSNE plot data PT")
    else:
        text_metrics = {
            "text_log_kde": _kde_log_density(text_tsne),
            "text_radius": _radius(text_tsne),
        }
        visual_metrics = {
            "visual_log_kde": _kde_log_density(visual_tsne),
            "visual_radius": _radius(visual_tsne),
        }
        text_core_score = text_metrics["text_log_kde"] - 0.35 * text_metrics["text_radius"]
        text_outlier_score = -text_metrics["text_log_kde"] + 0.45 * text_metrics["text_radius"]
        visual_core_score = visual_metrics["visual_log_kde"] - 0.35 * visual_metrics["visual_radius"]
        visual_outlier_score = -visual_metrics["visual_log_kde"] + 0.45 * visual_metrics["visual_radius"]

        text_core_idx = np.argsort(text_core_score)[-core_count:]
        text_outlier_idx = np.argsort(text_outlier_score)[-outlier_count:]
        visual_core_idx = np.argsort(visual_core_score)[-core_count:]
        visual_outlier_idx = np.argsort(visual_outlier_score)[-outlier_count:]

        text_core_mask = np.zeros(n_samples, dtype=bool)
        text_outlier_mask = np.zeros(n_samples, dtype=bool)
        visual_core_mask = np.zeros(n_samples, dtype=bool)
        visual_outlier_mask = np.zeros(n_samples, dtype=bool)
        text_core_mask[text_core_idx] = True
        text_outlier_mask[text_outlier_idx] = True
        visual_core_mask[visual_core_idx] = True
        visual_outlier_mask[visual_outlier_idx] = True

        core_pt = os.path.join(args.output_dir, f"{stem}_core-hidden.pt")
        outlier_pt = os.path.join(args.output_dir, f"{stem}_outlier-hidden.pt")
        plot_path = os.path.join(args.output_dir, f"{stem}_per_modality_tsne.png")
        text_core_indices = np.flatnonzero(text_core_mask)
        text_outlier_indices = np.flatnonzero(text_outlier_mask)
        visual_core_indices = np.flatnonzero(visual_core_mask)
        visual_outlier_indices = np.flatnonzero(visual_outlier_mask)
        merged_core_indices = np.unique(np.concatenate([text_core_indices, visual_core_indices]))
        merged_outlier_indices = np.unique(np.concatenate([text_outlier_indices, visual_outlier_indices]))

        torch.save(
            _subset_payload(
                payload,
                merged_core_indices,
                "core-hidden",
                stem,
                extra_metadata={
                    "selection_mode": "per_modality",
                    "count_mode": args.count_mode,
                    "text_core_indices": text_core_indices.tolist(),
                    "visual_core_indices": visual_core_indices.tolist(),
                    "merged_core_indices": merged_core_indices.tolist(),
                },
            ),
            core_pt,
        )
        torch.save(
            _subset_payload(
                payload,
                merged_outlier_indices,
                "outlier-hidden",
                stem,
                extra_metadata={
                    "selection_mode": "per_modality",
                    "count_mode": args.count_mode,
                    "text_outlier_indices": text_outlier_indices.tolist(),
                    "visual_outlier_indices": visual_outlier_indices.tolist(),
                    "merged_outlier_indices": merged_outlier_indices.tolist(),
                },
            ),
            outlier_pt,
        )

        _plot_modality_panels(
            plot_path,
            text_tsne,
            visual_tsne,
            text_core_mask,
            text_outlier_mask,
            visual_core_mask,
            visual_outlier_mask,
            args.dpi,
        )

        summary = {
            "teacher_path": args.teacher_path,
            "selection_space": "shared_tsne_2d",
            "selection_mode": "per_modality",
            "count_mode": args.count_mode,
            "num_teacher_samples": int(n_samples),
            "core_ratio": args.core_ratio,
            "outlier_ratio": args.outlier_ratio,
            "min_samples": args.min_samples,
            "text_core_num_samples": int(text_core_mask.sum()),
            "text_outlier_num_samples": int(text_outlier_mask.sum()),
            "visual_core_num_samples": int(visual_core_mask.sum()),
            "visual_outlier_num_samples": int(visual_outlier_mask.sum()),
            "merged_core_num_samples": int(merged_core_indices.shape[0]),
            "merged_outlier_num_samples": int(merged_outlier_indices.shape[0]),
            "text_core_indices": text_core_indices.tolist(),
            "text_outlier_indices": text_outlier_indices.tolist(),
            "visual_core_indices": visual_core_indices.tolist(),
            "visual_outlier_indices": visual_outlier_indices.tolist(),
            "merged_core_indices": merged_core_indices.tolist(),
            "merged_outlier_indices": merged_outlier_indices.tolist(),
            "artifacts": {
                "core_hidden_pt": core_pt,
                "outlier_hidden_pt": outlier_pt,
                "tsne_plot": plot_path,
                "plot_data_pt": plot_data_path,
                "summary_json": summary_path,
            },
        }
        _save_plot_data(
            plot_data_path,
            {
                "selection_space": "shared_tsne_2d",
                "selection_mode": "per_modality",
                "teacher_path": args.teacher_path,
                "text_tsne": torch.from_numpy(text_tsne).to(torch.float32),
                "visual_tsne": torch.from_numpy(visual_tsne).to(torch.float32),
                "text_core_mask": torch.from_numpy(text_core_mask),
                "text_outlier_mask": torch.from_numpy(text_outlier_mask),
                "visual_core_mask": torch.from_numpy(visual_core_mask),
                "visual_outlier_mask": torch.from_numpy(visual_outlier_mask),
                "text_core_indices": torch.from_numpy(np.flatnonzero(text_core_mask)),
                "text_outlier_indices": torch.from_numpy(np.flatnonzero(text_outlier_mask)),
                "visual_core_indices": torch.from_numpy(np.flatnonzero(visual_core_mask)),
                "visual_outlier_indices": torch.from_numpy(np.flatnonzero(visual_outlier_mask)),
                "metrics": {
                    "text_log_kde": torch.from_numpy(text_metrics["text_log_kde"]).to(torch.float32),
                    "text_radius": torch.from_numpy(text_metrics["text_radius"]).to(torch.float32),
                    "visual_log_kde": torch.from_numpy(visual_metrics["visual_log_kde"]).to(torch.float32),
                    "visual_radius": torch.from_numpy(visual_metrics["visual_radius"]).to(torch.float32),
                    "text_core_score": torch.from_numpy(text_core_score).to(torch.float32),
                    "text_outlier_score": torch.from_numpy(text_outlier_score).to(torch.float32),
                    "visual_core_score": torch.from_numpy(visual_core_score).to(torch.float32),
                    "visual_outlier_score": torch.from_numpy(visual_outlier_score).to(torch.float32),
                },
                "style": {"color_text": COLOR_TEXT, "color_visual": COLOR_VISUAL},
            },
        )
        print_saved_artifact_message(core_pt, "A3 merged core hidden subset")
        print_saved_artifact_message(outlier_pt, "A3 merged outlier hidden subset")
        print_saved_artifact_message(plot_path, "A3 per-modality shared-tSNE plot")
        print_saved_artifact_message(plot_data_path, "A3 per-modality plot data PT")

    _save_json(summary_path, summary)
    print_saved_artifact_message(summary_path, "A3 shared-tSNE summary JSON")


if __name__ == "__main__":
    main()
