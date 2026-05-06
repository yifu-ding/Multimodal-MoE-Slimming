#!/usr/bin/env python3
"""A3: plot teacher hidden t-SNE by dataset source and modality."""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_maes")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from observations.common import ensure_dir, print_saved_artifact_message  # noqa: E402


DEFAULT_TEACHER_PATH = (
    "/home/dyf/code/distill/MAES/storage/data_distill_kimi/"
    "mixed-num_1024-token_2048-sample_at1.0-0421180651/teacher_hidden.pt"
)
DEFAULT_OUTPUT_DIR = "/home/dyf/code/distill/MAES/observations/a3/results"
DEFAULT_OUTPUT_NAME = "a3_teacher_only_tsne_by_source_1024"
SOURCE_ORDER = ("gqa", "coco", "m4-instruct")
SOURCE_COLORS = {
    "gqa": "#1cb3b0",
    "coco": "#fcd924",
    "m4-instruct": "#f1895e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot teacher hidden t-SNE for gqa/coco/m4-instruct with text and visual centroids."
    )
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--summary_path", type=str, default="")
    parser.add_argument("--min_modality_tokens", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--refresh_from_pt",
        action="store_true",
        help="Ignore cached plot data in summary json and rebuild from teacher pt.",
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
    payload["__source_path__"] = path
    return payload


def _hidden_key(payload: Dict[str, object]) -> str:
    if "teacher_cache" in payload:
        return "teacher_cache"
    raise KeyError("Expected `teacher_cache` in teacher payload.")


def _valid_mask(payload: Dict[str, object]) -> torch.Tensor:
    if "attention_mask" in payload:
        return payload["attention_mask"] > 0
    if "modality_labels" in payload:
        return payload["modality_labels"] >= 0
    raise KeyError("Cannot infer valid token mask from payload.")


def _build_modality_points(
    payload: Dict[str, object],
    min_modality_tokens: int,
) -> Tuple[np.ndarray, List[str], List[str]]:
    hidden = payload[_hidden_key(payload)]
    modality = payload["modality_labels"]
    valid = _valid_mask(payload)
    n_samples = hidden.shape[0]
    dataset_ids = payload.get("dataset_ids")
    metadata = payload.get("metadata", {})
    dataset_id_to_name = metadata.get("dataset_id_to_name", {})
    if dataset_ids is None:
        raise KeyError("Expected `dataset_ids` in payload for source labels.")
    if n_samples != int(dataset_ids.shape[0]):
        raise ValueError(f"dataset_ids length {dataset_ids.shape[0]} does not match samples {n_samples}.")

    features: List[np.ndarray] = []
    source_labels: List[str] = []
    modality_labels: List[str] = []
    for idx in range(n_samples):
        source_id = int(dataset_ids[idx].item())
        raw_source_name = dataset_id_to_name.get(source_id, str(source_id))
        source_name = raw_source_name.replace("_", "-")
        sample_mask = valid[idx]
        for mod_id, mod_name in ((1, "visual"), (0, "text")):
            mod_mask = sample_mask & (modality[idx] == mod_id)
            n_tokens = int(mod_mask.sum().item())
            if n_tokens < min_modality_tokens:
                raise ValueError(
                    f"Sample {idx} ({source_name}) has only {n_tokens} {mod_name} tokens."
                )
            centroid = hidden[idx, mod_mask].to(torch.float32).mean(dim=0).cpu().numpy()
            features.append(centroid)
            source_labels.append(source_name)
            modality_labels.append(mod_name)
    return np.stack(features, axis=0), source_labels, modality_labels


def _fit_tsne(
    features: np.ndarray,
    pca_dim: int,
    perplexity: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    embedding = tsne.fit_transform(pca_features)
    return embedding, pca.explained_variance_ratio_


def _plot(
    output_path: str,
    embedding: np.ndarray,
    source_labels: List[str],
    modality_labels: List[str],
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "serif"]
    fig, ax = plt.subplots(figsize=(8.1, 6.4), constrained_layout=True)
    source_arr = np.asarray(source_labels)
    modality_arr = np.asarray(modality_labels)

    for source_name in SOURCE_ORDER:
        for mod_name, marker, size, alpha in (
            ("visual", "o", 28, 0.5),
            ("text", "^", 34, 1.0),
        ):
            mask = (source_arr == source_name) & (modality_arr == mod_name)
            if not np.any(mask):
                continue
            points = embedding[mask]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=size,
                c=SOURCE_COLORS[source_name],
                marker=marker,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.55,
                label=f"{source_name} {mod_name} ({points.shape[0]})",
            )

    ax.set_title("Teacher Hidden t-SNE by Data Source and Modality", fontsize=18, pad=10)
    ax.set_xlabel("t-SNE 1", fontsize=16)
    ax.set_ylabel("t-SNE 2", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="lower right", frameon=True, ncol=1, fontsize=12)
    ax.grid(alpha=0.16, linewidth=0.6)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _load_cached_plot_data(summary_path: str) -> Tuple[np.ndarray, List[str], List[str], Dict[str, object]] | None:
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    embedding = summary.get("embedding")
    source_labels = summary.get("source_labels")
    modality_labels = summary.get("modality_labels")
    if embedding is None or source_labels is None or modality_labels is None:
        return None
    return np.asarray(embedding, dtype=np.float32), list(source_labels), list(modality_labels), summary


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    output_stem = args.output_name.strip() or DEFAULT_OUTPUT_NAME
    figure_path = os.path.join(args.output_dir, f"{output_stem}.png")
    summary_path = args.summary_path.strip() or os.path.join(args.output_dir, f"{output_stem}_summary.json")

    cached = None if args.refresh_from_pt else _load_cached_plot_data(summary_path)
    if cached is not None:
        embedding, source_labels, modality_labels, summary = cached
    else:
        payload = _load_payload(args.teacher_path)
        features, source_labels, modality_labels = _build_modality_points(
            payload,
            min_modality_tokens=args.min_modality_tokens,
        )
        embedding, explained_variance_ratio = _fit_tsne(
            features,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
        summary = {
            "teacher_path": args.teacher_path,
            "output_figure": figure_path,
            "num_points": int(embedding.shape[0]),
            "num_samples_per_source": 1024,
            "sources": list(SOURCE_ORDER),
            "modalities": ["text", "visual"],
            "pca_dim": int(min(args.pca_dim, features.shape[0], features.shape[1])),
            "perplexity": float(min(args.perplexity, max(5.0, (embedding.shape[0] - 1) / 3.0))),
            "explained_variance_ratio_sum": float(np.sum(explained_variance_ratio)),
            "embedding": embedding.tolist(),
            "source_labels": source_labels,
            "modality_labels": modality_labels,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot(figure_path, embedding, source_labels, modality_labels, dpi=args.dpi)

    print_saved_artifact_message(figure_path, "A3 teacher-only source/modality t-SNE")
    print_saved_artifact_message(summary_path, "A3 teacher-only source/modality t-SNE summary")


if __name__ == "__main__":
    main()
