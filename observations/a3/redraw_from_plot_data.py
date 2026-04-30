#!/usr/bin/env python3
"""Redraw A3 figures from lightweight plot_data.pt without recomputing t-SNE."""

import argparse
import os
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_maes")

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw A3 plots directly from plot_data.pt."
    )
    parser.add_argument("--plot_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--bg_alpha", type=float, default=0.08)
    parser.add_argument("--fg_alpha", type=float, default=0.95)
    parser.add_argument("--bg_size", type=float, default=11.0)
    parser.add_argument("--fg_size", type=float, default=34.0)
    parser.add_argument("--core-count", type=int, default=None)
    parser.add_argument("--outlier-count", type=int, default=None)
    return parser.parse_args()


def _to_numpy(payload: Dict, key: str):
    return payload[key].detach().cpu().numpy()


def _mask_from_desc_ranking(ranking, count: int, n: int):
    mask = [False] * n
    for idx in ranking[:count]:
        mask[int(idx)] = True
    return mask


def _draw_joint(payload: Dict, output: str, title: str, dpi: int, bg_alpha: float, fg_alpha: float, bg_size: float, fg_size: float, core_count: int | None, outlier_count: int | None) -> None:
    text_tsne = _to_numpy(payload, "text_tsne")
    visual_tsne = _to_numpy(payload, "visual_tsne")
    defaults = payload.get("defaults", {})
    ranking = payload["ranking"]
    n = text_tsne.shape[0]
    core_count = int(core_count if core_count is not None else defaults.get("core_count", min(100, n)))
    outlier_count = int(outlier_count if outlier_count is not None else defaults.get("outlier_count", min(100, n)))
    core_mask = _mask_from_desc_ranking(_to_numpy(ranking, "core_desc"), core_count, n)
    outlier_mask = _mask_from_desc_ranking(_to_numpy(ranking, "outlier_desc"), outlier_count, n)
    color_text = payload["style"]["color_text"]
    color_visual = payload["style"]["color_visual"]

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4), constrained_layout=True)
    for ax, mask, subtitle in (
        (axes[0], core_mask, "Core Hidden"),
        (axes[1], outlier_mask, "Outlier Hidden"),
    ):
        ax.scatter(text_tsne[:, 0], text_tsne[:, 1], s=bg_size, c=color_text, alpha=bg_alpha, linewidths=0.0, label="Teacher text (all)", zorder=1)
        ax.scatter(visual_tsne[:, 0], visual_tsne[:, 1], s=bg_size, c=color_visual, alpha=bg_alpha, linewidths=0.0, label="Teacher visual (all)", zorder=1)
        ax.scatter(text_tsne[mask, 0], text_tsne[mask, 1], s=fg_size, c=color_text, alpha=fg_alpha, edgecolors="#7b6508", linewidths=0.35, label=f"{subtitle} text", zorder=3)
        ax.scatter(visual_tsne[mask, 0], visual_tsne[mask, 1], s=fg_size, c=color_visual, alpha=fg_alpha, edgecolors="#1f587d", linewidths=0.35, label=f"{subtitle} visual", zorder=3)
        ax.set_title(subtitle, fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        title or "A3 Redraw from plot_data.pt\nShared t-SNE geometry reused without recomputation",
        fontsize=15,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _draw_per_modality(payload: Dict, output: str, title: str, dpi: int, bg_alpha: float, fg_alpha: float, bg_size: float, fg_size: float, core_count: int | None, outlier_count: int | None) -> None:
    text_tsne = _to_numpy(payload, "text_tsne")
    visual_tsne = _to_numpy(payload, "visual_tsne")
    defaults = payload.get("defaults", {})
    ranking = payload["ranking"]
    n = text_tsne.shape[0]
    core_count = int(core_count if core_count is not None else defaults.get("core_count", min(100, n)))
    outlier_count = int(outlier_count if outlier_count is not None else defaults.get("outlier_count", min(100, n)))
    text_core_mask = _mask_from_desc_ranking(_to_numpy(ranking, "text_core_desc"), core_count, n)
    text_outlier_mask = _mask_from_desc_ranking(_to_numpy(ranking, "text_outlier_desc"), outlier_count, n)
    visual_core_mask = _mask_from_desc_ranking(_to_numpy(ranking, "visual_core_desc"), core_count, n)
    visual_outlier_mask = _mask_from_desc_ranking(_to_numpy(ranking, "visual_outlier_desc"), outlier_count, n)
    color_text = payload["style"]["color_text"]
    color_visual = payload["style"]["color_visual"]

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.6), constrained_layout=True)
    panel_specs = [
        (axes[0], "Core Hidden", text_core_mask, visual_core_mask),
        (axes[1], "Outlier Hidden", text_outlier_mask, visual_outlier_mask),
    ]
    for ax, subtitle, text_mask, visual_mask in panel_specs:
        ax.scatter(text_tsne[:, 0], text_tsne[:, 1], s=bg_size, c=color_text, alpha=bg_alpha, linewidths=0.0, label="Teacher text (all)", zorder=1)
        ax.scatter(visual_tsne[:, 0], visual_tsne[:, 1], s=bg_size, c=color_visual, alpha=bg_alpha, linewidths=0.0, label="Teacher visual (all)", zorder=1)
        ax.scatter(text_tsne[text_mask, 0], text_tsne[text_mask, 1], s=fg_size, c=color_text, alpha=fg_alpha, edgecolors="#7b6508", linewidths=0.35, label="Text", zorder=3)
        ax.scatter(visual_tsne[visual_mask, 0], visual_tsne[visual_mask, 1], s=fg_size, c=color_visual, alpha=fg_alpha, edgecolors="#1f587d", linewidths=0.35, label="Visual", zorder=3)
        ax.set_title(subtitle, fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        title or "A3 Redraw from plot_data.pt\nPer-modality selection reused without recomputing t-SNE",
        fontsize=15,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.plot_data, map_location="cpu")
    mode = payload["selection_mode"]
    if mode == "joint":
        _draw_joint(
            payload, args.output, args.title, args.dpi,
            args.bg_alpha, args.fg_alpha, args.bg_size, args.fg_size,
            args.core_count, args.outlier_count,
        )
    elif mode == "per_modality":
        _draw_per_modality(
            payload, args.output, args.title, args.dpi,
            args.bg_alpha, args.fg_alpha, args.bg_size, args.fg_size,
            args.core_count, args.outlier_count,
        )
    else:
        raise ValueError(f"Unsupported selection_mode in plot_data: {mode}")


if __name__ == "__main__":
    main()
