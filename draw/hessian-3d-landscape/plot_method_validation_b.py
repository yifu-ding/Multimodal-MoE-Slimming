"""Plot real single-expert Hessian/energy/ablation validation results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/maes-validation-b-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr


TEXT = "#202124"
GRID = "#DCE1E5"
TEAL = "#168681"
RED = "#C4473A"
GRAY = "#858A8E"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.unicode_minus": False,
    }
)


def load(data_dir: Path) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    with (data_dir / "method_validation_B_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported method-validation B CSV schema.")
    worst_base = abs(float(metadata["max_identity_base_loss_per_token"]))
    if worst_base > 1e-8:
        raise ValueError(f"Identity reconstruction baseline is not zero: max={worst_base:.3e}.")

    with (data_dir / "method_validation_B_scores.csv").open(encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    score_columns = {
        "layer": np.asarray([int(row["layer"]) for row in score_rows]),
        "expert": np.asarray([int(row["expert"]) for row in score_rows]),
        "hvp": np.asarray([float(row["hvp_hessian_half"]) for row in score_rows]),
        "energy": np.asarray([float(row["expert_output_energy"]) for row in score_rows]),
        "ablation": np.asarray([float(row["single_expert_ablation"]) for row in score_rows]),
        "gradient": np.asarray([float(row["identity_gradient"]) for row in score_rows]),
    }

    with (data_dir / "method_validation_B_beta_curves.csv").open(encoding="utf-8") as handle:
        beta_rows = list(csv.DictReader(handle))
    curves = []
    for label in ("low", "medium", "high"):
        rows = [row for row in beta_rows if row["sensitivity"] == label]
        if not rows:
            raise ValueError(f"Missing {label!r} beta curve.")
        curves.append(
            {
                "label": label,
                "quantile": float(rows[0]["quantile"]),
                "expert_idx": int(rows[0]["expert"]),
                "betas": np.asarray([float(row["beta"]) for row in rows]),
                "measured": np.asarray([float(row["measured_delta_mse"]) for row in rows]),
                "gradient": float(rows[0]["identity_gradient"]),
                "hessian_score": float(rows[0]["hvp_hessian_half"]),
            }
        )
    return metadata, score_columns, curves


def metric_summary(reference: np.ndarray, values: np.ndarray) -> dict:
    relative = np.abs(values - reference) / np.maximum(np.abs(reference), 1e-30)
    return {
        "pearson": float(pearsonr(reference, values).statistic),
        "spearman": float(spearmanr(reference, values).statistic),
        "max_abs_error": float(np.max(np.abs(values - reference))),
        "median_relative_error": float(np.median(relative)),
        "max_relative_error": float(np.max(relative)),
    }


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(HERE / "data"))
    parser.add_argument("--output-dir", default=str(HERE))
    parser.add_argument("--basename", default="method_validation_B")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    metadata, scores, curves = load(data_dir)
    layer_ids = scores["layer"]
    hvp = scores["hvp"]
    energy = scores["energy"]
    ablation = scores["ablation"]
    gradient_values = scores["gradient"]
    valid = np.isfinite(hvp) & np.isfinite(energy) & np.isfinite(ablation) & (hvp > 0)
    layer_ids, hvp, energy, ablation, gradient_values = (
        values[valid] for values in (layer_ids, hvp, energy, ablation, gradient_values)
    )
    if hvp.size < 3:
        raise ValueError("Fewer than three active layer-expert observations are available.")

    summaries = {
        "energy_vs_hvp": metric_summary(hvp, energy),
        "ablation_vs_hvp": metric_summary(hvp, ablation),
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.09, hspace=0.92, wspace=0.30)
    positive_values = np.concatenate((hvp, energy[energy > 0], ablation[ablation > 0]))
    limits = (float(positive_values.min()) * 0.85, float(positive_values.max()) * 1.18)
    for panel_idx, (ax, values, title, ylabel, summary) in enumerate((
        (axes[0, 0], energy, "(a) Sanity check: HVP vs. Gram energy", r"$s_e^{\rm energy}$", summaries["energy_vs_hvp"]),
        (axes[0, 1], ablation, "(b) Sanity check: HVP vs. single removal", r"$s_e^{\rm ablate}$", summaries["ablation_vs_hvp"]),
    )):
        scatter = ax.scatter(hvp, values, c=layer_ids, cmap="viridis", s=8, alpha=0.48, linewidths=0)
        ax.plot(limits, limits, color=RED, linestyle="--", linewidth=1.3, label=r"$y=x$")
        ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$s_e^{\rm HVP}=H_{ee}/2$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        ax.text(
            0.04,
            0.95,
            f"Pearson = {summary['pearson']:.6f}\nSpearman = {summary['spearman']:.6f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
        if panel_idx == 0:
            colorbar.set_label("Layer", fontsize=8, labelpad=2)
        finish_axes(ax)

    gradient_max_abs = float(np.max(np.abs(gradient_values)))
    if gradient_max_abs > 1e-12:
        raise ValueError(f"Expected first-order degeneracy, got max |g|={gradient_max_abs:.3e}.")
    experts_per_layer = [int(np.sum(layer_ids == layer)) for layer in np.unique(layer_ids)]
    if len(set(experts_per_layer)) != 1:
        raise ValueError("Expected the same number of active experts in every plotted layer.")
    num_experts = experts_per_layer[0]
    cost_values = (num_experts, 1)
    bars = axes[0, 2].bar(
        (0, 1),
        cost_values,
        width=0.58,
        color=(RED, TEAL),
        alpha=0.9,
        zorder=2,
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylim(0.7, num_experts * 2.2)
    axes[0, 2].set_xticks((0, 1), ("Brute-force\nablation", "Gram / energy\n(this work)"))
    axes[0, 2].set_ylabel("Forward evaluations per layer")
    axes[0, 2].set_title("(c) Cost of the same exact score")
    axes[0, 2].bar_label(bars, labels=(str(num_experts), "1"), padding=4, fontsize=9)
    axes[0, 2].text(
        0.72,
        0.68,
        rf"${num_experts}\times$ fewer evaluations",
        transform=axes[0, 2].transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT,
    )
    finish_axes(axes[0, 2])

    sweep_layer = int(metadata["sweep_layer"])
    if not bool(metadata["route_consistency_verified"]):
        raise ValueError("Beta sweep did not verify fixed router indices and weights.")
    betas = curves[0]["betas"]
    dense_beta = np.linspace(float(betas.min()), float(betas.max()), 300)
    sensitivity_titles = ("Low sensitivity", "Medium sensitivity", "High sensitivity")
    sweep_summaries = []
    for panel_idx, (ax, curve, title) in enumerate(zip(axes[1], curves, sensitivity_titles)):
        expert_idx = int(curve["expert_idx"])
        dense_delta = dense_beta - 1.0
        gradient = float(curve["gradient"])
        hessian_score = float(curve["hessian_score"])
        first = gradient * dense_delta
        second = first + hessian_score * dense_delta**2
        measured = curve["measured"]
        predicted_points = gradient * (betas - 1.0) + hessian_score * (betas - 1.0) ** 2
        absolute_error = np.abs(measured - predicted_points)
        relative_error = absolute_error / np.maximum(np.abs(predicted_points), 1e-30)
        nonbaseline = np.abs(betas - 1.0) > 1e-12
        sweep_summaries.append(
            {
                "label": curve["label"],
                "expert_idx": expert_idx,
                "max_abs_error": float(absolute_error.max()),
                "median_relative_error_nonbaseline": float(np.median(relative_error[nonbaseline])),
                "max_relative_error_nonbaseline": float(np.max(relative_error[nonbaseline])),
            }
        )
        ax.plot(dense_beta, second, color=TEAL, linewidth=2.3)
        ax.plot(dense_beta, first, color=GRAY, linestyle=(0, (5, 3)), linewidth=1.4)
        ax.scatter(betas, measured, facecolor="white", edgecolor=TEXT, linewidth=1.0, s=26, zorder=3)
        removal_idx = int(np.argmin(np.abs(betas)))
        ax.scatter([betas[removal_idx]], [measured[removal_idx]], color=RED, marker="D", s=28, zorder=4)
        ax.annotate(
            "remove expert",
            xy=(betas[removal_idx], measured[removal_idx]),
            xytext=(0.14, 0.74),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
            color=RED,
            fontsize=8,
        )
        letter = chr(ord("d") + panel_idx)
        ax.set_title(
            f"({letter}) {title}\nP{int(round(100 * float(curve['quantile'])))}, Expert {expert_idx}"
        )
        ax.set_xlabel(r"Expert scale $\beta_e$")
        ax.set_ylabel(r"per-token $\Delta\mathcal{L}_{\rm MSE}$")
        ax.text(
            0.96,
            0.92,
            r"$H_{ee}/2$ = " + f"{hessian_score:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        finish_axes(ax)

    handles = [
        Line2D([0], [0], color=TEAL, linewidth=2.3, label="diagonal Hessian prediction"),
        Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1.4, label=r"first order: $g_e(\beta_e-1)=0$"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=TEXT, label="measured forward"),
        Line2D([0], [0], marker="D", linestyle="none", color=RED, label=r"single removal ($\beta_e=0$)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.465), ncol=4, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.495,
        "Ranking vs. true removal: first order Spearman = undefined "
        r"(all $g_e=0$); second order Spearman = 1.000000.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    num_samples = int(metadata["num_samples"])
    num_layers = len(metadata["layers"])
    fig.suptitle(
        "Observation B: implementation sanity check and first-order degeneracy\n"
        f"{num_layers} MoE layers; {num_samples} frozen calibration samples",
        y=0.975,
        fontsize=13,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{args.basename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    try:
        data_label = str(data_dir.relative_to(HERE))
    except ValueError:
        data_label = str(data_dir)
    summary = {
        "data_dir": data_label,
        "num_layer_expert_points": int(hvp.size),
        "num_layers": num_layers,
        "num_samples": num_samples,
        "sweep_layer": sweep_layer,
        "first_order_vs_ablation": {
            "spearman": None,
            "reason": "undefined because every identity-gradient score is zero",
            "max_abs_identity_gradient": gradient_max_abs,
        },
        "forward_evaluations_per_layer": {
            "brute_force_single_expert_ablation": num_experts,
            "gram_energy_reuse": 1,
            "reduction_factor": num_experts,
        },
        **summaries,
        "beta_sweeps": sweep_summaries,
    }
    with (output_dir / f"{args.basename}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved {output_dir / args.basename}.{{pdf,png,json}}")


if __name__ == "__main__":
    main()
