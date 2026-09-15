"""Draw clearly watermarked layout previews for the MAES validation figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/maes-validation-preview-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


TEXT = "#202124"
GRID = "#DCE1E5"
TEAL = "#168681"
RED = "#C4473A"
GRAY = "#858A8E"
BLUE = "#356C9B"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="Layer Hessian probe used for scale only.")
    parser.add_argument("--output-dir", default=str(HERE))
    return parser.parse_args()


def load_probe(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    gradient = payload["gradient_per_token"].detach().cpu().double().numpy()
    score = 0.5 * payload["hessian_per_token"].diag().detach().cpu().double().numpy()
    if gradient.ndim != 1 or score.shape != gradient.shape:
        raise ValueError("Probe gradient and Hessian diagonal have incompatible shapes.")
    return gradient, score


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def add_preview_mark(fig: plt.Figure, label: str = "LAYOUT PREVIEW / SYNTHETIC PLACEHOLDERS") -> None:
    fig.text(
        0.992,
        0.992,
        label,
        ha="right",
        va="top",
        color=RED,
        fontsize=8,
        weight="bold",
    )


def save(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{basename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_signal_preview(
    output_dir: Path, gradient_l0: np.ndarray, score_l0: np.ndarray, rng: np.random.Generator
) -> None:
    num_layers = 30
    num_experts = score_l0.size

    # Synthetic layer variation is used only to preview the intended all-layer layout.
    expert_pattern = np.maximum(score_l0, np.quantile(score_l0[score_l0 > 0], 0.01))
    layer_scale = np.exp(rng.normal(0.0, 0.55, size=(num_layers, 1)))
    score = expert_pattern[None, :] * layer_scale * np.exp(
        rng.normal(0.0, 0.22, size=(num_layers, num_experts))
    )
    score[0] = np.maximum(score_l0, np.finfo(float).tiny)
    grad_floor = max(float(np.median(np.abs(gradient_l0))) * 1e-4, 1e-10)
    gradient = grad_floor * np.exp(rng.normal(0.0, 0.65, size=score.shape))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.4),
        gridspec_kw={"width_ratios": [1.35, 1.35, 1.0], "wspace": 0.34},
    )
    grad_log = np.log10(gradient)
    score_log = np.log10(score)

    im_g = axes[0].imshow(grad_log, aspect="auto", cmap="Blues", interpolation="nearest")
    cb_g = fig.colorbar(im_g, ax=axes[0], fraction=0.046, pad=0.025)
    cb_g.set_label(r"$\log_{10}|g_e|$")
    axes[0].set_title("(a) Near-zero first-order signal")
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("MoE layer")
    axes[0].text(
        0.02,
        0.03,
        "median = " + f"{np.median(gradient):.1e}" + "\nmax = " + f"{gradient.max():.1e}",
        transform=axes[0].transAxes,
        color="white",
        fontsize=8,
        va="bottom",
    )

    im_h = axes[1].imshow(score_log, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    cb_h = fig.colorbar(im_h, ax=axes[1], fraction=0.046, pad=0.025)
    cb_h.set_label(r"$\log_{10}(H_{ee}/2)$")
    axes[1].set_title("(b) Structured second-order signal")
    axes[1].set_xlabel("Expert ID")
    axes[1].set_ylabel("MoE layer")
    positive = score[score > 0]
    dynamic_range = np.quantile(positive, 0.99) / np.quantile(positive, 0.01)
    axes[1].text(
        0.02,
        0.03,
        "median = " + f"{np.median(positive):.1e}" + "\nP99/P1 = " + f"{dynamic_range:.0f}x",
        transform=axes[1].transAxes,
        color="white",
        fontsize=8,
        va="bottom",
    )

    bins = np.linspace(min(grad_log.min(), score_log.min()), max(grad_log.max(), score_log.max()), 42)
    axes[2].hist(
        grad_log.ravel(), bins=bins, density=True, histtype="step", linewidth=2.0,
        color=BLUE, label=r"$|g_e|$",
    )
    axes[2].hist(
        score_log.ravel(), bins=bins, density=True, histtype="step", linewidth=2.0,
        color=TEAL, label=r"$H_{ee}/2$",
    )
    axes[2].axvline(np.median(grad_log), color=BLUE, linestyle="--", linewidth=1.0)
    axes[2].axvline(np.median(score_log), color=TEAL, linestyle="--", linewidth=1.0)
    axes[2].set_title("(c) Separation in score magnitude")
    axes[2].set_xlabel(r"$\log_{10}$ per-token score")
    axes[2].set_ylabel("Density")
    axes[2].legend(frameon=False, loc="upper center")
    finish_axes(axes[2])

    fig.suptitle("Observation A: first-order degeneracy and second-order stiffness", y=1.02, fontsize=13)
    add_preview_mark(fig, "LEGACY / NOT USED")
    save(fig, output_dir, "method_validation_A_preview")


def draw_equivalence_preview(
    output_dir: Path, score_l0: np.ndarray, rng: np.random.Generator
) -> None:
    positive = score_l0[score_l0 > 0]
    lo, hi = np.quantile(positive, [0.02, 0.98])
    hvp = np.exp(rng.uniform(np.log(lo), np.log(hi), 30 * score_l0.size))
    energy = hvp * np.exp(rng.normal(0.0, 2.5e-3, hvp.size))
    ablate = hvp * np.exp(rng.normal(0.0, 4.5e-3, hvp.size))
    err_energy = np.abs(energy - hvp) / hvp
    err_ablate = np.abs(ablate - hvp) / hvp

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.09, hspace=0.92, wspace=0.30)

    limits = (min(hvp.min(), energy.min(), ablate.min()), max(hvp.max(), energy.max(), ablate.max()))
    for ax, values, title, color in (
        (axes[0, 0], energy, "(a) HVP vs. expert-output energy", TEAL),
        (axes[0, 1], ablate, "(b) HVP vs. single-expert ablation", BLUE),
    ):
        layer_ids = np.repeat(np.arange(30), score_l0.size)
        scatter = ax.scatter(hvp, values, c=layer_ids, cmap="viridis", s=7, alpha=0.45, linewidths=0)
        ax.plot(limits, limits, color=RED, linestyle="--", linewidth=1.3, label=r"$y=x$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$s_e^{\rm HVP}=H_{ee}/2$")
        ax.set_ylabel(r"$s_e^{\rm energy}$" if values is energy else r"$s_e^{\rm ablate}$")
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        ax.text(0.04, 0.95, r"Pearson $>0.99999$" + "\n" + r"Spearman $>0.99999$", transform=ax.transAxes, va="top", fontsize=8)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025, label="MoE layer")
        finish_axes(ax)

    for values, color, label in (
        (err_energy, TEAL, "Energy / HVP"),
        (err_ablate, BLUE, "Ablation / HVP"),
    ):
        x = np.sort(values)
        y = np.arange(1, x.size + 1) / x.size
        axes[0, 2].plot(x, y, color=color, linewidth=2.0, label=label)
    axes[0, 2].axvline(1e-4, color=GRAY, linestyle="--", linewidth=1.0, label=r"$10^{-4}$ target")
    axes[0, 2].axvline(1e-2, color=RED, linestyle=":", linewidth=1.2, label=r"$10^{-2}$ limit")
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_xlim(1e-5, 3e-2)
    axes[0, 2].set_ylim(0, 1.02)
    axes[0, 2].set_xlabel("Relative error")
    axes[0, 2].set_ylabel("Cumulative fraction")
    axes[0, 2].set_title("(c) Numerical agreement")
    axes[0, 2].legend(frameon=False, fontsize=8, loc="lower right")
    finish_axes(axes[0, 2])

    quantiles = (0.1, 0.5, 0.9)
    betas = np.asarray([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    dense_beta = np.linspace(betas.min(), betas.max(), 300)
    sensitivity_titles = ("Low sensitivity", "Medium sensitivity", "High sensitivity")
    expert_ids = (46, 33, 7)
    for panel_idx, (ax, quantile, title, expert_id) in enumerate(
        zip(axes[1], quantiles, sensitivity_titles, expert_ids)
    ):
        coefficient = float(score_l0[expert_id])
        theory = coefficient * (dense_beta - 1.0) ** 2
        measured = coefficient * (betas - 1.0) ** 2
        measured *= 1.0 + rng.normal(0.0, 3e-3, measured.size)
        measured[betas == 1.0] = 0.0
        ax.plot(dense_beta, theory, color=TEAL, linewidth=2.3, label=r"$\frac{1}{2}H_{ee}(\beta_e-1)^2$")
        ax.axhline(0.0, color=GRAY, linestyle=(0, (5, 3)), linewidth=1.4, label="1st order")
        ax.scatter(betas, measured, facecolor="white", edgecolor=TEXT, linewidth=1.0, s=26, zorder=3, label="measured forward")
        removal_idx = int(np.flatnonzero(betas == 0.0)[0])
        ax.scatter([0.0], [measured[removal_idx]], color=RED, marker="D", s=28, zorder=4)
        ax.annotate(
            "remove expert",
            xy=(0.0, measured[removal_idx]),
            xytext=(0.14, 0.74),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
            color=RED,
            fontsize=8,
        )
        panel_letter = chr(ord("d") + panel_idx)
        ax.set_title(f"({panel_letter}) {title}\nP{int(quantile * 100)}, Expert {expert_id}")
        ax.set_xlabel(r"Expert scale $\beta_e$")
        ax.set_ylabel(r"per-token $\Delta\mathcal{L}$")
        ax.text(0.96, 0.92, r"$H_{ee}/2$ = " + f"{coefficient:.2e}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        finish_axes(ax)

    handles = [
        Line2D([0], [0], color=TEAL, linewidth=2.3, label="diagonal Hessian prediction"),
        Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1.4, label="first-order prediction"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=TEXT, label="measured forward"),
        Line2D([0], [0], marker="D", linestyle="none", color=RED, label=r"single removal ($\beta_e=0$)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.465), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Observation B: Hessian = Gram = single-expert ablation", y=0.955, fontsize=13)
    fig.text(0.5, 0.495, "All comparisons are single-expert; no pairwise removal is used.", ha="center", va="bottom", fontsize=9, color=TEXT)
    add_preview_mark(fig)
    save(fig, output_dir, "method_validation_B_preview")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient, score = load_probe(Path(args.probe).expanduser().resolve())
    rng = np.random.default_rng(20260911)
    draw_signal_preview(output_dir, gradient, score, rng)
    draw_equivalence_preview(output_dir, score, rng)
    print(f"Saved {output_dir / 'method_validation_A_preview.{pdf,png}'}")
    print(f"Saved {output_dir / 'method_validation_B_preview.{pdf,png}'}")


if __name__ == "__main__":
    main()
