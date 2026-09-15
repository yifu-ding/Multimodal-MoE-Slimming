"""Plot measured and Hessian-predicted beta-loss curves for three experts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/maes-hessian-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


FONT_SIZE = 12
EXACT_COLOR = "#208A88"
MEASURED_COLOR = "#171717"
FIRST_ORDER_COLOR = "#929292"
GRID_COLOR = "#DFE3E5"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 2,
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw real P10/P50/P90 expert beta sweeps.")
    parser.add_argument("--input", required=True, help="Path to hessian_beta_sweep_L*.pt.")
    parser.add_argument("--output-dir", default=str(HERE))
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def load_sweep(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported beta-sweep payload: {path}")
    required = (
        "layer_idx",
        "loss_fn",
        "beta_values",
        "num_score_tokens",
        "route_consistency_verified",
        "curves",
        "metadata",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing beta-sweep fields: {missing}")
    betas = payload["beta_values"]
    if not isinstance(betas, torch.Tensor) or betas.ndim != 1 or betas.numel() < 3:
        raise ValueError("beta_values must be a rank-1 tensor with at least three entries.")
    if not bool(payload["route_consistency_verified"]):
        raise ValueError("Sweep did not verify fixed routing.")
    curves = payload["curves"]
    if not isinstance(curves, list) or [curve.get("label") for curve in curves] != [
        "low",
        "medium",
        "high",
    ]:
        raise ValueError("Expected low/medium/high curves in order.")
    for curve in curves:
        for key in (
            "expert_idx",
            "hessian_score_per_token",
            "active_batch_fraction",
            "gradient_per_token",
            "hessian_diagonal_per_token",
            "measured_delta_per_token",
        ):
            if key not in curve:
                raise KeyError(f"Missing curves[{curve.get('label')}][{key!r}].")
        values = curve["measured_delta_per_token"]
        if not isinstance(values, torch.Tensor) or values.shape != betas.shape:
            raise ValueError("Measured curve shape does not match beta_values.")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Measured curve contains non-finite values.")
    return payload


def _number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    payload = load_sweep(input_path)
    betas = payload["beta_values"].detach().cpu().double().numpy()
    dense_beta = np.linspace(float(betas.min()), float(betas.max()), 300)
    loss_fn = payload["loss_fn"]

    all_y = [0.0]
    prepared = []
    for curve in payload["curves"]:
        gradient = float(curve["gradient_per_token"])
        hessian = float(curve["hessian_diagonal_per_token"])
        dense_delta = dense_beta - 1.0
        exact = gradient * dense_delta + 0.5 * hessian * dense_delta**2
        first = gradient * dense_delta
        measured = curve["measured_delta_per_token"].detach().cpu().double().numpy()
        prepared.append((curve, exact, first, measured))
        all_y.extend(exact.tolist())
        all_y.extend(first.tolist())
        all_y.extend(measured.tolist())

    y_min, y_max = min(all_y), max(all_y)
    span = max(y_max - y_min, max(abs(y_min), abs(y_max)) * 0.05, 1e-12)
    y_limits = (y_min - 0.08 * span, y_max + 0.14 * span)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharex=True, sharey=True)
    titles = ("Low sensitivity", "Medium sensitivity", "High sensitivity")
    for panel_idx, (ax, title, item) in enumerate(zip(axes, titles, prepared)):
        curve, exact, first, measured = item
        ax.plot(
            dense_beta,
            exact,
            color=EXACT_COLOR,
            linewidth=2.3,
            label="exact Hessian curve",
            zorder=2,
        )
        ax.plot(
            dense_beta,
            first,
            color=FIRST_ORDER_COLOR,
            linestyle=(0, (5, 3)),
            linewidth=1.7,
            label="1st-order tangent",
            zorder=1,
        )
        ax.scatter(
            betas,
            measured,
            color=MEASURED_COLOR,
            facecolor="white",
            linewidth=1.15,
            s=30,
            label="measured forward",
            zorder=3,
        )
        ax.axvline(1.0, color="#C7CBCE", linewidth=0.9, zorder=0)
        ax.set_title(f"{title}\nExpert {int(curve['expert_idx'])}", fontsize=FONT_SIZE)
        ax.text(
            0.96,
            0.95,
            "$H_{ee}/2$ = " + _number(float(curve["hessian_score_per_token"]))
            + "\nactive batches = "
            + f"{100.0 * float(curve['active_batch_fraction']):.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_SIZE - 2,
        )
        ax.set_xlabel(r"Expert scale $\beta_e$")
        ax.set_xlim(float(betas.min()), float(betas.max()))
        ax.set_ylim(*y_limits)
        ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.text(
            0.5,
            -0.28,
            f"({chr(ord('a') + panel_idx)}) P{int(round(100 * float(curve['quantile'])))} expert",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
        )

    ylabel = (
        r"$\Delta \mathcal{L}_{\rm MSE}$ per score token"
        if loss_fn == "l2"
        else r"$\Delta \mathcal{L}_{\rm rel-L2}$ per score token"
    )
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.25, wspace=0.16)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename or f"hessian_beta_sweep_L{int(payload['layer_idx'])}"
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{basename}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input": str(input_path),
        "layer_idx": int(payload["layer_idx"]),
        "loss_fn": loss_fn,
        "num_score_tokens": int(payload["num_score_tokens"]),
        "experts": [
            {
                "label": curve["label"],
                "quantile": float(curve["quantile"]),
                "expert_idx": int(curve["expert_idx"]),
                "hessian_score_per_token": float(curve["hessian_score_per_token"]),
                "active_batch_fraction": float(curve["active_batch_fraction"]),
                "max_abs_measurement_error": float(
                    np.max(
                        np.abs(
                            curve["measured_delta_per_token"].double().numpy()
                            - curve["exact_quadratic_delta_per_token"].double().numpy()
                        )
                    )
                ),
            }
            for curve in payload["curves"]
        ],
    }
    with (output_dir / f"{basename}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved {output_dir / basename}.{{pdf,png,json}}")


if __name__ == "__main__":
    main()
