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
VIOLET = "#6E5AA0"

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
    # `local_gradient_at_beta` (README plan (1): re-differentiate at each swept
    # beta_0 instead of extrapolating identity_gradient from beta=1) is only
    # present in data collected after that column was added. Older data
    # directories (e.g. the published draw/hessian-3d-landscape/data/) do not
    # have it, so this stays optional rather than a hard schema requirement.
    has_local_gradient = bool(beta_rows) and "local_gradient_at_beta" in beta_rows[0]
    curves = []
    for label in ("low", "medium", "high"):
        rows = [row for row in beta_rows if row["sensitivity"] == label]
        if not rows:
            raise ValueError(f"Missing {label!r} beta curve.")
        curve = {
            "label": label,
            "quantile": float(rows[0]["quantile"]),
            "expert_idx": int(rows[0]["expert"]),
            "betas": np.asarray([float(row["beta"]) for row in rows]),
            "measured": np.asarray([float(row["measured_delta_mse"]) for row in rows]),
            "gradient": float(rows[0]["identity_gradient"]),
            "hessian_score": float(rows[0]["hvp_hessian_half"]),
        }
        if has_local_gradient:
            curve["local_gradient"] = np.asarray(
                [float(row["local_gradient_at_beta"]) for row in rows]
            )
        curves.append(curve)
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
    parser.add_argument("--kl-data-dir", default=str(HERE / "data_kl"))
    parser.add_argument("--output-dir", default=str(HERE))
    parser.add_argument("--basename", default="method_validation_B")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    metadata, scores, curves = load(data_dir)
    kl_data_dir = Path(args.kl_data_dir).expanduser().resolve()
    kl_metadata, kl_scores, kl_curves = load(kl_data_dir)
    if metadata.get("loss_fn", "l2") != "l2" or kl_metadata.get("loss_fn") != "kl_div":
        raise ValueError("Use --data-dir for L2 and --kl-data-dir for KL data.")
    for key in ("selection_manifest_sha256", "num_samples", "layers", "model_name_or_path"):
        if metadata[key] != kl_metadata[key]:
            raise ValueError(f"L2/KL calibration mismatch: {key}")
    kl_valid = np.isfinite(kl_scores["hvp"]) & np.isfinite(kl_scores["ablation"]) & (kl_scores["hvp"] > 0)
    kl_hvp = kl_scores["hvp"][kl_valid]
    kl_ablation = kl_scores["ablation"][kl_valid]
    kl_summary = metric_summary(kl_hvp, kl_ablation)
    if not kl_metadata["route_consistency_verified"]:
        raise ValueError("KL beta sweep did not verify fixed routing.")
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
    # Three score comparisons, followed by separate L2 and KL sweep rows.
    fig = plt.figure(figsize=(12.6, 11.2))
    grid = fig.add_gridspec(3, 6)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.87, bottom=0.065, hspace=0.85, wspace=1.65)
    top_axes = tuple(fig.add_subplot(grid[0, col:col + 2]) for col in (0, 2, 4))
    for panel_idx, (ax, values, title, ylabel, summary) in enumerate(zip(
        top_axes,
        (energy, ablation, kl_ablation),
        ("(a) L2: HVP vs. Gram energy", "(b) L2: HVP vs. single removal", "(c) KL: HVP vs. single removal"),
        (r"$s_e^{\rm energy}$", r"$s_e^{\rm ablate}$", r"$s_e^{\rm ablate}$ (KL)"),
        (summaries["energy_vs_hvp"], summaries["ablation_vs_hvp"], kl_summary),
    )):
        xvalues = kl_hvp if panel_idx == 2 else hvp
        panel_layers = kl_scores["layer"][kl_valid] if panel_idx == 2 else layer_ids
        panel_positive = np.concatenate((xvalues, values[values > 0]))
        panel_limits = (float(panel_positive.min()) * 0.85, float(panel_positive.max()) * 1.18)
        # y=x drawn thin, light, and behind the scatter (zorder=0): for the
        # exact-quadratic l2/rel_l2 case the points sit exactly on this line,
        # so a bold line on top of them mostly just redraws the data. Kept as
        # a faint guide rather than removed outright, since a future
        # non-exact loss variant (e.g. kl_div) would show real, visible
        # deviation from it worth comparing against.
        scatter = ax.scatter(xvalues, values, c=panel_layers, cmap="viridis", s=12, alpha=0.6,
                             linewidths=0, zorder=2)
        ax.plot(panel_limits, panel_limits, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.7,
                zorder=0, label=r"$y=x$")
        ax.set(xscale="log", yscale="log", xlim=panel_limits, ylim=panel_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$s_e^{\rm HVP}=H_{ee}/2$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        ax.text(
            0.04,
            0.95,
            f"Pearson = {summary['pearson']:.6f}\nSpearman = {summary['spearman']:.6f}\nMedian rel. error = {summary['median_relative_error']:.2%}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        if len(np.unique(panel_layers)) > 1:
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
            colorbar.set_label("Layer", fontsize=8, labelpad=2)
        finish_axes(ax)

    gradient_max_abs = float(np.max(np.abs(gradient_values)))
    if max(gradient_max_abs, float(np.max(np.abs(kl_scores["gradient"])))) > 1e-12:
        raise ValueError(f"Expected first-order degeneracy, got max |g|={gradient_max_abs:.3e}.")
    # Kept as a correctness check even without panel (c): every plotted layer
    # should still have the same number of active experts. num_experts still
    # feeds the forward_evaluations_per_layer JSON stat below.
    experts_per_layer = [int(np.sum(layer_ids == layer)) for layer in np.unique(layer_ids)]
    if len(set(experts_per_layer)) != 1:
        raise ValueError("Expected the same number of active experts in every plotted layer.")
    num_experts = experts_per_layer[0]

    bottom_axes = (
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:6]),
        fig.add_subplot(grid[2, 0:2]),
        fig.add_subplot(grid[2, 2:4]),
        fig.add_subplot(grid[2, 4:6]),
    )
    sweep_layer = int(metadata["sweep_layer"])
    if not bool(metadata["route_consistency_verified"]):
        raise ValueError("Beta sweep did not verify fixed router indices and weights.")
    sensitivity_titles = ("Low sensitivity", "Medium sensitivity", "High sensitivity")
    # Share both y-ranges within each loss; L2 and KL have different units.
    sweep_summaries = []
    for panel_idx, (ax, curve, title) in enumerate(zip(bottom_axes, curves + kl_curves, sensitivity_titles * 2)):
        row_curves = curves if panel_idx < 3 else kl_curves
        loss_label = "L2" if panel_idx < 3 else "KL"
        betas = curve["betas"]
        dense_beta = np.linspace(float(betas.min()), float(betas.max()), 300)
        left_min = min(float(c["measured"].min()) for c in row_curves)
        left_max = max(float(c["measured"].max()) for c in row_curves)
        left_pad = 0.06 * (left_max - left_min)
        shared_left_ylim = (left_min - left_pad, left_max + left_pad)
        if "local_gradient" in curve:
            right_min = min(float(c["local_gradient"].min()) for c in row_curves)
            right_max = max(float(c["local_gradient"].max()) for c in row_curves)
            right_pad = 0.06 * (right_max - right_min)
            shared_right_ylim = (right_min - right_pad, right_max + right_pad)
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
        summary_entry = {
            "loss_fn": "l2" if panel_idx < 3 else "kl_div",
            "label": curve["label"],
            "expert_idx": expert_idx,
            "max_abs_error": float(absolute_error.max()),
            "median_relative_error_nonbaseline": float(np.median(relative_error[nonbaseline])),
            "max_relative_error_nonbaseline": float(np.max(relative_error[nonbaseline])),
        }
        if "local_gradient" in curve:
            # hessian_score is H_ee/2 (the hvp_hessian_half field); the local
            # gradient identity is g_e(beta_0) = H_ee * (beta_0 - 1), so the
            # expected value needs the factor of 2 back.
            local_gradient = curve["local_gradient"]
            expected_local_gradient = 2.0 * hessian_score * (betas - 1.0)
            local_gradient_abs_error = np.abs(local_gradient - expected_local_gradient)
            local_relative_error = local_gradient_abs_error / np.maximum(
                np.abs(expected_local_gradient), 1e-30
            )
            summary_entry["local_gradient_vs_hee_delta"] = {
                "max_abs_error": float(local_gradient_abs_error.max()),
                "max_relative_error_nonbaseline": float(
                    np.max(local_relative_error[nonbaseline])
                ),
            }
        sweep_summaries.append(summary_entry)
        ax.plot(dense_beta, second, color=TEAL, linewidth=2.3)
        # The flat "extrapolate from beta=1" line is just one member of the
        # local-tangent family below, evaluated at beta_0=1 (where its slope
        # happens to be 0) -- draw it separately only when there is no
        # per-point local-gradient data to already cover that case.
        if "local_gradient" not in curve:
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
            f"({letter}) {loss_label}: {title}\nP{int(round(100 * float(curve['quantile'])))}, Expert {expert_idx}"
        )
        ax.set_xlabel(r"Expert scale $\beta_e$")
        ax.set_ylabel(r"per-token $\Delta\mathcal{L}_{\rm " + ("MSE" if panel_idx < 3 else "KL") + "}$")
        ax.set_ylim(shared_left_ylim)
        ax.text(
            0.96,
            0.92,
            r"$H_{ee}/2$ = " + f"{hessian_score:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        # README plan (1): at each swept beta_0, evaluate the local slope
        # directly (real forward+backward at that beta_0) instead of
        # extrapolating identity_gradient from beta=1. This plots only the
        # real measured local_gradient_at_beta values, connected point to
        # point -- no fitted/theoretical curve underneath. An earlier version
        # also drew the theoretical H_ee*(beta-1) line here, which sat almost
        # exactly on top of the real markers (they agree to ~1e-7, see
        # local_gradient_vs_hee_delta in the JSON) and made the panel look
        # like a drawn line rather than real per-point measurements.
        if "local_gradient" in curve:
            local_gradient = curve["local_gradient"]
            order = np.argsort(betas)
            right_ax = ax.twinx()
            right_ax.plot(betas[order], local_gradient[order], color=VIOLET, linewidth=1.4,
                         marker="^", markersize=6, markerfacecolor="white",
                         markeredgecolor=VIOLET, zorder=3)
            right_ax.set_ylabel(r"measured $dL/d\beta_e$ at $\beta_0$", color=VIOLET, fontsize=9)
            right_ax.tick_params(axis="y", labelcolor=VIOLET, labelsize=8)
            right_ax.set_ylim(shared_right_ylim)
            right_ax.spines["top"].set_visible(False)
        finish_axes(ax)

    has_local_gradient_any = any("local_gradient" in curve for curve in curves + kl_curves)
    handles = [
        Line2D([0], [0], color=TEAL, linewidth=2.3, label="diagonal Hessian prediction"),
        Line2D(
            [0], [0],
            marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=TEXT,
            label="measured forward (ground truth, not a prediction)",
        ),
        Line2D([0], [0], marker="D", linestyle="none", color=RED, label=r"single removal ($\beta_e=0$)"),
    ]
    if has_local_gradient_any:
        # Real measured dL/dbeta at each swept beta_0, connected point to
        # point -- right axis of panels (c)-(e), separate from the loss curve
        # on the left axis. No fitted/theoretical line underneath.
        handles.insert(
            1,
            Line2D([0], [0], color=VIOLET, linewidth=1.4, marker="^", markersize=6,
                   markerfacecolor="white", markeredgecolor=VIOLET,
                   label=r"measured $dL/d\beta_e$ (right axis)"),
        )
    else:
        handles.insert(
            1,
            Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1.4,
                   label=r"first order: $g_e(\beta_e-1)=0$"),
        )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.635,
        "Ranking vs. true removal: first order Spearman = undefined "
        r"(all $g_e\approx0$); second order: "
        f"L2 = {summaries['ablation_vs_hvp']['spearman']:.6f}, KL = {kl_summary['spearman']:.6f}.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    num_samples = int(metadata["num_samples"])
    num_layers = len(metadata["layers"])
    fig.suptitle(
        "Observation B: L2 and KL Hessian validation\n"
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
        "loss_fn": "l2",
        "kl": {
            "loss_fn": "kl_div",
            "data_dir": os.path.relpath(kl_data_dir, HERE),
            "ablation_vs_hvp": kl_summary,
            "max_abs_identity_gradient": float(np.max(np.abs(kl_scores["gradient"]))),
        },
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
