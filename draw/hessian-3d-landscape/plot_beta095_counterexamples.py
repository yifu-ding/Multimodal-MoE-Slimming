"""Plot measured displaced-gradient counterexamples and identity loss curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


COLORS = {"truth": "#252525", "hessian": "#c73637", "first": "#2678af"}


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=200)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def plot(directory):
    with (directory / "scores.csv").open() as handle:
        rows = [{key: (value if key == "loss_fn" else float(value)) for key, value in row.items()}
                for row in csv.DictReader(handle)]
    with (directory / "beta_curves.csv").open() as handle:
        curves = [{key: (value if key == "selection_group" else float(value)) for key, value in row.items()}
                  for row in csv.DictReader(handle)]
    summary = json.loads((directory / "statistics.json").read_text())
    meta = json.loads((directory / "metadata.json").read_text())
    label = "L2" if meta["loss_fn"] == "l2" else "KL"
    by_id = {int(row["expert"]): row for row in rows}
    truth = np.array([r["true_removal_at_identity"] for r in rows])
    hessian = np.array([r["hessian_half_at_identity"] for r in rows])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    ax = axes[0, 0]
    ax.scatter(truth, hessian, s=15, color=COLORS["hessian"], alpha=.75)
    bounds = [min(truth.min(), hessian.min()), max(truth.max(), hessian.max())]
    ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
    ax.set(xscale="log", yscale="log", xlabel="Measured removal loss, background 1",
           ylabel=r"Existing $H_{ee}(1)/2$", title=f"(a) {label}: all {len(rows)} experts")
    metric = summary["metrics"]["hessian_half_at_identity"]
    ax.text(.04, .96, f"Spearman = {metric['spearman']:.5f}\nMedian relative error = {metric['median_relative_error']:.2e}",
            transform=ax.transAxes, va="top", fontsize=9)
    for ax, key, panel in zip(axes[0, 1:], ("first_order_signed", "first_order_abs"), ("b", "c")):
        ranks = rankdata([r[key] for r in rows])
        truth_ranks = rankdata(truth)
        ax.scatter(truth_ranks, ranks, s=15, alpha=.7, color=COLORS["first"])
        ax.plot([1, len(rows)], [1, len(rows)], "k--", linewidth=1)
        metric = summary["metrics"][key]
        kind = "signed" if key.endswith("signed") else "absolute"
        ax.set(xlabel="Removal sensitivity rank at background 1", ylabel=f"{kind.capitalize()} gradient proxy rank at background 0.95",
               title=f"({panel}) Nonzero first order ({kind})")
        ax.text(.04, .96, f"Spearman = {metric['spearman']:.4f}\nInversions = {metric['inversion_rate']:.2%}",
                transform=ax.transAxes, va="top", fontsize=9)
    for index, ax in enumerate(axes[1]):
        if index >= len(summary["display_pairs"]):
            ax.text(.5, .5, "No qualifying counterexample", ha="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        pair = summary["display_pairs"][index]
        ids = [pair["expert_low"], pair["expert_high"]]
        true_values = np.array([by_id[e]["true_removal_at_identity"] for e in ids])
        h_values = np.array([by_id[e]["hessian_half_at_identity"] for e in ids])
        first_values = np.array([by_id[e][pair["baseline"]] for e in ids])
        ax.plot([0, 1], true_values / max(abs(true_values)), "o-", color=COLORS["truth"], label="Measured removal (bg 1)", linewidth=2)
        ax.plot([0, 1], h_values / max(abs(true_values)), "D--", color=COLORS["hessian"], label="Hessian / 2 (bg 1)", markersize=5)
        first_label = "Absolute gradient proxy" if pair["baseline"].endswith("abs") else "Signed gradient proxy"
        ax.plot([0, 1], first_values / max(abs(first_values)), "s-", color=COLORS["first"], label=f"{first_label} (bg 0.95)")
        ax.set(xticks=[0, 1], xticklabels=[f"Expert {e}" for e in ids], xlim=(-.15, 1.15),
               ylabel="Relative score within pair", title=f"({chr(ord('d')+index)}) Counterexample {index+1}")
        ax.text(.03, .97, f"Hessian relative errors: {pair['hessian_relative_error_low']:.2e}, {pair['hessian_relative_error_high']:.2e}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.set_ylim(min(0, min(first_values / max(abs(first_values)))) - .06, 1.18)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{label}: identity Hessian vs. a gradient measured with all experts scaled to 0.95", fontsize=14)
    fig.supxlabel("Pair panels: loss and Hessian share the maximum measured loss; the gradient proxy uses its own maximum magnitude.", fontsize=9)
    save(fig, directory, "counterexample_comparison")

    def curve_panel(ax, e, title):
        measured = sorted((c for c in curves if int(c["expert"]) == e and c["background_beta"] == 1), key=lambda c: c["beta"])
        beta = np.array([c["beta"] for c in measured])
        dense = np.linspace(beta.min(), beta.max(), 201)
        ax.plot(dense, by_id[e]["hessian_half_at_identity"] * (dense - 1)**2,
                color=COLORS["hessian"], label=r"$H_{ee}(1)(\beta-1)^2/2$")
        ax.scatter(beta, [c["measured_delta_loss"] for c in measured], facecolors="none", edgecolors=COLORS["truth"], s=24, label="Real forward")
        ax.scatter([0], [by_id[e]["true_removal_at_identity"]], color=COLORS["truth"], marker="D", s=30, label="Independent removal")
        ax.set(title=title, xlabel=r"Expert scale $\beta_e$ (others = 1)", ylabel=f"Measured {label} change")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained", sharex=True)
    for ax, item in zip(axes.flat, summary["representatives"]):
        curve_panel(ax, item["expert"], f"{item['group'].capitalize()} P{int(item['quantile']*100)}: expert {item['expert']}")
    for row_axes in axes:
        limits = [ax.get_ylim() for ax in row_axes]
        for ax in row_axes:
            ax.set_ylim(min(l[0] for l in limits), max(l[1] for l in limits))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    fig.suptitle(f"{label}: nine representatives selected by measured identity removal loss", fontsize=14)
    save(fig, directory, "representative_curves")

    if summary["display_pairs"]:
        count = len(summary["display_pairs"])
        fig, axes = plt.subplots(count, 2, figsize=(10, 3 * count), layout="constrained", squeeze=False)
        for row_axes, pair in zip(axes, summary["display_pairs"]):
            for ax, key in zip(row_axes, ("expert_low", "expert_high")):
                e = pair[key]
                curve_panel(ax, e, f"Expert {e}: |g(0.95)| = {abs(by_id[e]['gradient_at_work']):.3e}")
            limits = [ax.get_ylim() for ax in row_axes]
            for ax in row_axes:
                ax.set_ylim(min(l[0] for l in limits), max(l[1] for l in limits))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
        fig.suptitle(f"{label}: independent loss curves for selected counterexample pairs\n"
                     "Title gradients are measured with ALL experts at 0.95; curve backgrounds stay at 1.", fontsize=12)
        save(fig, directory, "counterexample_curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    plot(parser.parse_args().data_dir)
