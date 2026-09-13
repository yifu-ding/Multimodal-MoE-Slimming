"""Plot measured gradient magnitude percentages for every expert, in ID order."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BETAS = (0., .25, .5, .75, .95)
COLORS = ("#333333", "#277da8", "#278655", "#c18422", "#b43f56")


def plot(directory):
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    ratio_fig, ratio_axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, layout="constrained")
    residual_fig, residual_axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for loss, ratio_ax, residual_ax in zip(("l2", "kl_div"), ratio_axes, residual_axes):
        with (directory / loss / "gradients.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        validation = json.loads((directory / loss / "validation.json").read_text())
        ids = np.arange(128)
        zero = sorted((r for r in rows if float(r["beta_global"]) == 0), key=lambda r: int(r["expert"]))
        valid = np.array([r["ratio_valid"] == "True" for r in zero])
        noise = np.full(128, np.nan)
        noise[valid] = 100 * validation["ratio_denominator_threshold"] / np.array([abs(float(r["gradient_signed"])) for r in zero])[valid]
        residual_ax.fill_between(ids, -noise, noise, color="gray", alpha=.25, label="Numerical threshold")
        for beta, color in zip(BETAS, COLORS):
            selected = sorted((r for r in rows if float(r["beta_global"]) == beta), key=lambda r: int(r["expert"]))
            values = np.array([float(r["ratio_to_global_zero_pct"]) if r["ratio_valid"] == "True" else np.nan for r in selected])
            ratio_ax.plot(ids, values, linewidth=1.2, color=color, label=f"beta = {beta:g}")
            residual_ax.plot(ids, values - 100 * (1-beta), linewidth=1, color=color, label=f"beta = {beta:g}")
        title = "L2" if loss == "l2" else "Hidden-state softmax KL"
        for ax in (ratio_ax, residual_ax):
            ax.set(xlabel="Expert ID", xlim=(0, 127), title=f"{title} ({int(valid.sum())}/128 valid ratios)", xticks=[0, 32, 64, 96, 127])
            ax.grid(axis="y", alpha=.15)
        ratio_ax.set(ylim=(-3, 105), yticks=[0, 5, 25, 50, 75, 100])
        residual_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ratio_axes[0].set_ylabel(r"$100\,|g_e(\beta\mathbf{1})|/|g_e(\mathbf{0})|$ (%)")
    for ax in residual_axes:
        ax.set_ylabel("Measured ratio minus 100(1-beta)\n(percentage points)")
    handles, labels = ratio_axes[0].get_legend_handles_labels()
    ratio_fig.legend(handles, labels, loc="outside lower center", ncols=5)
    residual_handles, residual_labels = residual_axes[0].get_legend_handles_labels()
    residual_fig.legend(residual_handles, residual_labels, loc="outside lower center", ncols=6, fontsize=9)
    ratio_fig.suptitle("All experts scaled together: measured partial gradients, layer 0, 32 GQA samples", fontsize=13)
    residual_fig.suptitle("Deviation from endpoint scaling; shaded numerical floor is not a confidence interval", fontsize=12)
    for fig, name in ((ratio_fig, "gradient_ratios"), (residual_fig, "ratio_deviations")):
        fig.savefig(directory / f"{name}.png", dpi=200)
        fig.savefig(directory / f"{name}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("draw/hessian-3d-landscape/global_beta_sweep"))
    plot(parser.parse_args().data_dir)
