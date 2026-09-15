"""Plot a real expert-pair Hessian landscape collected by MAES calibration."""

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
from matplotlib.lines import Line2D
import numpy as np
import torch


FONT_SIZE = 12
TRUE_COLOR = "#238B8E"
FIRST_ORDER_COLOR = "#858585"
REMOVE_COLOR = "#C23B32"
DIRECT_COLOR = "#151515"
PLANE_COLOR = "#B9B9B9"

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
    parser = argparse.ArgumentParser(
        description="Draw a first-order versus exact-quadratic expert-removal figure."
    )
    parser.add_argument("--input", required=True, help="Path to hessian_probe_L*.pt")
    parser.add_argument(
        "--experts",
        type=int,
        nargs=2,
        metavar=("E", "F"),
        help="Expert IDs. Defaults to the validated pair or an automatic pair.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(HERE),
        help="Directory for PDF, PNG, and JSON outputs.",
    )
    parser.add_argument("--basename", default=None, help="Output basename without extension.")
    parser.add_argument(
        "--pair-pool-size",
        type=int,
        default=32,
        help="For automatic selection, search among this many largest diagonal entries.",
    )
    return parser.parse_args()


def load_probe(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported Hessian probe schema in {path}")
    required = (
        "layer_idx",
        "loss_fn",
        "num_batches",
        "num_score_tokens",
        "hessian_per_token",
        "gradient_per_token",
        "relative_hessian_asymmetry",
    )
    for key in required:
        if key not in payload:
            raise KeyError(f"Missing {key!r} in {path}")
    hessian = payload["hessian_per_token"]
    gradient = payload["gradient_per_token"]
    if not isinstance(hessian, torch.Tensor) or hessian.ndim != 2:
        raise TypeError("hessian_per_token must be a rank-2 torch.Tensor")
    if hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"Hessian must be square, got shape={tuple(hessian.shape)}")
    if not isinstance(gradient, torch.Tensor) or gradient.shape != hessian.shape[:1]:
        raise ValueError(
            "gradient_per_token must be a rank-1 tensor matching the Hessian size; "
            f"got gradient={getattr(gradient, 'shape', None)}, Hessian={tuple(hessian.shape)}"
        )
    if not bool(torch.isfinite(hessian).all()) or not bool(torch.isfinite(gradient).all()):
        raise ValueError("Hessian probe contains non-finite values")
    if int(payload["num_batches"]) <= 0 or int(payload["num_score_tokens"]) <= 0:
        raise ValueError("Hessian probe batch and score-token counts must be positive")

    pair_results = payload.get("pair_results")
    if pair_results is not None:
        pair_required = (
            "experts",
            "hessian",
            "gradient",
            "first_order_delta",
            "exact_quadratic_delta",
            "direct_ablation_delta",
        )
        for key in pair_required:
            if key not in pair_results:
                raise KeyError(f"Missing pair_results[{key!r}] in {path}")
        experts = tuple(int(value) for value in pair_results["experts"])
        if len(experts) != 2 or experts[0] == experts[1]:
            raise ValueError(f"Invalid stored expert pair: {experts}")
        if tuple(pair_results["hessian"].shape) != (2, 2):
            raise ValueError("pair_results['hessian'] must have shape (2, 2)")
        if tuple(pair_results["gradient"].shape) != (2,):
            raise ValueError("pair_results['gradient'] must have shape (2,)")
        removal_keys = {"remove_e", "remove_f", "remove_ef"}
        for name in ("first_order_delta", "exact_quadratic_delta"):
            if set(pair_results[name]) != removal_keys:
                raise ValueError(f"pair_results[{name!r}] has unexpected removal keys")
        direct = pair_results["direct_ablation_delta"]
        if direct is not None and set(direct) != removal_keys:
            raise ValueError("direct_ablation_delta has unexpected removal keys")
    return payload


def choose_pair(hessian: np.ndarray, pool_size: int) -> tuple[int, int, str]:
    diagonal = np.diag(hessian)
    valid = np.flatnonzero(diagonal > max(float(diagonal.max()) * 1e-10, 0.0))
    if valid.size < 2:
        raise ValueError("The probe does not contain two experts with positive curvature.")
    order = valid[np.argsort(diagonal[valid])[::-1]]
    pool = order[: min(max(pool_size, 2), order.size)]

    best = None
    for pos, e in enumerate(pool):
        for f in pool[pos + 1 :]:
            denom = np.sqrt(max(diagonal[e] * diagonal[f], 1e-30))
            rho = float(hessian[e, f] / denom)
            candidate = (abs(rho), min(int(e), int(f)), max(int(e), int(f)), rho)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    _, e, f, rho = best
    return e, f, f"auto: max |rho|={abs(rho):.4f} among top-{len(pool)} diagonal experts"


def orient(vector: np.ndarray) -> np.ndarray:
    vector = vector.copy()
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0:
        vector *= -1.0
    return vector


def number(value: float) -> str:
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude < 1e-3 or magnitude >= 1e3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    payload = load_probe(input_path)

    hessian_full = payload["hessian_per_token"].detach().cpu().double().numpy()
    gradient_full = payload["gradient_per_token"].detach().cpu().double().numpy()
    hessian_full = 0.5 * (hessian_full + hessian_full.T)
    pair_results = payload.get("pair_results")

    if args.experts is not None:
        e, f = (int(args.experts[0]), int(args.experts[1]))
        pair_source = "command line"
    elif pair_results is not None:
        e, f = (int(value) for value in pair_results["experts"])
        pair_source = "probe validation pair"
    else:
        e, f, pair_source = choose_pair(hessian_full, args.pair_pool_size)
    if e == f or min(e, f) < 0 or max(e, f) >= hessian_full.shape[0]:
        raise ValueError(f"Invalid expert pair {(e, f)} for Hessian shape {hessian_full.shape}")

    ids = np.asarray([e, f], dtype=np.int64)
    hessian = hessian_full[np.ix_(ids, ids)]
    gradient = gradient_full[ids]
    eigvals, eigvecs = np.linalg.eigh(hessian)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    v_max, v_min = orient(eigvecs[:, 0]), orient(eigvecs[:, 1])

    def exact_loss(db_e, db_f):
        return (
            gradient[0] * db_e
            + gradient[1] * db_f
            + 0.5
            * (
                hessian[0, 0] * db_e**2
                + 2.0 * hessian[0, 1] * db_e * db_f
                + hessian[1, 1] * db_f**2
            )
        )

    def first_order_loss(db_e, db_f):
        return gradient[0] * db_e + gradient[1] * db_f

    removal_points = {
        r"remove $e$": (-1.0, 0.0, "remove_e"),
        r"remove $f$": (0.0, -1.0, "remove_f"),
        r"remove $\{e,f\}$": (-1.0, -1.0, "remove_ef"),
    }
    direct = None
    if pair_results is not None:
        recorded_pair = tuple(int(value) for value in pair_results["experts"])
        if recorded_pair == (e, f):
            direct = pair_results.get("direct_ablation_delta")

    fig = plt.figure(figsize=(11.7, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.38)

    # Panel (a): the slice that actually reaches joint removal for a general H.
    ax_a = fig.add_subplot(grid[0, 0])
    joint_direction = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    alpha = np.linspace(-1.65, 0.3, 240)
    slice_e = alpha * joint_direction[0]
    slice_f = alpha * joint_direction[1]
    exact_slice = exact_loss(slice_e, slice_f)
    first_slice = first_order_loss(slice_e, slice_f)
    ax_a.plot(alpha, exact_slice, color=TRUE_COLOR, linewidth=2.4, label="exact quadratic")
    ax_a.plot(
        alpha,
        first_slice,
        color=FIRST_ORDER_COLOR,
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        label="first-order tangent",
    )
    alpha_joint = -np.sqrt(2.0)
    joint_exact = float(exact_loss(-1.0, -1.0))
    joint_first = float(first_order_loss(-1.0, -1.0))
    ax_a.plot(
        [alpha_joint, alpha_joint],
        [joint_first, joint_exact],
        color=REMOVE_COLOR,
        linewidth=1.6,
        zorder=4,
    )
    ax_a.scatter([alpha_joint], [joint_exact], color=REMOVE_COLOR, s=34, zorder=5)
    ax_a.scatter(
        [alpha_joint],
        [joint_first],
        facecolors="white",
        edgecolors=REMOVE_COLOR,
        s=34,
        zorder=5,
    )
    if direct is not None and "remove_ef" in direct:
        ax_a.scatter(
            [alpha_joint],
            [direct["remove_ef"]],
            color=DIRECT_COLOR,
            marker="x",
            s=48,
            linewidth=1.4,
            zorder=7,
            label="direct removal",
        )
    ax_a.annotate(
        f"curvature gap\n$={number(joint_exact - joint_first)}$",
        xy=(alpha_joint, 0.5 * (joint_exact + joint_first)),
        xytext=(alpha_joint + 0.52, 0.58 * joint_exact),
        fontsize=FONT_SIZE - 2,
        color=REMOVE_COLOR,
        arrowprops=dict(arrowstyle="-", color=REMOVE_COLOR, lw=0.8),
    )
    ax_a.scatter([0], [0], color="black", s=20, zorder=5)
    ax_a.set_xlabel(r"$\alpha$ along joint-removal direction", fontsize=FONT_SIZE - 1)
    ax_a.set_ylabel(r"per-token $\Delta\mathcal{L}$", fontsize=FONT_SIZE)
    ax_a.legend(loc="upper left", frameon=False, fontsize=FONT_SIZE - 3)
    ax_a.grid(True, color="#E1E4E7", linestyle="--", linewidth=0.7, alpha=0.8)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.tick_params(labelsize=FONT_SIZE - 2)
    ax_a.set_box_aspect(0.72)
    ax_a.text(
        0.5,
        -0.34,
        "(a) Joint-removal slice",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )

    # Panel (b): measured 2x2 Hessian contours and its principal directions.
    ax_b = fig.add_subplot(grid[0, 1])
    db_e = np.linspace(-1.45, 0.45, 220)
    db_f = np.linspace(-1.45, 0.45, 220)
    DBE, DBF = np.meshgrid(db_e, db_f)
    exact_grid = exact_loss(DBE, DBF)
    ax_b.contour(DBE, DBF, exact_grid, levels=11, cmap="YlGnBu", linewidths=0.9)
    ax_b.scatter([0], [0], color="black", s=24, zorder=5)
    removal_label_positions = {
        "remove_e": (-1.0, 0.11, "center"),
        "remove_f": (0.0, -1.14, "center"),
        "remove_ef": (-1.0, -1.14, "center"),
    }
    for label, (de, df, key) in removal_points.items():
        ax_b.annotate(
            "",
            xy=(de, df),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=REMOVE_COLOR, lw=1.25),
        )
        ax_b.scatter([de], [df], color=REMOVE_COLOR, s=27, zorder=5)
        label_x, label_y, label_align = removal_label_positions[key]
        ax_b.text(
            label_x,
            label_y,
            label,
            fontsize=FONT_SIZE - 3,
            color=REMOVE_COLOR,
            ha=label_align,
        )
    for vector, name in ((v_max, r"$v_{\max}$"), (v_min, r"$v_{\min}$")):
        scale = 0.32
        ax_b.annotate(
            "",
            xy=(scale * vector[0], scale * vector[1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#245F73", lw=1.4),
        )
        ax_b.text(
            scale * vector[0] * 1.22,
            scale * vector[1] * 1.22,
            name,
            fontsize=FONT_SIZE - 2,
            color="#245F73",
        )
    ax_b.set_xlabel(r"$\Delta\beta_e$", fontsize=FONT_SIZE)
    ax_b.set_ylabel(r"$\Delta\beta_f$", fontsize=FONT_SIZE)
    ax_b.grid(True, color="#E1E4E7", linestyle="--", linewidth=0.7, alpha=0.8)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.tick_params(labelsize=FONT_SIZE - 2)
    ax_b.set_box_aspect(0.72)
    ax_b.text(
        0.5,
        -0.34,
        "(b) Curvature and removal steps",
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )

    # Panel (c): exact quadratic surface over the measured first-order plane.
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")
    first_grid = first_order_loss(DBE, DBF)
    ax_c.plot_surface(
        DBE,
        DBF,
        first_grid,
        color=PLANE_COLOR,
        alpha=0.36,
        rstride=14,
        cstride=14,
        edgecolor="#777777",
        linewidth=0.25,
    )
    ax_c.plot_surface(
        DBE,
        DBF,
        exact_grid,
        color=TRUE_COLOR,
        alpha=0.58,
        rstride=7,
        cstride=7,
        edgecolor="#246C70",
        linewidth=0.2,
        antialiased=True,
    )
    z_span = max(float(exact_grid.max() - exact_grid.min()), 1e-12)
    surface_label_offsets = {
        "remove_e": (0.12, 0.08, 0.09),
        "remove_f": (0.10, -0.04, 0.08),
        "remove_ef": (-0.28, -0.08, 0.10),
    }
    removal_names = {
        "remove_e": f"remove E{e}",
        "remove_f": f"remove E{f}",
        "remove_ef": "remove both",
    }
    for _, (de, df, key) in removal_points.items():
        exact_value = float(exact_loss(de, df))
        first_value = float(first_order_loss(de, df))
        ax_c.plot(
            [de, de],
            [df, df],
            [first_value, exact_value],
            color=REMOVE_COLOR,
            linewidth=1.8,
            zorder=10,
        )
        ax_c.scatter(
            [de], [df], [exact_value], color=TRUE_COLOR, s=36, depthshade=False, zorder=11
        )
        ax_c.scatter(
            [de],
            [df],
            [first_value],
            facecolors="white",
            edgecolors=FIRST_ORDER_COLOR,
            linewidths=1.4,
            s=36,
            depthshade=False,
            zorder=11,
        )
        if direct is not None and key in direct:
            ax_c.scatter(
                [de],
                [df],
                [direct[key]],
                color=DIRECT_COLOR,
                marker="x",
                s=42,
                linewidth=1.3,
                depthshade=False,
                zorder=12,
            )
        offset_e, offset_f, offset_z = surface_label_offsets[key]
        ax_c.text(
            de + offset_e,
            df + offset_f,
            exact_value + offset_z * z_span,
            removal_names[key]
            + "\n1st = "
            + number(first_value)
            + "\n2nd = "
            + number(exact_value),
            fontsize=FONT_SIZE - 5.5,
            color="#222222",
            weight="semibold",
        )
    ax_c.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=FIRST_ORDER_COLOR,
                markeredgewidth=1.4,
                label="1st-order value",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=TRUE_COLOR,
                markeredgecolor=TRUE_COLOR,
                label="2nd-order value",
            ),
            Line2D([0], [0], color=REMOVE_COLOR, linewidth=1.8, label="curvature gap"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        fontsize=FONT_SIZE - 5,
        ncol=1,
        handlelength=1.5,
        borderaxespad=0.0,
    )
    ax_c.set_xlabel(r"$\Delta\beta_e$", fontsize=FONT_SIZE - 1, labelpad=2)
    ax_c.set_ylabel(r"$\Delta\beta_f$", fontsize=FONT_SIZE - 1, labelpad=2)
    ax_c.set_zlabel(r"$\Delta\mathcal{L}$", fontsize=FONT_SIZE - 1, labelpad=0)
    z_min = min(float(exact_grid.min()), float(first_grid.min()))
    z_max = max(float(exact_grid.max()), float(first_grid.max()))
    ax_c.set_zlim(z_min - 0.04 * z_span, z_max + 0.10 * z_span)
    ax_c.tick_params(labelsize=FONT_SIZE - 5, pad=-2)
    ax_c.view_init(elev=24, azim=-48)
    ax_c.set_box_aspect((1, 1, 0.72))
    ax_c.text2D(
        0.5,
        -0.08,
        "(c) Exact quadratic vs. tangent plane",
        transform=ax_c.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    for pane in (ax_c.xaxis.pane, ax_c.yaxis.pane, ax_c.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("#DDDDDD")

    layer_idx = int(payload["layer_idx"])
    fig.suptitle(
        f"Layer {layer_idx}, experts ({e}, {f}): first-order tangent and exact curvature",
        fontsize=FONT_SIZE + 1,
        y=0.99,
    )
    fig.subplots_adjust(bottom=0.27, top=0.88)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename or f"hessian_landscape_L{layer_idx}_E{e}_E{f}"
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{basename}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    rho = float(hessian[0, 1] / np.sqrt(max(hessian[0, 0] * hessian[1, 1], 1e-30)))
    summary = {
        "input": str(input_path),
        "layer_idx": layer_idx,
        "experts": [e, f],
        "pair_source": pair_source,
        "gradient": gradient.tolist(),
        "hessian": hessian.tolist(),
        "normalized_coupling_rho": rho,
        "eigenvalues": eigvals.tolist(),
        "first_order_delta": {
            key: float(first_order_loss(de, df))
            for _, (de, df, key) in removal_points.items()
        },
        "exact_quadratic_delta": {
            key: float(exact_loss(de, df))
            for _, (de, df, key) in removal_points.items()
        },
        "direct_ablation_delta": direct,
        "relative_hessian_asymmetry": float(payload["relative_hessian_asymmetry"]),
    }
    with (output_dir / f"{basename}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Pair source: {pair_source}")
    print(f"Layer {layer_idx}, experts e={e}, f={f}")
    print(
        f"H_ee={hessian[0, 0]:.8g}, H_ff={hessian[1, 1]:.8g}, "
        f"H_ef={hessian[0, 1]:.8g}, rho={rho:.5f}"
    )
    for _, (de, df, key) in removal_points.items():
        measured = ""
        if direct is not None and key in direct:
            measured = f", direct={direct[key]:.8g}"
        print(
            f"{key}: first={first_order_loss(de, df):.8g}, "
            f"quadratic={exact_loss(de, df):.8g}{measured}"
        )
    print(f"Saved {output_dir / (basename + '.pdf')}")
    print(f"Saved {output_dir / (basename + '.png')}")


if __name__ == "__main__":
    main()
