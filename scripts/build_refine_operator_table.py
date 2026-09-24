#!/usr/bin/env python3
"""Build the Refine-operator comparison table (paper Table 6) from a
depth-sweep checkpoint.

Unlike ``summarize_placement_ablation.py`` (which prints one row per
window), this aggregates across windows and prune ratios: for each
(operator, L) cell it reports the *median* rank-load spread and the median
gap to the theoretical lower bound tau0, in absolute MiB, following
docs' Table 6 format (spread with a gray gap subscript, plus a tau0 row and
a per-rank load Phi-bar row for scale).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_LAYERS = (4, 8, 12, 16, 24, 32, 48)
DEFAULT_METHODS = (
    "sort_only",
    "greedy_pairwise_swap",
    "greedy_full_bijection",
    "simulated_annealing",
    "tabu_search",
    "beam_search",
    "milp_assignment",
)
METHOD_LABELS = {
    "sort_only": "Sort only",
    "greedy_pairwise_swap": "Pairwise swap",
    "greedy_full_bijection": "All bijections",
    "simulated_annealing": "Simulated annealing",
    "tabu_search": "Tabu search",
    "beam_search": "Beam search",
    "milp_assignment": "Exact solver (MILP)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--methods", nargs="+", type=str, default=list(DEFAULT_METHODS)
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=2048,
        help="Hidden size used to convert phi units (channel*expert) to bytes.",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=2,
        help="Bytes per weight element (2 for bf16/fp16).",
    )
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def instance_key(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        record["plan_sha256"],
        record.get("layer_start"),
        record.get("layer_end"),
    )


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_aggregates(
    records: list[dict[str, Any]], methods: tuple[str, ...], layers: tuple[int, ...]
) -> tuple[
    dict[str, dict[int, list[tuple[float, float]]]],
    dict[int, dict[tuple[Any, ...], float]],
    dict[int, dict[tuple[Any, ...], float]],
]:
    by_method_layer: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tau0_by_layer: dict[int, dict[tuple[Any, ...], float]] = defaultdict(dict)
    phibar_by_layer: dict[int, dict[tuple[Any, ...], float]] = defaultdict(dict)

    for record in records:
        num_layers = record.get("layers")
        if num_layers not in layers:
            continue
        key = instance_key(record)
        if record.get("tau0") is not None:
            tau0_by_layer[num_layers][key] = float(record["tau0"])
        loads = record.get("rank_weight_loads")
        if loads:
            phibar_by_layer[num_layers][key] = sum(loads) / len(loads)

        method = record.get("method")
        if method not in methods:
            continue
        spread = record.get("spread")
        tau0 = record.get("tau0")
        if spread is None or tau0 is None:
            continue
        by_method_layer[method][num_layers].append((float(spread), float(spread) - float(tau0)))

    return by_method_layer, tau0_by_layer, phibar_by_layer


def format_cell(spread_mib: float | None, gap_mib: float | None) -> str:
    if spread_mib is None:
        return "-"
    return f"{spread_mib:.1f}<sub>{gap_mib:.1f}</sub>"


def build_markdown(
    by_method_layer: dict[str, dict[int, list[tuple[float, float]]]],
    tau0_by_layer: dict[int, dict[tuple[Any, ...], float]],
    phibar_by_layer: dict[int, dict[tuple[Any, ...], float]],
    methods: tuple[str, ...],
    layers: tuple[int, ...],
    phi_to_bytes: float,
) -> str:
    header = "| Refine operator | " + " | ".join(str(layer) for layer in layers) + " |"
    separator = "|---|" + "---:|" * len(layers)
    lines = [header, separator]

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        cells = []
        for layer in layers:
            pairs = by_method_layer.get(method, {}).get(layer, [])
            spreads = [pair[0] for pair in pairs]
            gaps = [pair[1] for pair in pairs]
            spread_median = median_or_none(spreads)
            gap_median = median_or_none(gaps)
            if spread_median is None:
                cells.append("-")
                continue
            spread_mib = spread_median * phi_to_bytes / (1024**2)
            gap_mib = gap_median * phi_to_bytes / (1024**2)
            cells.append(format_cell(spread_mib, gap_mib))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    tau0_cells = []
    phibar_cells = []
    for layer in layers:
        tau0_values = list(tau0_by_layer.get(layer, {}).values())
        tau0_median = median_or_none(tau0_values)
        tau0_cells.append(
            "-" if tau0_median is None else f"{tau0_median * phi_to_bytes / (1024**2):.1f}"
        )
        phibar_values = list(phibar_by_layer.get(layer, {}).values())
        phibar_median = median_or_none(phibar_values)
        phibar_cells.append(
            "-"
            if phibar_median is None
            else f"{phibar_median * phi_to_bytes / (1024**3):.2f}"
        )
    lines.append("| *tau0 (MiB)* | " + " | ".join(f"*{c}*" for c in tau0_cells) + " |")
    lines.append(
        "| *per-rank load Phi-bar (GiB)* | " + " | ".join(f"*{c}*" for c in phibar_cells) + " |"
    )
    return "\n".join(lines) + "\n"


def build_csv_rows(
    by_method_layer: dict[str, dict[int, list[tuple[float, float]]]],
    tau0_by_layer: dict[int, dict[tuple[Any, ...], float]],
    phibar_by_layer: dict[int, dict[tuple[Any, ...], float]],
    methods: tuple[str, ...],
    layers: tuple[int, ...],
    phi_to_bytes: float,
) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        for layer in layers:
            pairs = by_method_layer.get(method, {}).get(layer, [])
            spreads = [pair[0] for pair in pairs]
            gaps = [pair[1] for pair in pairs]
            spread_median = median_or_none(spreads)
            gap_median = median_or_none(gaps)
            tau0_median = median_or_none(list(tau0_by_layer.get(layer, {}).values()))
            phibar_median = median_or_none(
                list(phibar_by_layer.get(layer, {}).values())
            )
            rows.append(
                {
                    "method": method,
                    "layers": layer,
                    "n_instances": len(pairs),
                    "median_spread_phi": spread_median,
                    "median_gap_phi": gap_median,
                    "median_spread_mib": (
                        None
                        if spread_median is None
                        else spread_median * phi_to_bytes / (1024**2)
                    ),
                    "median_gap_mib": (
                        None
                        if gap_median is None
                        else gap_median * phi_to_bytes / (1024**2)
                    ),
                    "tau0_median_mib": (
                        None
                        if tau0_median is None
                        else tau0_median * phi_to_bytes / (1024**2)
                    ),
                    "phibar_median_gib": (
                        None
                        if phibar_median is None
                        else phibar_median * phi_to_bytes / (1024**3)
                    ),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "depth-sweep":
        raise ValueError(
            f"expected a depth-sweep checkpoint, got experiment={payload.get('experiment')!r}"
        )
    records = payload["records"]
    methods = tuple(args.methods)
    layers = tuple(args.layers)
    phi_to_bytes = 3 * args.d_model * args.dtype_bytes

    by_method_layer, tau0_by_layer, phibar_by_layer = build_aggregates(
        records, methods, layers
    )
    markdown = build_markdown(
        by_method_layer, tau0_by_layer, phibar_by_layer, methods, layers, phi_to_bytes
    )
    markdown_path = args.markdown.expanduser().resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"markdown={markdown_path}")

    if args.csv is not None:
        import csv as csv_module

        rows = build_csv_rows(
            by_method_layer, tau0_by_layer, phibar_by_layer, methods, layers, phi_to_bytes
        )
        csv_path = args.csv.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv_module.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
