#!/usr/bin/env python3
"""Assemble the cross-model EP4 deployment efficiency table from batch=512
sweep_summary.json files (one row per implementation, one column pair per
model): Mem. = four-GPU max non-KV peak memory (GiB); Tput. = input tokens/s.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("Kimi", "kimi_vl_gqa_ep4"),
    ("Qwen3-VL-30B", "qwen3_gqa_ep4"),
    ("InternVL3.5", "internvl3_5_gqa_ep4"),
    ("Mistral-119B", "mistral_119b_gqa_ep4"),
    ("Qwen3-VL-235B", "qwen3_vl_235b_gqa_ep4"),
]

ROWS = [
    (0, "Default", "default"),
    (30, "Padded", "padded"),
    (30, "Multi-kernel", "multi_kernel"),
    (30, "Single-width", "single_width"),
    (30, "Ours (cross-layer)", "cross_layer"),
    (50, "Padded", "padded"),
    (50, "Multi-kernel", "multi_kernel"),
    (50, "Single-width", "single_width"),
    (50, "Ours (cross-layer)", "cross_layer"),
]


def load_point(model_dir: str, ratio: int, strategy: str) -> dict | None:
    path = (
        REPO_ROOT
        / "artifacts/efficiency_figure"
        / model_dir
        / "batch512_main"
        / f"prune_{ratio}"
        / "sweep_summary.json"
    )
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.get("best_by_strategy", []):
        if entry.get("strategy") == strategy:
            return entry
    return None


def format_cell(entry: dict | None) -> tuple[str, str]:
    if entry is None:
        return "-", "-"
    mem_gib = entry["best_max_non_kv_peak_memory_mib"] / 1024.0
    tput = entry["best_input_tokens_per_second"]
    return f"{mem_gib:.2f}", f"{tput:.0f}"


def main() -> int:
    header = ["p", "Batch size", "Implementation"]
    for model_name, _ in MODELS:
        header += [f"{model_name} Mem. (GiB)", f"{model_name} Tput. (tok/s)"]
    md_rows = [header]

    for ratio, label, strategy in ROWS:
        row = [f"0.{ratio:02d}" if ratio else "0", "512", label]
        for _, model_dir in MODELS:
            entry = load_point(model_dir, ratio, strategy)
            mem, tput = format_cell(entry)
            row += [mem, tput]
        md_rows.append(row)

    md_lines = []
    md_lines.append("| " + " | ".join(md_rows[0]) + " |")
    md_lines.append("|" + "|".join(["---:"] * len(md_rows[0])) + "|")
    for row in md_rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    markdown = "\n".join(md_lines) + "\n"

    latex_lines = []
    current_ratio = None
    for ratio, label, strategy in ROWS:
        cells = []
        for _, model_dir in MODELS:
            entry = load_point(model_dir, ratio, strategy)
            mem, tput = format_cell(entry)
            cells += [mem, tput]
        ratio_label = "0" if ratio == 0 else f"{ratio / 100:.1f}"
        prefix = ""
        if ratio != current_ratio:
            if current_ratio is not None:
                latex_lines.append(r"\midrule")
            if ratio == 0:
                prefix = "0"
            else:
                prefix = rf"\multirow{{4}}{{*}}{{{ratio_label}}}"
            current_ratio = ratio
        row_label = label if ratio != 0 else "Default"
        row_prefix = r"\rowcolor{ours} " if strategy == "cross_layer" else " "
        display_label = rf"\textit{{{row_label}}}" if strategy == "cross_layer" else row_label
        latex_lines.append(
            f"{row_prefix}{prefix} & {display_label} & "
            + " & ".join(cells)
            + r" \\"
        )
    latex = "\n".join(latex_lines) + "\n"

    out_dir = REPO_ROOT / "artifacts/efficiency_figure/main_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main_table.md").write_text(markdown, encoding="utf-8")
    (out_dir / "main_table_rows.tex").write_text(latex, encoding="utf-8")
    print(markdown)
    print(f"saved={out_dir}/main_table.md")
    print(f"saved={out_dir}/main_table_rows.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
