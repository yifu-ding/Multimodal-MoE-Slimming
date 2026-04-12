import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for path in (REPO_PARENT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from observations.common import (
    build_base_arg_parser,
    collect_observation_stats,
    compute_ema,
    compute_routing_freq,
    dump_json,
    ensure_dir,
    plot_heatmap,
    print_saved_artifact_message,
    save_tensor_dict,
    stack_layer_tensors,
)


def main():
    parser = build_base_arg_parser("O1: Expert modality affinity analysis.")
    parser.add_argument("--ema_threshold", type=float, default=0.5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    raw_stats_path = args.raw_stats_path or os.path.join(args.output_dir, "raw_stats.pt")
    if os.path.exists(raw_stats_path) and not args.force:
        print(f"[O1] 检测到已有缓存，直接加载统计结果: {raw_stats_path}", flush=True)
        raw_stats = torch.load(raw_stats_path, weights_only=False)
    else:
        print("[O1] 未命中缓存，开始重新收集 routing 统计。", flush=True)
        raw_stats = collect_observation_stats(args)
        save_tensor_dict(raw_stats_path, raw_stats)
        print_saved_artifact_message(raw_stats_path, "原始统计张量")

    routing_freq = compute_routing_freq(raw_stats)
    ema = compute_ema(routing_freq)

    layers = raw_stats["layers"]
    ema_matrix = stack_layer_tensors(ema, layers)
    routing_text = stack_layer_tensors(
        {layer: routing_freq[layer]["text"] for layer in layers}, layers
    )
    routing_visual = stack_layer_tensors(
        {layer: routing_freq[layer]["visual"] for layer in layers}, layers
    )

    plot_heatmap(
        ema_matrix,
        "O1 Expert Modality Affinity (EMA)",
        os.path.join(args.output_dir, "ema_heatmap.png"),
    )
    print("[O1说明] EMA 热力图：横轴是 expert id，纵轴是 layer id。", flush=True)
    print(
        "[O1说明] 数值接近 +1，表示该 expert 更偏视觉 token；接近 -1，表示更偏文本 token；接近 0，表示两种模态路由频率接近。",
        flush=True,
    )
    print(
        "[O1说明] 这里的值来自 (visual_freq - text_freq) / (visual_freq + text_freq)。因此绝对值越大，模态偏好越强。",
        flush=True,
    )

    abs_ema = ema_matrix.abs()
    strong_mask = abs_ema > args.ema_threshold
    summary = {
        "experiment": "O1",
        "model_name_or_path": raw_stats["model_name_or_path"],
        "dataset": raw_stats["dataset"],
        "num_samples": raw_stats["num_samples"],
        "ema_threshold": args.ema_threshold,
        "num_layers": len(layers),
        "num_strong_affinity_cells": int(strong_mask.sum().item()),
        "strong_affinity_ratio": float(strong_mask.double().mean().item()),
        "max_abs_ema": float(abs_ema.max().item()),
        "mean_abs_ema": float(abs_ema.mean().item()),
        "heatmap_readme": {
            "what_is_row": "layer id",
            "what_is_column": "expert id",
            "large_positive_means": "visual-preferring expert",
            "large_negative_means": "text-preferring expert",
            "near_zero_means": "balanced routing between text and visual tokens",
        },
    }

    metrics_path = os.path.join(args.output_dir, "o1_metrics.pt")
    save_tensor_dict(
        metrics_path,
        {
            "layers": layers,
            "routing_freq_text": routing_text,
            "routing_freq_visual": routing_visual,
            "ema": ema_matrix,
        },
    )
    print_saved_artifact_message(os.path.join(args.output_dir, "ema_heatmap.png"), "EMA 热力图")
    print_saved_artifact_message(metrics_path, "O1 指标张量")
    summary_path = os.path.join(args.output_dir, "summary.json")
    dump_json(summary_path, summary)
    print_saved_artifact_message(summary_path, "O1 汇总说明")


if __name__ == "__main__":
    main()
