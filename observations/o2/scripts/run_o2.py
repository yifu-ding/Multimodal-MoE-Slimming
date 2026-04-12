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
    compute_channel_response,
    compute_ema,
    compute_modal_bias,
    compute_routing_freq,
    dump_json,
    ensure_dir,
    pick_top_experts_global,
    plot_expert_channel_lines_per_expert,
    plot_heatmap,
    print_saved_artifact_message,
    save_tensor_dict,
    stack_layer_tensors,
)


def main():
    parser = build_base_arg_parser("O2: Expert channel modality response analysis.")
    parser.add_argument("--top_experts", type=int, default=10)
    parser.add_argument("--min_count", type=int, default=5,
                        help="过滤掉任一模态 token 数 < min_count 的 expert，避免低样本噪声。")
    parser.add_argument("--sort_by", type=str, default="ema_abs",
                        choices=["modal_bias", "ema_abs"],
                        help="expert 选取排序依据：ema_abs 按模态偏好强度，modal_bias 按通道差异强度。")
    parser.add_argument("--high_bias_threshold", type=float, default=0.5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    raw_stats_path = args.raw_stats_path or os.path.join(args.output_dir, "raw_stats.pt")
    if os.path.exists(raw_stats_path) and not args.force:
        print(f"[O2] 检测到已有缓存，直接加载统计结果: {raw_stats_path}")
        raw_stats = torch.load(raw_stats_path, weights_only=False)
    else:
        print("[O2] 未命中缓存，开始重新收集 routing 与 channel 响应统计。")
        raw_stats = collect_observation_stats(args)
        save_tensor_dict(raw_stats_path, raw_stats)
        print_saved_artifact_message(raw_stats_path, "原始统计张量")

    routing_freq = compute_routing_freq(raw_stats)
    ema = compute_ema(routing_freq)
    channel_response = compute_channel_response(raw_stats)
    modal_bias = compute_modal_bias(channel_response)
    layers = raw_stats["layers"]

    modal_bias_matrix = stack_layer_tensors(
        {layer: modal_bias[layer].mean(dim=-1) for layer in layers}, layers
    )
    plot_heatmap(
        modal_bias_matrix,
        "O2 Mean ModalBias per Expert",
        os.path.join(args.output_dir, "modal_bias_heatmap.png"),
        cmap="viridis",
    )

    # 全模型选 top-K 模态偏置最强的 expert，每个单独出一张折线图
    top_candidates = pick_top_experts_global(
        modal_bias, raw_stats, top_k=args.top_experts, min_count=args.min_count,
        ema=ema, sort_by=args.sort_by,
    )
    print(f"[O2] 全模型 Top-{args.top_experts} expert (sort_by={args.sort_by}):")
    for layer_idx, expert_idx, bias_val in top_candidates:
        n_text = int(raw_stats["channel_count"][layer_idx]["text"][expert_idx].item())
        n_visual = int(raw_stats["channel_count"][layer_idx]["visual"][expert_idx].item())
        print(
            f"  Layer {layer_idx:3d}  Expert {expert_idx:3d}"
            f"  mean_ModalBias={bias_val:.4f}"
            f"  (n_text={n_text}, n_visual={n_visual})"
        )

    saved_paths = plot_expert_channel_lines_per_expert(
        channel_response, top_candidates, args.output_dir, ema=ema
    )
    for p in saved_paths:
        print_saved_artifact_message(p, "通道响应折线图")

    opposite_modality_counts = []
    for layer in layers:
        layer_ema = ema[layer]
        layer_bias = modal_bias[layer].mean(dim=-1)
        text_pref_visual_bias = ((layer_ema < 0) & (layer_bias > args.high_bias_threshold)).sum().item()
        visual_pref_text_bias = ((layer_ema > 0) & (layer_bias > args.high_bias_threshold)).sum().item()
        opposite_modality_counts.append(text_pref_visual_bias + visual_pref_text_bias)

    summary = {
        "experiment": "O2",
        "model_name_or_path": raw_stats["model_name_or_path"],
        "dataset": raw_stats["dataset"],
        "num_samples": raw_stats["num_samples"],
        "high_bias_threshold": args.high_bias_threshold,
        "mean_modal_bias": float(modal_bias_matrix.mean().item()),
        "max_modal_bias": float(modal_bias_matrix.max().item()),
        "top_experts": [
            {
                "layer": layer_idx,
                "expert": expert_idx,
                "mean_modal_bias": round(bias_val, 6),
                "n_text": int(raw_stats["channel_count"][layer_idx]["text"][expert_idx].item()),
                "n_visual": int(raw_stats["channel_count"][layer_idx]["visual"][expert_idx].item()),
            }
            for layer_idx, expert_idx, bias_val in top_candidates
        ],
        "opposite_modality_high_bias_cells": int(sum(opposite_modality_counts)),
    }

    metrics_path = os.path.join(args.output_dir, "o2_metrics.pt")
    save_tensor_dict(
        metrics_path,
        {
            "layers": layers,
            "channel_response": channel_response,
            "modal_bias": modal_bias,
            "mean_modal_bias_per_expert": modal_bias_matrix,
        },
    )
    print_saved_artifact_message(metrics_path, "O2 指标张量")
    summary_path = os.path.join(args.output_dir, "summary.json")
    dump_json(summary_path, summary)
    print_saved_artifact_message(summary_path, "O2 汇总说明")


if __name__ == "__main__":
    main()
