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
    pick_plot_layer,
    plot_expert_channel_bars,
    plot_heatmap,
    print_saved_artifact_message,
    save_tensor_dict,
    stack_layer_tensors,
)


def main():
    parser = build_base_arg_parser("O2: Expert channel modality response analysis.")
    parser.add_argument("--top_experts", type=int, default=4)
    parser.add_argument("--plot_layer", type=int, default=-1)
    parser.add_argument("--high_bias_threshold", type=float, default=0.5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    raw_stats_path = args.raw_stats_path or os.path.join(args.output_dir, "raw_stats.pt")
    if os.path.exists(raw_stats_path) and not args.force_recompute:
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
    print("[O2说明] ModalBias 热力图：横轴是 expert id，纵轴是 layer id。")
    print("[O2说明] 每个格子的值是该 expert 内所有通道的平均模态偏置强度。")
    print("[O2说明] 数值越大，表示这个 expert 内部的通道对 text / visual 的响应差异越明显；越接近 0，表示通道响应更均衡。")

    plot_layer = args.plot_layer if args.plot_layer >= 0 else pick_plot_layer(raw_stats)
    layer_total = raw_stats["channel_count"][plot_layer]["text"] + raw_stats["channel_count"][plot_layer]["visual"]
    top_experts = torch.topk(
        layer_total,
        k=min(args.top_experts, int(layer_total.numel())),
    ).indices.tolist()
    plot_expert_channel_bars(
        channel_response[plot_layer]["text"],
        channel_response[plot_layer]["visual"],
        plot_layer,
        top_experts,
        os.path.join(args.output_dir, f"channel_response_layer_{plot_layer}.png"),
    )
    print(
        f"[O2说明] 通道响应折线图选择了 layer={plot_layer} 的 experts={top_experts}。"
        " 每条曲线展示该 expert 内各通道的平均 |activation|。"
    )
    print("[O2说明] 如果同一 expert 中 text 曲线和 visual 曲线明显分离，说明通道存在模态专属性。")

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
        "selected_plot_layer": int(plot_layer),
        "selected_plot_experts": top_experts,
        "opposite_modality_high_bias_cells": int(sum(opposite_modality_counts)),
        "heatmap_readme": {
            "what_is_row": "layer id",
            "what_is_column": "expert id",
            "larger_value_means": "stronger channel-level modality asymmetry inside the expert",
            "smaller_value_means": "more balanced channel responses between text and visual tokens",
        },
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
    print_saved_artifact_message(os.path.join(args.output_dir, "modal_bias_heatmap.png"), "ModalBias 热力图")
    print_saved_artifact_message(
        os.path.join(args.output_dir, f"channel_response_layer_{plot_layer}.png"),
        "通道响应对比图",
    )
    print_saved_artifact_message(metrics_path, "O2 指标张量")
    summary_path = os.path.join(args.output_dir, "summary.json")
    dump_json(summary_path, summary)
    print_saved_artifact_message(summary_path, "O2 汇总说明")


if __name__ == "__main__":
    main()
