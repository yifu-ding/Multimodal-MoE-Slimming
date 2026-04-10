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
    compute_conflict_scores,
    compute_modal_bias,
    compute_routing_freq,
    dump_json,
    ensure_dir,
    plot_heatmap,
    print_saved_artifact_message,
    save_tensor_dict,
    stack_layer_tensors,
)


def main():
    parser = build_base_arg_parser("O3: Conflict score analysis.")
    parser.add_argument("--high_conflict_threshold", type=float, default=0.5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    raw_stats_path = args.raw_stats_path or os.path.join(args.output_dir, "raw_stats.pt")
    if os.path.exists(raw_stats_path) and not args.force_recompute:
        print(f"[O3] 检测到已有缓存，直接加载统计结果: {raw_stats_path}")
        raw_stats = torch.load(raw_stats_path, weights_only=False)
    else:
        print("[O3] 未命中缓存，开始重新收集 routing 与 channel 冲突统计。")
        raw_stats = collect_observation_stats(args)
        save_tensor_dict(raw_stats_path, raw_stats)
        print_saved_artifact_message(raw_stats_path, "原始统计张量")

    routing_freq = compute_routing_freq(raw_stats)
    channel_response = compute_channel_response(raw_stats)
    modal_bias = compute_modal_bias(channel_response)
    conflict_scores, mean_modal_bias = compute_conflict_scores(
        raw_stats, routing_freq, modal_bias, channel_response
    )
    layers = raw_stats["layers"]

    for modality in ("text", "visual"):
        conflict_matrix = stack_layer_tensors(
            {layer: conflict_scores[layer][modality] for layer in layers}, layers
        )
        plot_heatmap(
            conflict_matrix,
            f"O3 Conflict Score ({modality})",
            os.path.join(args.output_dir, f"conflict_heatmap_{modality}.png"),
            cmap="magma",
        )
        print(
            f"[O3说明] {modality} conflict 热力图：横轴是 expert id，纵轴是 layer id。"
            " 数值越大，表示“router 不愿把该模态 token 分给这个 expert，但该 expert 内部仍有对该模态明显响应的通道”这一冲突越强。"
        )
        print(
            f"[O3说明] {modality} conflict 值越接近 0，表示要么 router 并不回避该 expert，"
            "要么该 expert 内部对该模态没有明显的额外响应通道。"
        )

    text_conflict = stack_layer_tensors(
        {layer: conflict_scores[layer]["text"] for layer in layers}, layers
    )
    visual_conflict = stack_layer_tensors(
        {layer: conflict_scores[layer]["visual"] for layer in layers}, layers
    )
    all_conflict = torch.cat([text_conflict.reshape(-1), visual_conflict.reshape(-1)], dim=0)

    summary = {
        "experiment": "O3",
        "model_name_or_path": raw_stats["model_name_or_path"],
        "dataset": raw_stats["dataset"],
        "num_samples": raw_stats["num_samples"],
        "high_conflict_threshold": args.high_conflict_threshold,
        "max_conflict_score": float(all_conflict.max().item()),
        "mean_conflict_score": float(all_conflict.mean().item()),
        "high_conflict_ratio": float(
            (all_conflict > args.high_conflict_threshold).double().mean().item()
        ),
        "heatmap_readme": {
            "what_is_row": "layer id",
            "what_is_column": "expert id",
            "larger_value_means": "stronger router-channel conflict / more likely ineffective capacity",
            "smaller_value_means": "weaker conflict",
        },
    }

    metrics_path = os.path.join(args.output_dir, "o3_metrics.pt")
    save_tensor_dict(
        metrics_path,
        {
            "layers": layers,
            "conflict_scores": conflict_scores,
            "mean_modal_bias": mean_modal_bias,
        },
    )
    print_saved_artifact_message(os.path.join(args.output_dir, "conflict_heatmap_text.png"), "text conflict 热力图")
    print_saved_artifact_message(os.path.join(args.output_dir, "conflict_heatmap_visual.png"), "visual conflict 热力图")
    print_saved_artifact_message(metrics_path, "O3 指标张量")
    summary_path = os.path.join(args.output_dir, "summary.json")
    dump_json(summary_path, summary)
    print_saved_artifact_message(summary_path, "O3 汇总说明")


if __name__ == "__main__":
    main()
