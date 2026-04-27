import argparse
import copy
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


def _model_artifact_tag(model_name_or_path: str) -> str:
    return os.path.basename(model_name_or_path.rstrip("/")).replace(".", "_")


def _artifact_basename(dataset: str, base_args: argparse.Namespace) -> str:
    model_tag = _model_artifact_tag(base_args.model_name_or_path)
    suffix = (getattr(base_args, "artifact_suffix", None) or "").strip()
    stem = f"{dataset}_{model_tag}"
    return f"{stem}_{suffix}" if suffix else stem


def _resolve_raw_stats_path(
    base_args: argparse.Namespace,
    dataset: str,
    n_datasets: int,
) -> str:
    if n_datasets > 1 and base_args.raw_stats_path:
        raise SystemExit(
            "与 --datasets 多数据集同时使用时不能指定 --raw_stats_path（会对缓存路径产生歧义）。"
            "请去掉 --raw_stats_path，将使用 {output_dir}/raw_stats_<dataset>_<model>[_<suffix>].pt。"
        )
    if n_datasets == 1 and base_args.raw_stats_path:
        return base_args.raw_stats_path
    stem = _artifact_basename(dataset, base_args)
    return os.path.join(base_args.output_dir, f"raw_stats_{stem}.pt")


def run_o2_for_dataset(
    base_args: argparse.Namespace,
    dataset: str,
    n_datasets: int,
) -> None:
    args = copy.copy(base_args)
    args.dataset = dataset
    ensure_dir(args.output_dir)
    stem = _artifact_basename(dataset, base_args)
    run_output_dir = os.path.join(args.output_dir, stem)
    ensure_dir(run_output_dir)

    raw_stats_path = _resolve_raw_stats_path(base_args, dataset, n_datasets)
    if os.path.exists(raw_stats_path) and not args.force:
        print(f"[O2] [{dataset}] 检测到已有缓存，直接加载统计结果: {raw_stats_path}")
        raw_stats = torch.load(raw_stats_path, weights_only=False)
    else:
        print(f"[O2] [{dataset}] 未命中缓存，开始重新收集 routing 与 channel 响应统计。")
        raw_stats = collect_observation_stats(args)
        save_tensor_dict(raw_stats_path, raw_stats)
        print_saved_artifact_message(raw_stats_path, f"原始统计张量 ({dataset})")

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
        f"O2 Mean ModalBias per Expert — {dataset}",
        os.path.join(run_output_dir, f"modal_bias_heatmap_{stem}.png"),
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
        channel_response, top_candidates, run_output_dir, ema=ema
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

    metrics_path = os.path.join(run_output_dir, f"o2_metrics_{stem}.pt")
    save_tensor_dict(
        metrics_path,
        {
            "layers": layers,
            "channel_response": channel_response,
            "modal_bias": modal_bias,
            "mean_modal_bias_per_expert": modal_bias_matrix,
        },
    )
    print_saved_artifact_message(metrics_path, f"O2 指标张量 ({stem})")
    summary_path = os.path.join(run_output_dir, f"summary_{stem}.json")
    dump_json(summary_path, summary)
    print_saved_artifact_message(summary_path, f"O2 汇总说明 ({stem})")


def main():
    parser = build_base_arg_parser("O2: Expert channel modality response analysis.")
    parser.add_argument("--top_experts", type=int, default=10)
    parser.add_argument("--min_count", type=int, default=5,
                        help="过滤掉任一模态 token 数 < min_count 的 expert，避免低样本噪声。")
    parser.add_argument("--sort_by", type=str, default="ema_abs",
                        choices=["modal_bias", "ema_abs"],
                        help="expert 选取排序依据：ema_abs 按模态偏好强度，modal_bias 按通道差异强度。")
    parser.add_argument("--high_bias_threshold", type=float, default=0.5)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help="可指定多个数据集依次各跑一轮并写出各自动态产物，如: gqa coco。未设置时只使用 --dataset 跑一轮。",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="非空时写入文件名：raw_stats / modal_bias_heatmap / o2_metrics / summary 均为 <dataset>_<model>_<suffix>。",
    )
    args = parser.parse_args()

    if args.datasets is not None:
        dataset_list = list(args.datasets)
    else:
        dataset_list = [args.dataset]

    n = len(dataset_list)
    for dataset in dataset_list:
        run_o2_for_dataset(args, dataset, n)


if __name__ == "__main__":
    main()
