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

# EMA 热力图 colormap（matplotlib 命名；PRGn: 紫-绿分歧色图，适合有正有负的 EMA）
EMA_HEATMAP_CMAP = "PRGn"

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


def _dataset_artifact_basename(dataset: str, base_args: argparse.Namespace) -> str:
    """文件名中的 dataset 段：可带 --artifact-suffix，如 gqa_128sample。"""
    tag = (getattr(base_args, "artifact_suffix", None) or "").strip()
    return f"{dataset}_{tag}" if tag else dataset


def _resolve_raw_stats_path(
    base_args: argparse.Namespace, dataset: str, n_datasets: int
) -> str:
    if n_datasets > 1 and base_args.raw_stats_path:
        raise SystemExit(
            "与 --datasets 多数据集同时使用时不能指定 --raw_stats_path（会对缓存路径产生歧义）。"
            "请去掉 --raw_stats_path，将使用 {output_dir}/raw_stats_<dataset>[_<suffix>].pt。"
        )
    if n_datasets == 1 and base_args.raw_stats_path:
        return base_args.raw_stats_path
    stem = _dataset_artifact_basename(dataset, base_args)
    return os.path.join(base_args.output_dir, f"raw_stats_{stem}.pt")


def run_o1_for_dataset(
    base_args: argparse.Namespace,
    dataset: str,
    n_datasets: int,
) -> None:
    args = copy.copy(base_args)
    args.dataset = dataset
    ensure_dir(args.output_dir)

    raw_stats_path = _resolve_raw_stats_path(base_args, dataset, n_datasets)
    if os.path.exists(raw_stats_path) and not args.force:
        print(f"[O1] [{dataset}] 检测到已有缓存，直接加载: {raw_stats_path}", flush=True)
        raw_stats = torch.load(raw_stats_path, weights_only=False)
    else:
        print(f"[O1] [{dataset}] 未命中缓存，开始重新收集 routing 统计。", flush=True)
        raw_stats = collect_observation_stats(args)
        save_tensor_dict(raw_stats_path, raw_stats)
        print_saved_artifact_message(raw_stats_path, f"原始统计张量 ({dataset})")

    if "topk_routing_counts" in raw_stats:
        counts_key = "topk_routing_counts"
    else:
        print(
            "[O1] 未找到 topk_routing_counts, 对旧版缓存回退为 routing_counts（全槽位）统计。",
            flush=True,
        )
        counts_key = "routing_counts"
    routing_freq = compute_routing_freq(raw_stats, counts_key=counts_key)
    ema = compute_ema(routing_freq)

    layers = raw_stats["layers"]
    ema_matrix = stack_layer_tensors(ema, layers)
    routing_text = stack_layer_tensors(
        {layer: routing_freq[layer]["text"] for layer in layers}, layers
    )
    routing_visual = stack_layer_tensors(
        {layer: routing_freq[layer]["visual"] for layer in layers}, layers
    )
    router_k = int(raw_stats.get("router_topk", getattr(args, "router_topk", 8)))
    if "router_logits_sum" in raw_stats:
        router_logits_text = stack_layer_tensors(
            {layer: raw_stats["router_logits_sum"][layer]["text"] for layer in layers},
            layers,
        )
        router_logits_visual = stack_layer_tensors(
            {layer: raw_stats["router_logits_sum"][layer]["visual"] for layer in layers},
            layers,
        )
    else:
        router_logits_text = None
        router_logits_visual = None
        print(
            "[O1] 未找到 router_logits_sum（旧版缓存无此项）。",
            flush=True,
        )

    name_stem = _dataset_artifact_basename(dataset, base_args)
    heatmap_path = os.path.join(args.output_dir, f"ema_heatmap_{name_stem}.png")
    plot_heatmap(
        ema_matrix,
        f"O1 Expert Modality Affinity (EMA) — {dataset} (top-{router_k} 槽位路由 / token)",
        heatmap_path,
        cmap=EMA_HEATMAP_CMAP,
    )
    print(f"[O1说明] [{dataset}] EMA 热力图：横轴是 expert id，纵轴是 layer id。", flush=True)
    print(
        "[O1说明] 数值接近 +1，表示该 expert 更偏视觉 token；接近 -1，表示更偏文本 token；接近 0，表示两种模态路由频率接近。",
        flush=True,
    )
    print(
        f"[O1说明] 路由项：仅将各层 router 的 top-k 中前 {router_k} 个槽位上的命中计入计数；"
        f"EMA = (Vfreq - Tfreq) / (Vfreq + Tfreq)，其中 Vfreq/Tfreq 为上述 top-{router_k} 计数 / 各模态 token 数。",
        flush=True,
    )
    print(
        "[O1说明] router_logits 项：为 gate 线性层输出的未 softmax 的 logits，在 expert 维上对该模态所有 token 求和。",
        flush=True,
    )

    abs_ema = ema_matrix.abs()
    strong_mask = abs_ema > args.ema_threshold
    summary = {
        "experiment": "O1",
        "model_name_or_path": raw_stats["model_name_or_path"],
        "dataset": raw_stats["dataset"],
        "artifact_file_suffix": getattr(base_args, "artifact_suffix", "") or None,
        "num_samples": raw_stats["num_samples"],
        "router_topk": router_k,
        "routing_freq_counts_key": counts_key,
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

    metrics_path = os.path.join(args.output_dir, f"o1_metrics_{name_stem}.pt")
    metrics_out = {
        "layers": layers,
        "router_topk": router_k,
        "routing_freq_counts_key": counts_key,
        "routing_freq_text": routing_text,
        "routing_freq_visual": routing_visual,
        "ema": ema_matrix,
    }
    if router_logits_text is not None:
        metrics_out["router_logits_sum_text"] = router_logits_text
        metrics_out["router_logits_sum_visual"] = router_logits_visual
    save_tensor_dict(metrics_path, metrics_out)
    print_saved_artifact_message(heatmap_path, f"EMA 热力图 ({name_stem})")
    print_saved_artifact_message(metrics_path, f"O1 指标张量 ({name_stem})")
    summary_path = os.path.join(args.output_dir, f"summary_{name_stem}.json")
    dump_json(summary_path, summary)
    print_saved_artifact_message(summary_path, f"O1 汇总说明 ({name_stem})")


def main():
    parser = build_base_arg_parser("O1: Expert modality affinity analysis.")
    parser.add_argument("--ema_threshold", type=float, default=0.5)
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
        help="非空时写入文件名：raw_stats / ema_heatmap / o1_metrics / summary 均为 <dataset>_<suffix>，例如 128sample。",
    )
    args = parser.parse_args()

    if args.datasets is not None:
        dataset_list = list(args.datasets)
    else:
        dataset_list = [args.dataset]

    n = len(dataset_list)
    for dataset in dataset_list:
        run_o1_for_dataset(args, dataset, n)


if __name__ == "__main__":
    main()
