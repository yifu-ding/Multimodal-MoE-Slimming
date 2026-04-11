"""
绘制 storage/prune/scores/kimi_gqa/affinity.pt 的 EMA affinity 热力图。

affinity.pt 结构：
  {
    "affinity": {layer_idx: {expert_idx: float, ...}, ...},
    "threshold": float
  }

用法：
  python observations/o1/scripts/plot_affinity.py \
      --affinity_path storage/prune/scores/kimi_gqa/affinity.pt \
      --output_dir storage/prune/scores/kimi_gqa/plots
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from observations.common import ensure_dir, plot_heatmap


def affinity_dict_to_matrix(affinity: dict) -> tuple[torch.Tensor, list]:
    """将 {layer: {expert: value}} 转成 [num_layers, num_experts] 矩阵。"""
    layers = sorted(affinity.keys())
    num_experts = max(len(affinity[l]) for l in layers)
    matrix = torch.zeros(len(layers), num_experts)
    for row, layer in enumerate(layers):
        for expert_idx, value in affinity[layer].items():
            matrix[row, expert_idx] = value
    return matrix, layers


def main():
    parser = argparse.ArgumentParser(description="绘制 affinity.pt 热力图")
    parser.add_argument(
        "--affinity_path",
        type=str,
        default="storage/prune/scores/kimi_gqa/affinity.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="storage/prune/scores/kimi_gqa/plots",
    )
    args = parser.parse_args()

    payload = torch.load(args.affinity_path, map_location="cpu", weights_only=False)
    affinity = payload["affinity"]
    threshold = payload.get("threshold", 0.9)

    matrix, layers = affinity_dict_to_matrix(affinity)
    num_layers, num_experts = matrix.shape
    print(f"[plot_affinity] layers={num_layers} ({layers[0]}~{layers[-1]}), experts={num_experts}")
    print(f"[plot_affinity] affinity range: [{matrix.min():.3f}, {matrix.max():.3f}]")
    print(f"[plot_affinity] threshold={threshold}")

    # 统计
    vis_only = (matrix > threshold).sum().item()
    txt_only = (matrix < -threshold).sum().item()
    print(f"[plot_affinity] visual-only experts (>{threshold}): {vis_only}")
    print(f"[plot_affinity] text-only experts (<-{threshold}): {txt_only}")

    ensure_dir(args.output_dir)

    # 1. EMA affinity 全图（coolwarm: 红=visual, 蓝=text）
    plot_heatmap(
        matrix,
        f"Expert Modality Affinity  [layers {layers[0]}~{layers[-1]}, {num_experts} experts]",
        os.path.join(args.output_dir, "affinity_heatmap.png"),
        cmap="coolwarm",
    )

    # 2. 绝对值图（显示模态偏好强度，不区分方向）
    plot_heatmap(
        matrix.abs(),
        f"Expert Modality Affinity |abs|  [threshold={threshold}]",
        os.path.join(args.output_dir, "affinity_abs_heatmap.png"),
        cmap="Reds",
    )


if __name__ == "__main__":
    main()
