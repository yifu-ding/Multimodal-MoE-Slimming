"""从教师模型 hidden 缓存中优化出一小份合成 hidden, 用于表征级校准.

训练目标: 用 MMD, 协方差, 多样性等分布损失, 辅以均值与方差矩匹配, 使合成样本在统计上逼近教师缓存.
"""
import argparse
import math
import os
import sys

# 将仓库根与父目录加入 sys.path, 以便以包形式导入 src.*
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from tqdm import trange

from src.calibration.representation_distill.common import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    ensure_dir,
    seed_everything,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Helpers (行采样与下采样等工具函数)
# ---------------------------------------------------------------------------

def _sample_rows(tensor: torch.Tensor, count: int) -> torch.Tensor:
    # 从首维随机抽 count 行, 若不足则全取
    if count >= tensor.shape[0]:
        return tensor
    indices = torch.randint(0, tensor.shape[0], (count,), device=tensor.device)
    return tensor.index_select(0, indices)


def _subsample_flat(tensor: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Subsample rows for efficient kernel computation."""
    # 对行做随机下采样, 控制核矩阵规模, 降低 MMD 等计算开销
    if tensor.shape[0] <= max_tokens:
        return tensor
    indices = torch.randperm(tensor.shape[0], device=tensor.device)[:max_tokens]
    return tensor[indices]


# ---------------------------------------------------------------------------
# Loss functions (MMD, 协方差, 多样性, 矩匹配)
# ---------------------------------------------------------------------------

def _mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidth_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> torch.Tensor:
    """Unbiased MMD^2 estimate with multi-bandwidth RBF kernel.

    Parameters
    ----------
    X, Y : (N, D) and (M, D) token-level feature matrices.
    bandwidth_multipliers : scales applied to the median pairwise distance.
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.tensor(0.0, device=X.device)

    # 成对欧氏距离平方, 用于 RBF 核
    XX = torch.cdist(X, X).pow(2)
    YY = torch.cdist(Y, Y).pow(2)
    XY = torch.cdist(X, Y).pow(2)

    # 带宽: 用全体距离的中位数启发式, 再乘多组 multiplier
    with torch.no_grad():
        all_dists = torch.cat([XX.view(-1), YY.view(-1), XY.view(-1)])
        median_dist = all_dists.median().clamp(min=1e-6)

    # 多带宽 RBF 下无偏 MMD^2 估计, 各带宽结果相加
    mmd = torch.tensor(0.0, device=X.device)
    for mult in bandwidth_multipliers:
        bw_sq = 2.0 * (mult * median_dist)
        k_XX = (-XX / bw_sq).exp().mean()
        k_YY = (-YY / bw_sq).exp().mean()
        k_XY = (-XY / bw_sq).exp().mean()
        mmd = mmd + k_XX + k_YY - 2.0 * k_XY
    return mmd


def _cov_loss(teacher_flat: torch.Tensor, synth_flat: torch.Tensor) -> torch.Tensor:
    """MSE between cross-channel covariance matrices."""
    # 通道维协方差矩阵的逐元素 MSE, 对齐二阶结构
    t_centered = teacher_flat - teacher_flat.mean(dim=0, keepdim=True)
    s_centered = synth_flat - synth_flat.mean(dim=0, keepdim=True)
    t_cov = (t_centered.T @ t_centered) / max(teacher_flat.shape[0] - 1, 1)
    s_cov = (s_centered.T @ s_centered) / max(synth_flat.shape[0] - 1, 1)
    return (t_cov - s_cov).pow(2).mean()


def _diversity_loss(synthetic_flat: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity (lower = more diverse)."""
    if synthetic_flat.shape[0] <= 1:
        return torch.tensor(0.0, device=synthetic_flat.device)
    normed = F.normalize(synthetic_flat, dim=-1)
    sim = normed @ normed.T
    n = sim.shape[0]
    # 非对角线余弦相似度均值, 越小表示样本间越分散
    off_diag = sim.masked_select(~torch.eye(n, dtype=torch.bool, device=sim.device))
    return off_diag.mean()


def _compute_stat_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Legacy moment-matching losses (kept as auxiliaries)."""
    # 展平为 (token 数, hidden_dim), 匹配一阶与二阶矩
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    teacher_mean = teacher_flat.mean(dim=0)
    synth_mean = synth_flat.mean(dim=0)
    teacher_var = teacher_flat.var(dim=0, unbiased=False)
    synth_var = synth_flat.var(dim=0, unbiased=False)

    return {
        "mean": (teacher_mean - synth_mean).pow(2).mean(),
        "var": (teacher_var - synth_var).pow(2).mean(),
    }


def _compute_distribution_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    teacher_labels: torch.Tensor | None,
    synth_labels: torch.Tensor | None,
    mmd_subsample: int,
) -> dict[str, torch.Tensor]:
    """Distribution-aware losses: MMD (modality-conditioned), covariance, diversity."""
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    losses: dict[str, torch.Tensor] = {}

    # 按模态分别算 MMD, 再对参与的模态取平均, 若无标签则退化为全局 MMD
    if teacher_labels is not None and synth_labels is not None:
        teacher_labels_flat = teacher_labels.reshape(-1) if teacher_labels is not None else None
        synth_labels_flat = synth_labels.reshape(-1) if synth_labels is not None else None

        mmd_total = torch.tensor(0.0, device=teacher_flat.device)
        n_modalities = 0
        for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
            t_mask = teacher_labels_flat == mod_id
            s_mask = synth_labels_flat == mod_id
            if t_mask.any() and s_mask.any():
                t_sub = _subsample_flat(teacher_flat[t_mask], mmd_subsample)
                s_sub = _subsample_flat(synth_flat[s_mask], mmd_subsample)
                mmd_total = mmd_total + _mmd_rbf(t_sub, s_sub)
                n_modalities += 1
        if n_modalities > 0:
            losses["mmd"] = mmd_total / n_modalities
        else:
            losses["mmd"] = _mmd_rbf(
                _subsample_flat(teacher_flat, mmd_subsample),
                _subsample_flat(synth_flat, mmd_subsample),
            )
    else:
        losses["mmd"] = _mmd_rbf(
            _subsample_flat(teacher_flat, mmd_subsample),
            _subsample_flat(synth_flat, mmd_subsample),
        )

    # 教师侧下采样以控内存, 合成侧用全部 token 与协方差项
    losses["cov"] = _cov_loss(
        _subsample_flat(teacher_flat, mmd_subsample),
        synth_flat,
    )

    # 对合成 hidden 做多样性正则, 同样经下采样控制规模
    losses["div"] = _diversity_loss(_subsample_flat(synth_flat, mmd_subsample))

    return losses


# ---------------------------------------------------------------------------
# CLI (命令行参数)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill a small synthetic hidden calibration set from teacher hidden cache."
    )
    parser.add_argument("--teacher_cache_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--synthetic_size", type=int, default=256)
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_std", type=float, default=1e-3)
    # 分布类损失权重: MMD, 协方差, 多样性
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_cov", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    # 辅助矩匹配: 均值与方差
    parser.add_argument("--lambda_mean", type=float, default=0.5)
    parser.add_argument("--lambda_var", type=float, default=0.5)
    # MMD 等核计算时最多参与的 token 数
    parser.add_argument("--mmd_subsample", type=int, default=2048,
                        help="Max tokens for kernel matrix computation.")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    ensure_dir(os.path.dirname(args.output_path))

    # 1) 加载教师缓存: hidden, 元数据, 可选模态标签
    cache_payload = torch.load(args.teacher_cache_path, map_location="cpu")
    teacher_cache = cache_payload["teacher_cache"].float()
    teacher_meta = cache_payload["metadata"]
    teacher_labels = cache_payload.get("modality_labels", None)

    # 2) 设备与张量迁移
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_cache = teacher_cache.to(device)
    if teacher_labels is not None:
        teacher_labels = teacher_labels.to(device)

    # 3) 初始化: 从教师缓存随机抽 synthetic_size 条序列作为可学习参数, 可选加高斯噪声
    init_indices = torch.randint(0, teacher_cache.shape[0], (args.synthetic_size,), device=device)
    synthetic_hidden = torch.nn.Parameter(teacher_cache.index_select(0, init_indices).clone())
    if args.init_std > 0:
        synthetic_hidden.data.add_(torch.randn_like(synthetic_hidden) * args.init_std)

    # 合成样本的模态标签与初始化样本一致, 训练过程中不更新
    synth_labels = None
    if teacher_labels is not None:
        synth_labels = teacher_labels.index_select(0, init_indices).clone()  # not a Parameter

    # 4) 仅优化 synthetic_hidden, Adam
    optimizer = torch.optim.Adam([synthetic_hidden], lr=args.lr)
    history = []
    final_losses = None

    for step in trange(args.train_steps, desc="Distilling synthetic hidden", leave=False):
        # 每步从教师缓存随机采 teacher_batch_size 条序列 (有放回), 作为当前步对教师分布的蒙特卡洛近似,
        # 与 synthetic_size 不同: 后者是合成校准集的总条数, 每步仍用全部 synthetic_hidden 与本轮 teacher_batch 对齐.
        batch_indices = torch.randint(0, teacher_cache.shape[0], (args.teacher_batch_size,), device=device)
        teacher_batch = teacher_cache.index_select(0, batch_indices)
        teacher_batch_labels = teacher_labels.index_select(0, batch_indices) if teacher_labels is not None else None

        # 辅助: 均值与方差矩匹配
        stat_losses = _compute_stat_losses(teacher_batch, synthetic_hidden)

        # 主项: MMD, 协方差, 多样性 (可模态条件 MMD)
        dist_losses = _compute_distribution_losses(
            teacher_batch, synthetic_hidden,
            teacher_batch_labels, synth_labels,
            mmd_subsample=args.mmd_subsample,
        )

        total_loss = (
            args.lambda_mmd * dist_losses["mmd"]
            + args.lambda_cov * dist_losses["cov"]
            + args.lambda_div * dist_losses["div"]
            + args.lambda_mean * stat_losses["mean"]
            + args.lambda_var * stat_losses["var"]
        )

        # 反传并更新合成 hidden
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_losses = {
            "total": float(total_loss.detach().cpu().item()),
            "mmd": float(dist_losses["mmd"].detach().cpu().item()),
            "cov": float(dist_losses["cov"].detach().cpu().item()),
            "div": float(dist_losses["div"].detach().cpu().item()),
            "mean": float(stat_losses["mean"].detach().cpu().item()),
            "var": float(stat_losses["var"].detach().cpu().item()),
        }
        # 按 log_interval 记录标量损失曲线
        if step % args.log_interval == 0 or step == args.train_steps - 1:
            history.append({"step": step, **final_losses})

    # 5) 导出: CPU float32, 全 1 attention_mask, 写入 metadata 与可选 modality_labels
    synthetic_hidden_cpu = synthetic_hidden.detach().cpu().float()
    attention_mask = torch.ones(
        synthetic_hidden_cpu.shape[0],
        synthetic_hidden_cpu.shape[1],
        dtype=torch.long,
    )
    payload = {
        "synthetic_hidden": synthetic_hidden_cpu,
        "attention_mask": attention_mask,
        "metadata": {
            "method": "multimodal_representation_level_calibration_distillation",
            "teacher_cache_path": args.teacher_cache_path,
            "teacher_metadata": teacher_meta,
            "synthetic_size": args.synthetic_size,
            "compressed_length": int(synthetic_hidden_cpu.shape[1]),
            "hidden_size": int(synthetic_hidden_cpu.shape[2]),
            "dtype": "float32",
            "position_ids_strategy": "sequential_from_attention_mask",
            "train_steps": args.train_steps,
            "teacher_batch_size": args.teacher_batch_size,
            "lr": args.lr,
            "loss_weights": {
                "mmd": args.lambda_mmd,
                "cov": args.lambda_cov,
                "div": args.lambda_div,
                "mean": args.lambda_mean,
                "var": args.lambda_var,
            },
            "mmd_subsample": args.mmd_subsample,
            "final_losses": final_losses,
            "history": history,
            "created_at": utc_now_iso(),
        },
    }
    if synth_labels is not None:
        payload["modality_labels"] = synth_labels.cpu()

    torch.save(payload, args.output_path)
    print(f"[representation_distill] Saved synthetic calibration hidden to {args.output_path}")


if __name__ == "__main__":
    main()
