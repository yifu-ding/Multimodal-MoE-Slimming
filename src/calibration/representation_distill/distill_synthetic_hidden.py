"""从教师模型 hidden 缓存中优化出一小份合成 hidden, 用于表征级校准.

训练目标: 用 MMD, 协方差, 多样性等分布损失, 辅以均值与方差矩匹配, 使合成样本在统计上逼近教师缓存.
"""
import argparse
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

from src.calibration.helpers.helpers import compute_block_loss
from src.calibration.representation_distill.common import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    ensure_dir,
    get_decoder_layer,
    seed_everything,
    sort_sequence_by_position_ids,
    utc_now_iso,
)
from src.calibration.representation_distill.runtime.forward_from_hidden import forward_from_hidden


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


def _build_synthetic_banks(
    init_hidden: torch.Tensor,
    init_labels: torch.Tensor | None,
    init_std: float,
) -> tuple[torch.nn.ParameterDict, torch.Tensor, torch.Tensor]:
    flat_hidden = init_hidden.reshape(-1, init_hidden.shape[-1])
    if init_labels is None:
        flat_labels = torch.full(
            (flat_hidden.shape[0],),
            MODALITY_TEXT,
            dtype=torch.int8,
            device=flat_hidden.device,
        )
    else:
        flat_labels = init_labels.reshape(-1).to(torch.int8)

    bank_params: dict[str, torch.nn.Parameter] = {}
    flat_bank_indices = torch.empty(flat_labels.shape[0], dtype=torch.long, device=flat_labels.device)
    for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
        mod_mask = flat_labels == mod_id
        if int(mod_mask.sum()) == 0:
            continue
        bank_value = flat_hidden[mod_mask].clone()
        if init_std > 0:
            bank_value.add_(torch.randn_like(bank_value) * init_std)
        bank_params[str(int(mod_id))] = torch.nn.Parameter(bank_value)
        flat_bank_indices[mod_mask] = torch.arange(
            int(mod_mask.sum()),
            device=flat_labels.device,
            dtype=torch.long,
        )
    return torch.nn.ParameterDict(bank_params), flat_labels, flat_bank_indices


def _assemble_synthetic_hidden(
    bank_params: torch.nn.ParameterDict,
    template_labels: torch.Tensor,
    template_bank_indices: torch.Tensor,
    synthetic_size: int,
    compressed_length: int,
    hidden_size: int,
) -> torch.Tensor:
    flat_labels = template_labels.reshape(-1)
    flat_bank_indices = template_bank_indices.reshape(-1)
    flat_hidden = torch.empty(
        flat_labels.shape[0],
        hidden_size,
        device=flat_labels.device,
        dtype=next(iter(bank_params.values())).dtype,
    )
    for mod_id_str, bank in bank_params.items():
        mod_mask = flat_labels == int(mod_id_str)
        flat_hidden[mod_mask] = bank.index_select(0, flat_bank_indices[mod_mask])
    return flat_hidden.view(synthetic_size, compressed_length, hidden_size)


def _sample_synthetic_batch_indices(
    synthetic_size: int,
    synthetic_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if synthetic_batch_size <= 0 or synthetic_batch_size >= synthetic_size:
        return torch.arange(synthetic_size, device=device, dtype=torch.long)
    return torch.randperm(synthetic_size, device=device)[:synthetic_batch_size]


def _sample_synthetic_batch_indices(
    synthetic_size: int,
    synthetic_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if synthetic_batch_size <= 0 or synthetic_batch_size >= synthetic_size:
        return torch.arange(synthetic_size, device=device, dtype=torch.long)
    return torch.randperm(synthetic_size, device=device)[:synthetic_batch_size]


def _compute_next_block_rel_l2(
    *,
    bundle,
    layer_idx: int,
    teacher_hidden: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_position_ids: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    synthetic_attention_mask: torch.Tensor,
    synthetic_position_ids: torch.Tensor,
) -> torch.Tensor:
    block_layer = get_decoder_layer(bundle, layer_idx)
    block_dtype = next(block_layer.parameters()).dtype
    teacher_hidden = teacher_hidden.to(dtype=block_dtype)
    synthetic_hidden = synthetic_hidden.to(dtype=block_dtype)

    with torch.no_grad():
        teacher_target = forward_from_hidden(
            bundle=bundle,
            hidden_states=teacher_hidden,
            attention_mask=teacher_attention_mask,
            start_layer=layer_idx,
            end_layer=layer_idx,
            position_ids=teacher_position_ids,
            apply_final_norm=False,
        )
    synth_pred = forward_from_hidden(
        bundle=bundle,
        hidden_states=synthetic_hidden,
        attention_mask=synthetic_attention_mask,
        start_layer=layer_idx,
        end_layer=layer_idx,
        position_ids=synthetic_position_ids,
        apply_final_norm=False,
    )
    loss_sum, _ = compute_block_loss(
        pred=synth_pred,
        teacher_target=teacher_target,
        attn_mask=synthetic_attention_mask,
        loss_fn="rel_l2",
    )
    denom = synthetic_attention_mask.float().sum().clamp_min(1.0)
    return loss_sum / denom


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


def _mean_var_loss(
    teacher_flat: torch.Tensor,
    synth_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel 1st/2nd moment MSE, returned as (mean_loss, var_loss)."""
    t_mean = teacher_flat.mean(dim=0)
    s_mean = synth_flat.mean(dim=0)
    t_var = teacher_flat.var(dim=0, unbiased=False)
    s_var = synth_flat.var(dim=0, unbiased=False)
    return (t_mean - s_mean).pow(2).mean(), (t_var - s_var).pow(2).mean()


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


def _iter_modality_groups(
    teacher_flat: torch.Tensor,
    synth_flat: torch.Tensor,
    teacher_labels_flat: torch.Tensor | None,
    synth_labels_flat: torch.Tensor | None,
    min_tokens: int = 2,
):
    """Yield (modality_id, teacher_sub, synth_sub) token groups.

    When labels are missing, yields a single ``(None, teacher_flat, synth_flat)`` pair
    so that the caller can fall back to a global (pooled) computation.
    """
    if teacher_labels_flat is None or synth_labels_flat is None:
        yield None, teacher_flat, synth_flat
        return

    for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
        t_mask = teacher_labels_flat == mod_id
        s_mask = synth_labels_flat == mod_id
        if int(t_mask.sum()) < min_tokens or int(s_mask.sum()) < min_tokens:
            continue
        yield mod_id, teacher_flat[t_mask], synth_flat[s_mask]


def _compute_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    teacher_labels: torch.Tensor | None,
    synth_labels: torch.Tensor | None,
    mmd_subsample: int,
) -> dict[str, torch.Tensor]:
    """All distillation losses, computed per-modality and averaged across modalities.

    Groups covered (MMD / cov / mean / var): text, image, video — whichever are
    present in *both* teacher and synthetic batches. When no modality labels are
    available the computation gracefully degrades to a single pooled pair.
    ``div`` stays pooled on the synthetic tokens (regardless of modality) because
    diversity should be measured on the whole synthetic set.
    """
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    teacher_labels_flat = teacher_labels.reshape(-1) if teacher_labels is not None else None
    synth_labels_flat = synth_labels.reshape(-1) if synth_labels is not None else None

    device = teacher_flat.device
    zero = torch.tensor(0.0, device=device)

    mmd_sum = zero.clone()
    cov_sum = zero.clone()
    mean_sum = zero.clone()
    var_sum = zero.clone()
    n_groups = 0
    per_group: dict[str, torch.Tensor] = {}

    for mod_id, t_group, s_group in _iter_modality_groups(
        teacher_flat, synth_flat, teacher_labels_flat, synth_labels_flat
    ):
        t_sub = _subsample_flat(t_group, mmd_subsample)
        s_sub = _subsample_flat(s_group, mmd_subsample)

        mmd_g = _mmd_rbf(t_sub, s_sub)
        cov_g = _cov_loss(t_sub, s_sub)
        mean_g, var_g = _mean_var_loss(t_group, s_group)

        mmd_sum = mmd_sum + mmd_g
        cov_sum = cov_sum + cov_g
        mean_sum = mean_sum + mean_g
        var_sum = var_sum + var_g
        n_groups += 1

        if mod_id is not None:
            per_group[f"mmd/mod{int(mod_id)}"] = mmd_g.detach()
            per_group[f"cov/mod{int(mod_id)}"] = cov_g.detach()
            per_group[f"mean/mod{int(mod_id)}"] = mean_g.detach()
            per_group[f"var/mod{int(mod_id)}"] = var_g.detach()

    if n_groups == 0:
        # Nothing matched (e.g. degenerate batch); fall back to pooled.
        t_sub = _subsample_flat(teacher_flat, mmd_subsample)
        s_sub = _subsample_flat(synth_flat, mmd_subsample)
        mmd_sum = _mmd_rbf(t_sub, s_sub)
        cov_sum = _cov_loss(t_sub, s_sub)
        mean_sum, var_sum = _mean_var_loss(teacher_flat, synth_flat)
        n_groups = 1

    div_loss = _diversity_loss(_subsample_flat(synth_flat, mmd_subsample))

    losses: dict[str, torch.Tensor] = {
        "mmd": mmd_sum / n_groups,
        "cov": cov_sum / n_groups,
        "mean": mean_sum / n_groups,
        "var": var_sum / n_groups,
        "div": div_loss,
    }
    losses.update(per_group)
    return losses


def _compute_weighted_total_loss(
    *,
    args,
    losses: dict[str, torch.Tensor],
    div_scale: float,
    ema_state: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base_device = losses["mmd"].device
    raw_terms = {
        "mmd": losses["mmd"],
        "cov": losses["cov"],
        "mean": losses["mean"],
        "var": losses["var"],
        "div": losses["div"],
    }
    if "block_rel_l2" in losses:
        raw_terms["block_rel_l2"] = losses["block_rel_l2"]

    weights = {
        "mmd": torch.tensor(args.lambda_mmd, device=base_device),
        "cov": torch.tensor(args.lambda_cov, device=base_device),
        "mean": torch.tensor(args.lambda_mean, device=base_device),
        "var": torch.tensor(args.lambda_var, device=base_device),
        "div": torch.tensor(div_scale * args.lambda_div, device=base_device),
    }
    if "block_rel_l2" in raw_terms:
        weights["block_rel_l2"] = torch.tensor(args.lambda_block, device=base_device)

    total_loss = torch.tensor(0.0, device=base_device)
    weighted_terms: dict[str, torch.Tensor] = {}
    normalized_terms: dict[str, torch.Tensor] = {}
    for name, raw in raw_terms.items():
        term = raw
        if args.use_ema_normalized_losses:
            if ema_state is None:
                raise ValueError("EMA state is required when EMA-normalized losses are enabled.")
            ema_value = ema_state.get(name)
            if ema_value is None:
                ema_value = raw.detach().float()
            else:
                ema_value = args.loss_ema_decay * ema_value + (1.0 - args.loss_ema_decay) * raw.detach().float()
            ema_state[name] = ema_value
            term = raw / ema_value.clamp_min(1e-8).to(device=raw.device, dtype=raw.dtype)
            normalized_terms[name] = term.detach()
        weighted = weights[name] * term
        weighted_terms[name] = weighted.detach()
        total_loss = total_loss + weighted
    return total_loss, weighted_terms, normalized_terms


def _build_output_payload(
    *,
    synthetic_hidden_cpu: torch.Tensor,
    synth_position_ids_cpu: torch.Tensor,
    synth_labels_cpu: torch.Tensor | None,
    teacher_meta: dict,
    args,
    final_losses: dict[str, float] | None,
    history: list[dict],
    block_constraint_layer: int | None,
) -> dict:
    attention_mask = torch.ones(
        synthetic_hidden_cpu.shape[0],
        synthetic_hidden_cpu.shape[1],
        dtype=torch.long,
    )
    payload = {
        "synthetic_hidden": synthetic_hidden_cpu,
        "attention_mask": attention_mask,
        "position_ids": synth_position_ids_cpu,
        "metadata": {
            "method": "multimodal_representation_level_calibration_distillation",
            "teacher_cache_path": args.teacher_cache_path,
            "teacher_metadata": teacher_meta,
            "synthetic_size": args.synthetic_size,
            "synthetic_batch_size": args.synthetic_batch_size,
            "compressed_length": int(synthetic_hidden_cpu.shape[1]),
            "hidden_size": int(synthetic_hidden_cpu.shape[2]),
            "dtype": "float32",
            "position_ids_strategy": "frozen_from_init_teacher_samples",
            "train_steps": args.train_steps,
            "teacher_batch_size": args.teacher_batch_size,
            "lr": args.lr,
            "loss_weights": {
                "mmd": args.lambda_mmd,
                "cov": args.lambda_cov,
                "div": args.lambda_div,
                "mean": args.lambda_mean,
                "var": args.lambda_var,
                "block_rel_l2": args.lambda_block,
            },
            "modality_grouped_losses": ["mmd", "cov", "mean", "var"],
            "div_warmup_steps": args.div_warmup_steps,
            "mmd_subsample": args.mmd_subsample,
            "use_ema_normalized_losses": args.use_ema_normalized_losses,
            "loss_ema_decay": args.loss_ema_decay,
            "synthetic_bank_mode": "independent_per_modality",
            "block_constraint_layer": block_constraint_layer,
            "final_losses": final_losses,
            "history": history,
            "created_at": utc_now_iso(),
        },
    }
    if synth_labels_cpu is not None:
        payload["modality_labels"] = synth_labels_cpu
    return payload


def _save_payload(
    *,
    output_path: str,
    synthetic_hidden: torch.Tensor,
    synth_position_ids: torch.Tensor,
    synth_labels: torch.Tensor | None,
    teacher_meta: dict,
    args,
    final_losses: dict[str, float] | None,
    history: list[dict],
    block_constraint_layer: int | None,
) -> dict:
    synthetic_hidden_cpu = synthetic_hidden.detach().cpu().float()
    synth_position_ids_cpu = synth_position_ids.detach().cpu().long()
    synth_labels_cpu = synth_labels.cpu() if synth_labels is not None else None
    payload = _build_output_payload(
        synthetic_hidden_cpu=synthetic_hidden_cpu,
        synth_position_ids_cpu=synth_position_ids_cpu,
        synth_labels_cpu=synth_labels_cpu,
        teacher_meta=teacher_meta,
        args=args,
        final_losses=final_losses,
        history=history,
        block_constraint_layer=block_constraint_layer,
    )
    torch.save(payload, output_path)
    return payload


def _assemble_full_synthetic_hidden(
    *,
    bank_params,
    synth_template_labels: torch.Tensor,
    synth_template_bank_indices: torch.Tensor,
    synthetic_size: int,
    compressed_length: int,
    hidden_size: int,
) -> torch.Tensor:
    return _assemble_synthetic_hidden(
        bank_params=bank_params,
        template_labels=synth_template_labels,
        template_bank_indices=synth_template_bank_indices,
        synthetic_size=synthetic_size,
        compressed_length=compressed_length,
        hidden_size=hidden_size,
    )


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
    parser.add_argument("--synthetic_batch_size", type=int, default=0,
                        help="How many synthetic samples to use per optimization step. "
                             "0 means using all synthetic samples.")
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_std", type=float, default=0.0)
    # Distribution-aware loss weights
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_cov", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    # 辅助矩匹配: 均值与方差
    parser.add_argument("--lambda_mean", type=float, default=0.5)
    parser.add_argument("--lambda_var", type=float, default=0.5)
    parser.add_argument("--lambda_block", type=float, default=0.0)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    # Diversity warmup: avoids ``div`` dominating the first few hundred steps
    # when the other losses are still on the order of 1e-5.
    parser.add_argument("--div_warmup_steps", type=int, default=200,
                        help="Linearly ramp ``lambda_div`` from 0 to its full value over "
                             "this many steps. Set to 0 to disable.")
    # MMD efficiency
    parser.add_argument("--mmd_subsample", type=int, default=2048,
                        help="Max tokens for kernel matrix computation.")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--use_ema_normalized_losses", action="store_true")
    parser.add_argument("--loss_ema_decay", type=float, default=0.99)
    parser.add_argument("--checkpoint_interval", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    ensure_dir(os.path.dirname(args.output_path))

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required when --wandb_project is set. "
                "Install it with `pip install wandb` or disable logging with "
                "`--wandb_mode disabled`."
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "teacher_cache_path": args.teacher_cache_path,
                "output_path": args.output_path,
                "synthetic_size": args.synthetic_size,
                "synthetic_batch_size": args.synthetic_batch_size,
                "teacher_batch_size": args.teacher_batch_size,
                "train_steps": args.train_steps,
                "lr": args.lr,
                "seed": args.seed,
                "init_std": args.init_std,
                "lambda_mmd": args.lambda_mmd,
                "lambda_cov": args.lambda_cov,
                "lambda_div": args.lambda_div,
                "lambda_mean": args.lambda_mean,
                "lambda_var": args.lambda_var,
                "lambda_block": args.lambda_block,
                "use_ema_normalized_losses": args.use_ema_normalized_losses,
                "loss_ema_decay": args.loss_ema_decay,
                "checkpoint_interval": args.checkpoint_interval,
                "div_warmup_steps": args.div_warmup_steps,
                "mmd_subsample": args.mmd_subsample,
                "log_interval": args.log_interval,
                "model_name_or_path": args.model_name_or_path,
                "device": args.device,
            },
        )

    cache_payload = torch.load(args.teacher_cache_path, map_location="cpu")
    teacher_cache = cache_payload["teacher_cache"].float()
    teacher_meta = cache_payload["metadata"]
    teacher_labels = cache_payload.get("modality_labels", None)
    teacher_position_ids = cache_payload.get("position_ids", None)

    # 2) 设备与张量迁移
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_cache = teacher_cache.to(device)
    if teacher_labels is not None:
        teacher_labels = teacher_labels.to(device)
    if teacher_position_ids is None:
        teacher_position_ids = torch.arange(
            teacher_cache.shape[1], device=device, dtype=torch.long
        ).unsqueeze(0).expand(teacher_cache.shape[0], -1)
    else:
        teacher_position_ids = teacher_position_ids.to(device=device, dtype=torch.long)
        teacher_cache, teacher_position_ids, teacher_labels = sort_sequence_by_position_ids(
            teacher_cache,
            teacher_position_ids,
            teacher_labels,
        )

    # 3) 初始化: 从教师缓存随机抽 synthetic_size 条序列作为可学习参数, 可选加高斯噪声
    init_indices = torch.randint(0, teacher_cache.shape[0], (args.synthetic_size,), device=device)
    init_hidden = teacher_cache.index_select(0, init_indices).clone()
    init_position_ids = teacher_position_ids.index_select(0, init_indices).clone()

    # 合成样本的模态标签与初始化样本一致, 训练过程中不更新
    synth_labels = None
    if teacher_labels is not None:
        synth_labels = teacher_labels.index_select(0, init_indices).clone()  # not a Parameter

    bank_params, synth_template_labels_flat, synth_template_bank_indices_flat = _build_synthetic_banks(
        init_hidden=init_hidden,
        init_labels=synth_labels,
        init_std=args.init_std,
    )
    synth_template_labels = (
        synth_labels
        if synth_labels is not None
        else synth_template_labels_flat.view(init_hidden.shape[0], init_hidden.shape[1])
    )
    synth_template_bank_indices = synth_template_bank_indices_flat.view(
        init_hidden.shape[0], init_hidden.shape[1]
    )
    optimizer = torch.optim.Adam(list(bank_params.parameters()), lr=args.lr)
    history = []
    final_losses = None
    loss_ema_state: dict[str, torch.Tensor] | None = {} if args.use_ema_normalized_losses else None

    block_bundle = None
    block_constraint_layer = None
    teacher_anchor_indices = init_indices.clone()
    teacher_anchor_attention_mask = torch.ones(init_hidden.shape[:2], dtype=torch.long, device=device)
    teacher_anchor_position_ids = init_position_ids
    synth_attention_mask = torch.ones(init_hidden.shape[:2], dtype=torch.long, device=device)
    synth_position_ids = init_position_ids.clone()
    effective_synth_batch_size = (
        args.synthetic_size if args.synthetic_batch_size <= 0
        else min(args.synthetic_batch_size, args.synthetic_size)
    )

    if args.lambda_block > 0:
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required when --lambda_block > 0.")
        from observations.common import load_model_bundle

        block_bundle = load_model_bundle(
            args.model_name_or_path,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
        block_constraint_layer = int(teacher_meta["teacher_layer"]) + 1
        block_layer = get_decoder_layer(block_bundle, block_constraint_layer)
        block_device = next(block_layer.parameters()).device
        teacher_anchor_attention_mask = teacher_anchor_attention_mask.to(block_device)
        teacher_anchor_position_ids = teacher_anchor_position_ids.to(block_device)
        synth_attention_mask = synth_attention_mask.to(block_device)
        synth_position_ids = synth_position_ids.to(block_device)
        if device != str(block_device):
            teacher_cache = teacher_cache.to(block_device)
            if teacher_labels is not None:
                teacher_labels = teacher_labels.to(block_device)
            if synth_labels is not None:
                synth_labels = synth_labels.to(block_device)
                synth_template_labels = synth_labels
            teacher_position_ids = teacher_position_ids.to(block_device)
            for _, bank in bank_params.items():
                bank.data = bank.data.to(block_device)
            synth_template_labels_flat = synth_template_labels_flat.to(block_device)
            synth_template_bank_indices_flat = synth_template_bank_indices_flat.to(block_device)
            synth_template_bank_indices = synth_template_bank_indices.to(block_device)
            teacher_anchor_indices = teacher_anchor_indices.to(block_device)
            device = str(block_device)

    for step in trange(args.train_steps, desc="Distilling compact hidden", leave=False):
        synth_batch_indices = _sample_synthetic_batch_indices(
            synthetic_size=args.synthetic_size,
            synthetic_batch_size=effective_synth_batch_size,
            device=synth_template_labels.device,
        )
        synth_batch_labels = synth_template_labels.index_select(0, synth_batch_indices)
        synth_batch_bank_indices = synth_template_bank_indices.index_select(0, synth_batch_indices)
        synth_batch_position_ids = synth_position_ids.index_select(0, synth_batch_indices)
        synth_batch_attention_mask = synth_attention_mask.index_select(0, synth_batch_indices)
        synthetic_hidden = _assemble_synthetic_hidden(
            bank_params=bank_params,
            template_labels=synth_batch_labels,
            template_bank_indices=synth_batch_bank_indices,
            synthetic_size=synth_batch_indices.shape[0],
            compressed_length=init_hidden.shape[1],
            hidden_size=init_hidden.shape[2],
        )
        # Sample a teacher batch
        batch_indices = torch.randint(0, teacher_cache.shape[0], (args.teacher_batch_size,), device=device)
        teacher_batch = teacher_cache.index_select(0, batch_indices)
        teacher_batch_labels = teacher_labels.index_select(0, batch_indices) if teacher_labels is not None else None

        # Per-modality grouped losses (primary). ``div`` stays pooled.
        losses = _compute_losses(
            teacher_batch, synthetic_hidden,
            teacher_batch_labels, synth_batch_labels,
            mmd_subsample=args.mmd_subsample,
        )

        # Warmup: ramp ``div`` in linearly so it doesn't dominate at step 0 when
        # the other losses are tiny (and would be destabilized by a large cosine
        # regularizer). After ``div_warmup_steps`` the full weight is applied.
        div_scale = 1.0
        if args.div_warmup_steps > 0:
            div_scale = min(1.0, (step + 1) / float(args.div_warmup_steps))

        if args.lambda_block > 0:
            teacher_anchor_hidden = teacher_cache.index_select(
                0,
                teacher_anchor_indices.index_select(0, synth_batch_indices),
            )
            block_rel_l2 = _compute_next_block_rel_l2(
                bundle=block_bundle,
                layer_idx=block_constraint_layer,
                teacher_hidden=teacher_anchor_hidden,
                teacher_attention_mask=teacher_anchor_attention_mask.index_select(0, synth_batch_indices),
                teacher_position_ids=teacher_anchor_position_ids.index_select(0, synth_batch_indices),
                synthetic_hidden=synthetic_hidden,
                synthetic_attention_mask=synth_batch_attention_mask,
                synthetic_position_ids=synth_batch_position_ids,
            )
            losses["block_rel_l2"] = block_rel_l2

        total_loss, weighted_terms, normalized_terms = _compute_weighted_total_loss(
            args=args,
            losses=losses,
            div_scale=div_scale,
            ema_state=loss_ema_state,
        )

        # 反传并更新合成 hidden
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_losses = {
            "total": float(total_loss.detach().cpu().item()),
            "mmd": float(losses["mmd"].detach().cpu().item()),
            "cov": float(losses["cov"].detach().cpu().item()),
            "div": float(losses["div"].detach().cpu().item()),
            "mean": float(losses["mean"].detach().cpu().item()),
            "var": float(losses["var"].detach().cpu().item()),
            "div_scale": div_scale,
        }
        if "block_rel_l2" in losses:
            final_losses["block_rel_l2"] = float(losses["block_rel_l2"].detach().cpu().item())
        for k, v in weighted_terms.items():
            final_losses[f"weighted/{k}"] = float(v.detach().cpu().item())
        for k, v in normalized_terms.items():
            final_losses[f"normalized/{k}"] = float(v.detach().cpu().item())
        for k, v in losses.items():
            if "/" in k:
                final_losses[k] = float(v.detach().cpu().item())
        if wandb_run is not None:
            wandb_payload = {
                "step": step,
                "loss/total": final_losses["total"],
                "loss/raw/mmd": final_losses["mmd"],
                "loss/raw/cov": final_losses["cov"],
                "loss/raw/mean": final_losses["mean"],
                "loss/raw/var": final_losses["var"],
                "loss/raw/div": final_losses["div"],
                "loss/weighted/mmd": final_losses["weighted/mmd"],
                "loss/weighted/cov": final_losses["weighted/cov"],
                "loss/weighted/mean": final_losses["weighted/mean"],
                "loss/weighted/var": final_losses["weighted/var"],
                "loss/weighted/div": final_losses["weighted/div"],
                "schedule/div_scale": div_scale,
                "meta/lr": args.lr,
                "meta/use_ema_normalized_losses": int(args.use_ema_normalized_losses),
            }
            if "block_rel_l2" in final_losses:
                wandb_payload["loss/raw/block_rel_l2"] = final_losses["block_rel_l2"]
                wandb_payload["loss/weighted/block_rel_l2"] = final_losses["weighted/block_rel_l2"]
            for k, v in final_losses.items():
                if "/" in k:
                    wandb_payload[f"loss/{k}"] = v
            wandb_run.log(wandb_payload, step=step)
        if step % args.log_interval == 0 or step == args.train_steps - 1:
            history.append({"step": step, **final_losses})
        if (
            args.checkpoint_interval > 0
            and (step + 1) % args.checkpoint_interval == 0
            and step != args.train_steps - 1
        ):
            checkpoint_hidden = _assemble_full_synthetic_hidden(
                bank_params=bank_params,
                synth_template_labels=synth_template_labels,
                synth_template_bank_indices=synth_template_bank_indices,
                synthetic_size=args.synthetic_size,
                compressed_length=init_hidden.shape[1],
                hidden_size=init_hidden.shape[2],
            )
            checkpoint_path = os.path.join(
                os.path.dirname(args.output_path),
                f"{os.path.splitext(os.path.basename(args.output_path))[0]}-step{step + 1}.pt",
            )
            _save_payload(
                output_path=checkpoint_path,
                synthetic_hidden=checkpoint_hidden,
                synth_position_ids=synth_position_ids,
                synth_labels=synth_labels,
                teacher_meta=teacher_meta,
                args=args,
                final_losses=final_losses,
                history=history,
                block_constraint_layer=block_constraint_layer,
            )
            print(f"[representation_distill] Saved checkpoint: {checkpoint_path}")

    synthetic_hidden = _assemble_full_synthetic_hidden(
        bank_params=bank_params,
        synth_template_labels=synth_template_labels,
        synth_template_bank_indices=synth_template_bank_indices,
        synthetic_size=args.synthetic_size,
        compressed_length=init_hidden.shape[1],
        hidden_size=init_hidden.shape[2],
    )
    payload = _save_payload(
        output_path=args.output_path,
        synthetic_hidden=synthetic_hidden,
        synth_position_ids=synth_position_ids,
        synth_labels=synth_labels,
        teacher_meta=teacher_meta,
        args=args,
        final_losses=final_losses,
        history=history,
        block_constraint_layer=block_constraint_layer,
    )
    print(f"[representation_distill] Saved synthetic calibration hidden to {args.output_path}")
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "output_path": args.output_path,
                "created_at": payload["metadata"]["created_at"],
                **{f"final/{k}": v for k, v in (final_losses or {}).items()},
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
