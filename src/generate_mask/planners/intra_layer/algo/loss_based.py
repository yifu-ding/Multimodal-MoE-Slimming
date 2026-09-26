import torch

from src.base.shared_utils import _print

def _enforce_layerwise_keep(plan: torch.Tensor,
                            keep_ratio: torch.Tensor,
                            tol: float = 1e-6) -> torch.Tensor:
    """
    调整每层 expert keep 计划，使得 per-layer 均值严格等于 keep_ratio。
    保证输出仍在 [0, 1] 区间，并且尽可能贴近原分布。
    """
    L, E = plan.shape
    target_sums = keep_ratio * E  # [L]
    adjusted = plan.clone()
    for layer_idx in range(L):
        row = adjusted[layer_idx]
        row.clamp_(0.0, 1.0)  # enforce box constraint

        target = target_sums[layer_idx].item()
        current = row.sum().item()
        diff = target - current

        if abs(diff) <= tol:
            continue

        if diff > 0:
            cap = 1.0 - row
            cap_sum = cap.sum().item()
            if cap_sum <= tol:
                continue
            row.add_(cap * (diff / cap_sum))
        else:
            surplus = -diff
            cap = row
            cap_sum = cap.sum().item()
            if cap_sum <= tol:
                continue
            row.sub_(cap * (surplus / cap_sum))

        row.clamp_(0.0, 1.0)

    return adjusted


def loss_based_expert_keep_plan(keep_ratio,
                                L: int,
                                E: int,
                                loss_result: torch.Tensor,
                                verbose=False):
    """
    依据每层内各 expert 的“损失/重要度”分布，将层级保留率 keep_ratio[L] 分配到专家维度得到 [L, E]。
    返回: expert_wise_keep_plan, 形状 [L, E], 数值范围 [0, 1]。
    """
    eps = 1e-25
    device = loss_result.device
    assert loss_result.shape == (L, E), f"loss_result must have shape (L, E), but got {loss_result.shape}"
    if loss_result.dtype != torch.float32:
        loss_result = loss_result.float()

    if isinstance(keep_ratio, (float, int)):
        keep_ratio = torch.ones(L, dtype=torch.float32, device=device) * keep_ratio
    elif isinstance(keep_ratio, (list, tuple)):
        keep_ratio = torch.tensor(keep_ratio, dtype=torch.float32, device=device)  # [L]
    else:
        keep_ratio = keep_ratio.to(device=device, dtype=torch.float32)
    assert keep_ratio.shape == (L,), f"keep_ratio must have shape (L,), but got {keep_ratio.shape}"
    keep_ratio = keep_ratio.clamp(min=0.0, max=1.0)  # [L]
     
    # 归一化每层内的 expert 重要度分布，零和层回退为均匀分配
    layerwise_loss_sum = loss_result.sum(dim=1, keepdim=True)  # [L, 1]
    expert_wise_loss_norm = loss_result / layerwise_loss_sum.clamp_min(eps)  # [L, E]
    zero_sum_mask = (layerwise_loss_sum.squeeze(1) <= eps)  # [L]
    if zero_sum_mask.any():
        expert_wise_loss_norm[zero_sum_mask] = 1.0 / E

    # 将层级保留率分配到专家维度：要求 per-layer 平均值≈keep_ratio[l]
    # 因每 expert 比例需在 [0,1]，乘以 E 后做 clamp
    raw_keep_plan = expert_wise_loss_norm * (keep_ratio[:, None] * E)  # [L, E]
    expert_wise_keep_plan = _enforce_layerwise_keep(raw_keep_plan, keep_ratio)

    if verbose:
        mean_keep = expert_wise_keep_plan.mean().item()
        _print(f"[loss_based_expert_keep_plan] plan_shape={tuple(expert_wise_keep_plan.shape)}, "
              f"min={expert_wise_keep_plan.min().item():.4f}, "
              f"max={expert_wise_keep_plan.max().item():.4f}, "
              f"mean={mean_keep:.4f}, target_mean={keep_ratio.mean().item():.4f}")
    return expert_wise_keep_plan