import torch
import math
from time import time

from src.base.shared_utils import _print

def build_masks_coverage(scores, loss_matrix=None, keep_ratio=None, L=None, E=None, I=None,  verbose=False):
    """
    结合 per-expert loss 覆盖度, 以“覆盖多少分数”来决定每个 expert 需要保留的最少通道。

    步骤:
        1. 加载离线收集的 loss 表 (CSV / PTH), 得到 [L, E] 的 loss matrix。
        2. 在每层内对 loss 做归一化, 得到 coverage seed (越大越重要)。
        3. 在满足 per-layer keep_ratio 的前提下, 通过 coverage 比例 + 二分搜索确定每个 expert
           需要保留的通道数 (以“覆盖分数”而非固定 channel 数为目标)。
        4. 回填 mask (1/0), 同时返回每个 expert 的保留通道数 K_E。
    """
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor shaped [L, E, I]")
    scores = scores.to(dtype=torch.float32, copy=False)
    if scores.ndim != 3:
        raise ValueError(f"expect scores with shape [L, E, I], got {scores.shape}")

    if L is None or E is None or I is None:
        L, E, I = scores.shape
    elif scores.shape != (L, E, I):
        raise ValueError(f"scores shape {scores.shape} mismatches provided (L={L}, E={E}, I={I})")

    start_time = time()
    keep_ratio_vec = _prepare_keep_ratio(keep_ratio, L, device=scores.device)
    prepare_time = time() - start_time
    coverage_weights = _loss_to_layerwise_weights(loss_matrix)

    masks = torch.zeros((L, E, I), dtype=torch.float32, device=scores.device)
    K_E = torch.zeros((L, E), dtype=torch.int64, device=scores.device)

    layer_total_channels = E * I

    start_time = time()
    for lid in range(L):
        target_keep = _target_channels_per_layer(keep_ratio_vec[lid].item(), layer_total_channels)
        if target_keep <= 0:
            continue

        layer_scores = scores[lid]
        sorted_vals, sorted_idx = torch.sort(layer_scores, dim=-1, descending=True)
        non_negative = torch.relu(sorted_vals)   # 非负
        prefix = torch.cumsum(non_negative, dim=-1)
        totals = prefix[:, -1]

        counts = _plan_layer_counts(
            coverage_weight=coverage_weights[lid],
            prefix=prefix,
            totals=totals,
            sorted_vals=sorted_vals,
            target_keep=target_keep,
        )

        for eid in range(E):
            keep_e = int(counts[eid].item())
            if keep_e <= 0:
                continue
            chosen = sorted_idx[eid, :keep_e]
            masks[lid, eid].index_fill_(0, chosen, 1.0)
            K_E[lid, eid] = keep_e

    total_time = time() - start_time
    
    if verbose:
        mask_ratio = masks.float().mean().item()
        _print(f"[loss_coverage] keep ratio: {mask_ratio:.4f}")
        _print("[Time Summary] intra_layer_planning time summary: ")
        _print("Probe of intra_layer_planning:")
        _print(f"\t intra_layer_planning: prepare_time: {prepare_time * 1000:.2f}ms", flush=True)
        _print(f"\t intra_layer_planning: total_time: {total_time * 1000:.2f}ms ({L} layers, {E} experts, {total_time * 1000/L:.2f}ms/layer)", flush=True)
        _print(f"\t remaining channels (L1): {K_E[0]}")
    return masks, K_E

def _prepare_keep_ratio(keep_ratio, L: int, device: torch.device) -> torch.Tensor:
    if keep_ratio is None:
        raise ValueError("keep_ratio must be provided for loss_coverage.")
    if isinstance(keep_ratio, torch.Tensor):
        ratio = keep_ratio.to(device=device, dtype=torch.float32)
    elif isinstance(keep_ratio, (float, int)):
        ratio = torch.full((L,), float(keep_ratio), dtype=torch.float32, device=device)
    else:
        ratio = torch.tensor(keep_ratio, dtype=torch.float32, device=device)

    if ratio.ndim == 0:
        ratio = ratio.expand(L)
    elif ratio.ndim == 1 and ratio.numel() == 1:
        ratio = ratio.expand(L)
    elif ratio.ndim != 1 or ratio.numel() != L:
        raise ValueError(f"keep_ratio must broadcast to [L], got shape {ratio.shape}")
    return ratio.clamp_(0.0, 1.0)


def _target_channels_per_layer(keep_ratio: float, layer_size: int) -> int:
    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0 or layer_size <= 0:
        return 0
    return min(layer_size, max(0, int(math.ceil(layer_size * keep_ratio))))


def _plan_layer_counts(
    coverage_weight: torch.Tensor,
    prefix: torch.Tensor,
    totals: torch.Tensor,
    sorted_vals: torch.Tensor,
    target_keep: int,
    max_iter: int = 32,
) -> torch.Tensor:
    E, max_channels = prefix.shape
    coverage_weight = coverage_weight.clone().detach()
    coverage_weight = coverage_weight.clamp_min(1e-6)
    counts = torch.zeros(E, dtype=torch.int64, device=prefix.device)
    if target_keep <= 0:
        return counts

    low, high = 0.0, float(1.0 / coverage_weight.min().item()) * 1.05
    best_counts = counts.clone()
    best_diff = target_keep

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        ratios = torch.clamp(coverage_weight * mid, max=1.0)
        curr_counts = _counts_from_ratios(ratios, prefix, totals, max_channels)
        total = int(curr_counts.sum().item())
        diff = abs(total - target_keep)
        if diff < best_diff:
            best_diff = diff
            best_counts = curr_counts.clone()
        if total == target_keep:
            best_counts = curr_counts
            break
        if total < target_keep:
            low = mid
        else:
            high = mid

    adjusted = _adjust_counts_to_target(best_counts, target_keep, sorted_vals)
    return adjusted


def _counts_from_ratios(
    ratios: torch.Tensor,
    prefix: torch.Tensor,
    totals: torch.Tensor,
    max_channels: int,
) -> torch.Tensor:
    counts = torch.zeros_like(ratios, dtype=torch.int64)
    for eid in range(ratios.numel()):
        counts[eid] = _channels_needed_for_ratio(
            prefix_row=prefix[eid],
            total=float(totals[eid].item()),
            coverage_ratio=float(ratios[eid].item()),
            max_channels=max_channels,
        )
    return counts


def _channels_needed_for_ratio(
    prefix_row: torch.Tensor,
    total: float,
    coverage_ratio: float,
    max_channels: int,
) -> int:
    coverage_ratio = float(max(0.0, min(1.0, coverage_ratio)))
    if coverage_ratio <= 0.0 or max_channels <= 0:
        return 0
    if total <= 0.0:
        return min(max_channels, max(0, int(math.ceil(max_channels * coverage_ratio))))
    target = total * coverage_ratio
    idx = torch.searchsorted(prefix_row, prefix_row.new_tensor(target), right=False)
    k = int(idx.item()) + 1
    return min(max_channels, max(k, 0))


def _adjust_counts_to_target(
    counts: torch.Tensor,
    target_keep: int,
    sorted_vals: torch.Tensor,
) -> torch.Tensor:
    counts = counts.clone()
    E, max_channels = sorted_vals.shape
    expert_ids = torch.arange(E, device=counts.device, dtype=torch.long)
    total = int(counts.sum().item())

    while total < target_keep:
        available = counts < max_channels
        if not bool(torch.any(available)):
            break
        next_idx = counts.clone()
        next_idx[~available] = 0
        candidate_scores = sorted_vals[expert_ids, next_idx]
        candidate_scores[~available] = float("-inf")
        step = min(target_keep - total, int(torch.count_nonzero(available).item()))
        if step <= 0:
            break
        _, top_idx = torch.topk(candidate_scores, k=step)
        top_idx = top_idx.to(counts.device)
        counts[top_idx] += 1
        total += step

    while total > target_keep:
        available = counts > 0
        if not bool(torch.any(available)):
            break
        prev_idx = (counts - 1).clamp(min=0)
        candidate_scores = sorted_vals[expert_ids, prev_idx]
        candidate_scores[~available] = float("inf")
        step = min(total - target_keep, int(torch.count_nonzero(available).item()))
        if step <= 0:
            break
        _, top_idx = torch.topk(-candidate_scores, k=step)
        top_idx = top_idx.to(counts.device)
        counts[top_idx] -= 1
        total -= step

    return counts.clamp_(min=0, max=max_channels)


# def _loss_to_layerwise_weights(loss_matrix: torch.Tensor) -> torch.Tensor:
#     loss = loss_matrix.clone().float()
#     min_vals = loss.min(dim=1, keepdim=True).values
#     loss = loss - min_vals
#     loss = loss + 1e-6
#     denom = loss.sum(dim=1, keepdim=True).clamp_min(1e-6)
#     return loss / denom

def _loss_to_layerwise_weights(loss_matrix: torch.Tensor) -> torch.Tensor:
    """
    loss_matrix: [L, E], 每层每个 expert 的 Δloss

    返回:
        weights: [L, E], 每层沿 E 归一化的权重.
        - 对于本层完全不重要的 expert, 对应位置会是 0.
        - 非 0 的权重可以用来映射到 coverage_ratio 或 K_e.
    """
    # import ipdb; ipdb.set_trace()
    loss = loss_matrix.clone().float()  # [L, E]
    # import ipdb; ipdb.set_trace()
    # 1. 只保留正的 Δloss, 负的直接视为 0 (删掉变好或没差)
    loss = torch.clamp(loss, min=0.0)   # [L, E]

    # # 2. 按层做一个相对阈值, 极小的 importance 直接置 0, 作为“可以直接删”
    # #    比如保留大约前 95 percent 的强 expert, 下面这个 0.05 是超参数
    # max_vals = loss.max(dim=1, keepdim=True).values  # [L, 1]
    # thresh = 0.05 * max_vals                         # [L, 1]
    # # 对 max 很小的层, thresh 也很小, clamp 一下避免数值问题
    # thresh = torch.clamp(thresh, min=0.0)
    # loss = torch.where(loss >= thresh, loss, torch.zeros_like(loss))  # 小于阈值的 expert 置 0

    # 3. 对剩下的正值做轻微平滑, 减少 dynamic 范围
    #    可以换成 torch.log1p(loss / (loss.mean(dim=1,keepdim=True)+eps)) 之类, 先用 sqrt 即可
    loss = torch.sqrt(loss)
    # loss = torch.log1p(loss/loss.mean(dim=1, keepdim=True))

    # 4. 按层归一化到和为 1, 但保留全 0 行用于“整层都不重要”的情况
    weights = torch.zeros_like(loss)
    row_sums = loss.sum(dim=1)  # [L]

    nonzero_rows = row_sums > 0
    if nonzero_rows.any():
        weights[nonzero_rows] = loss[nonzero_rows] / row_sums[nonzero_rows].clamp_min(1e-6).unsqueeze(1)

    # 此时:
    # - 有 importance 的 expert 得到正的权重, 且本层和为 1
    # - 被阈值干掉的 expert 权重为 0
    # - 整层 Δloss 几乎全为 0 的层, weights 那一行全 0, 你后面可以直接当“这层不用分 budget”
    return weights
