import math
import torch
from typing import Dict, Any, Tuple, List, Optional, Set
from time import time
from src.base.shared_utils import _print

def prepare_layer_info(
    scores: Dict[int, torch.Tensor],
    clamp_non_negative: bool = True,
) -> Tuple[Dict[int, Any], int, float]:
    """
    预处理每层的 saliency:
      - flatten 成一维
      - 可选: clamp 到非负
      - 按降序排序
      - 计算前缀和 (prefix sum)
    
    返回:
      layer_info: dict[layer_idx] -> {
          "size": int,                # 该层 channel 数 D_l
          "sorted": Tensor[D_l],      # 降序排好
          "prefix": Tensor[D_l],      # 前缀和
          "total_saliency": float,    # 前缀和最后一个元素
      }
      total_channels: 所有层的 channel 总数 N
      total_saliency: 所有层 saliency 总和 (用于算全局 coverage)
    """
    layer_info: Dict[int, Any] = {}
    total_channels = 0
    total_saliency = 0.0

    for l in range(len(scores)):
        v = scores[l]
        # flatten 成一维
        vals = v.reshape(-1).float()
        if clamp_non_negative:
            vals = vals.clamp_min(0.0)

        d_l = vals.numel()
        total_channels += d_l

        if d_l == 0:
            layer_info[l] = {
                "size": 0,
                "sorted": None,
                "prefix": None,
                "total_saliency": 0.0,
            }
            continue

        # 降序排序
        sorted_vals, _ = torch.sort(vals, descending=True)
        prefix = torch.cumsum(sorted_vals, dim=0)
        total = prefix[-1].item() if d_l > 0 else 0.0

        total_saliency += total

        layer_info[l] = {
            "size": d_l,
            "sorted": sorted_vals,
            "prefix": prefix,
            "total_saliency": total,
        }

    return layer_info, total_channels, float(total_saliency)


def eval_prune_ratio_for_s(
    layer_info: Dict[int, Any],
    total_channels: int,
    s_list: List[float],
) -> Tuple[float, Dict[int, int]]:
    """
    给定每层目标 "最少覆盖 s 比例的 saliency 总和",
    计算在该 s 下每层保留的 channel 数 k_l(s) 以及全局剪枝率 p(s)。

    思路:
      - 对每一层:
          如果 total_saliency <= 0: 退化为均匀 saliency, 保留 ceil(s * D_l) 个
          否则:
              找到最小的 k, 使得 prefix[k-1] >= s * total_saliency
      - 汇总 K(s) = sum_l k_l(s)
      - 全局剪枝率 p(s) = 1 - K(s) / N
    """
    keep_counts: Dict[int, int] = {}
    if total_channels == 0:
        return 0.0, keep_counts  # 极端情况

    K = 0  # 全局保留 channel 数
    for l, info in layer_info.items():
        s = s_list[l].item()
        d_l = info["size"]
        if d_l == 0:
            keep_counts[l] = 0
            continue

        total = info["total_saliency"]
        prefix = info["prefix"]

        if total <= 0.0:
            # 退化情况: 本层所有 saliency 都是 0, 等价于均匀
            if s <= 0.0:
                k_l = 0
            else:
                k_l = int(math.ceil(s * d_l))
        else:
            target = float(s) * total
            if target <= 0.0:
                k_l = 0
            else:
                # 使用 searchsorted 寻找第一个 >= target 的位置
                # prefix 是升序的 (因为是累加) 虽然 sorted_vals 是降序
                # 这里注意 prefix 在 CPU / GPU 的 device
                t = torch.tensor(target, device=prefix.device)
                idx = torch.searchsorted(prefix, t, right=False)
                k_l = int(idx.item()) + 1   # idx 是 0-based, channel 数要 +1

        # 防止越界
        if k_l > d_l:
            k_l = d_l
        if k_l < 0:
            k_l = 0

        keep_counts[l] = k_l
        K += k_l

    # 全局剪枝率
    prune_ratio = 1.0 - float(K) / float(total_channels)
    return prune_ratio, keep_counts


def binary_search_s_for_target_prune(
    layer_info: Dict[int, Any],
    total_channels: int,
    p_target: float,
    layerwise_loss: torch.Tensor = None, 
    max_iter: int = 32,
    tol: float = 1e-3,  # tol: 容忍误差
) -> Tuple[float, Dict[int, int], float]:
    """
    在 s ∈ [0, 1] 上做二分搜索, 找到使得全局剪枝率 p(s) ≈ p_target 的 s。
    返回:
      s_star: 最终选择的 s
      keep_counts: dict[layer] -> k_l(s_star)
      p_actual: 实际得到的全局剪枝率
    """
    # 边界检查
    p_target = float(p_target)
    if p_target < 0.0 or p_target > 1.0:
        raise ValueError(f"p_target must be in [0,1], got {p_target}")

    # p(s) 随 s 单调减小:
    #   s 增大 -> 每层 k_l(s) 增大 -> 全局保留数增大 -> 剪枝率减小
    low, high = 0.0, 1.0

    best_s = 0.0
    best_keep = {}
    best_p = None
    best_err = float("inf")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        s_list = [mid] * len(layer_info)
        s_list = torch.tensor(s_list)
        if layerwise_loss is not None:
            s_list = layerwise_loss * s_list.to(layerwise_loss.device)
        p_mid, keep_mid = eval_prune_ratio_for_s(layer_info, total_channels, s_list=s_list)

        err = abs(p_mid - p_target)
        if err < best_err:
            best_err = err
            best_s = s_list
            best_keep = keep_mid
            best_p = p_mid

        # 如果已经足够接近, 可以提前退出
        if err < tol:
            break

        # p(s) 大于目标, 说明剪枝太多了 (保留太少), 需要增大 s 以保留更多
        if p_mid > p_target:
            low = mid
        else:
            # p(s) 小于目标, 剪枝不够多, 需要减小 s
            high = mid

    # 防御一下 None 的情况
    if best_p is None:
        best_p, best_keep = eval_prune_ratio_for_s(layer_info, total_channels, best_s)

    return best_s, best_keep, best_p


def _keep_counts_to_ratio_tensor(
    num_layers: int,
    layer_info: Dict[int, Any],
    keep_counts: Dict[int, int],
) -> torch.Tensor:
    keep_ratio_per_layer = torch.zeros(num_layers, dtype=torch.float32)
    for l in range(num_layers):
        info = layer_info[l]
        d_l = info["size"]
        k_l = int(keep_counts.get(l, 0))
        if d_l > 0:
            keep_ratio_per_layer[l] = float(k_l) / float(d_l)
        else:
            raise ValueError(f"No channels found in layer {l}")
    return keep_ratio_per_layer


def _per_layer_min_keep_counts(
    layer_info: Dict[int, Any],
    num_layers: int,
    min_keep_channels: int,
    min_keep_ratio: float,
) -> Dict[int, int]:
    """
    Lower bound k_min[l] on kept channels per layer (cap at d_l).
    Use max(absolute floor, relative floor). If both min_keep_channels and min_keep_ratio are 0,
    returns all zeros (no floor).
    """
    k_min: Dict[int, int] = {}
    for l in range(num_layers):
        d_l = int(layer_info[l]["size"])
        if d_l <= 0:
            k_min[l] = 0
            continue
        if min_keep_channels <= 0 and min_keep_ratio <= 0.0:
            k_min[l] = 0
            continue
        abs_floor = int(min_keep_channels) if min_keep_channels > 0 else 0
        rel_floor = int(math.ceil(float(min_keep_ratio) * d_l)) if min_keep_ratio > 0.0 else 0
        k_min[l] = min(d_l, max(abs_floor, rel_floor))
    return k_min


def _relax_min_keep_to_fit_budget(
    k_min: Dict[int, int],
    layer_info: Dict[int, Any],
    num_layers: int,
    total_channels: int,
    k_target: int,
    verbose: bool,
) -> None:
    """If sum(k_min) > k_target, shrink floors ~proportionally to layer size (mutates k_min)."""
    s = sum(k_min[l] for l in range(num_layers))
    if s <= k_target:
        return
    if verbose:
        _print(
            f"[warn] per-layer min keep sum={s} > global keep budget k_target={k_target}; "
            f"relaxing floors toward proportional allocation.",
            flush=True,
        )
    for l in range(num_layers):
        d_l = int(layer_info[l]["size"])
        if d_l <= 0:
            k_min[l] = 0
            continue
        k_min[l] = min(d_l, max(0, int(round(float(k_target) * float(d_l) / float(total_channels)))))
    s2 = sum(k_min[l] for l in range(num_layers))
    while s2 > k_target:
        reduced = False
        for l in range(num_layers):
            if k_min[l] > 0:
                k_min[l] -= 1
                s2 -= 1
                reduced = True
                if s2 <= k_target:
                    break
        if not reduced:
            break


def _apply_min_keep_then_match_total(
    keep_counts: Dict[int, int],
    layer_info: Dict[int, Any],
    num_layers: int,
    k_min: Dict[int, int],
    k_target: int,
    verbose: bool,
) -> None:
    """
    Mutates keep_counts: enforce k_l >= k_min[l], then adjust total to k_target by
    dropping lowest-saliency kept channels first, then adding highest-saliency unkept channels.
    """
    for l in range(num_layers):
        d_l = int(layer_info[l]["size"])
        k = int(keep_counts.get(l, 0))
        keep_counts[l] = min(d_l, max(k, k_min[l]))

    def _total() -> int:
        return int(sum(keep_counts[l] for l in range(num_layers)))

    cur = _total()
    while cur > k_target:
        best_l: Optional[int] = None
        best_marginal = float("inf")
        for l in range(num_layers):
            k = int(keep_counts[l])
            if k <= k_min[l]:
                continue
            sv = layer_info[l]["sorted"]
            if sv is None or k < 1:
                continue
            marginal = float(sv[k - 1].item())
            if marginal < best_marginal:
                best_marginal = marginal
                best_l = l
        if best_l is None:
            if verbose:
                _print(
                    f"[warn] cannot squeeze to k_target={k_target}: stuck at total={cur} "
                    f"(all layers at min_keep floor).",
                    flush=True,
                )
            break
        keep_counts[best_l] -= 1
        cur -= 1

    while cur < k_target:
        best_l: Optional[int] = None
        best_marginal = float("-inf")
        for l in range(num_layers):
            k = int(keep_counts[l])
            d_l = int(layer_info[l]["size"])
            if k >= d_l:
                continue
            sv = layer_info[l]["sorted"]
            if sv is None:
                continue
            marginal = float(sv[k].item())
            if marginal > best_marginal:
                best_marginal = marginal
                best_l = l
        if best_l is None:
            break
        keep_counts[best_l] += 1
        cur += 1


def coverage_binary_search_keep_plan_with_high_loss_full_keep(
    scores: Dict[int, torch.Tensor],
    p_target: float,
    layerwise_loss: torch.Tensor,
    max_iter: int = 32,
    tol: float = None,
    min_keep_channels: int = 1048,
    min_keep_ratio: float = 0.1,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Same goal as coverage_binary_search_keep_plan, but when layerwise_loss scales differ
    wildly, plain s * loss scaling can make discrete per-layer keep counts miss p_target.

    Strategy: greedily freeze layers with the largest loss to **keep all channels** (k_l = d_l),
    subtract their channels from the global budget, then run the usual binary search only on
    the remaining layers with an adjusted local prune target so the **global** keep count
    matches (1 - p_target) * N.

    Tries increasing number of frozen layers (in order of decreasing loss) and picks the
    **smallest** freeze set whose global |p_actual - p_target| < tol; if none qualify, returns
    the best-effort plan with minimum error.

    Per-layer minimum keep (avoids k_l=0):
      k_min[l] = min(d_l, max(min_keep_channels, ceil(min_keep_ratio * d_l))).
    Set both to 0 to disable. After choosing raw keep counts, enforce floors then **re-balance**
    to the discrete global keep budget: remove channels with smallest saliency among currently
    kept (above floor), add channels with largest saliency among currently pruned.

    If layerwise_loss is None, raises ValueError (caller should use coverage_binary_search_keep_plan).
    """
    if layerwise_loss is None:
        raise ValueError("layerwise_loss is required for coverage_binary_search_keep_plan_with_high_loss_full_keep")

    st_time = time()
    layer_info, total_channels, _total_saliency = prepare_layer_info(scores)
    num_layers = len(scores)

    if tol is None:
        tol = 1 / max(layer_info[0]["size"], 1)

    if total_channels == 0:
        raise ValueError("No channels found in scores. Check your input.")

    loss_vec = layerwise_loss.detach().float().view(-1).cpu()
    if loss_vec.numel() != num_layers:
        raise ValueError(
            f"layerwise_loss length {loss_vec.numel()} != num_layers {num_layers}"
        )

    # Global target: total channels to KEEP (not prune)
    k_target_keep = (1.0 - float(p_target)) * float(total_channels)

    # Layers with highest loss first → freeze candidates
    sorted_by_loss = sorted(
        range(num_layers),
        key=lambda l: float(loss_vec[l].item()),
        reverse=True,
    )

    best_err = float("inf")
    best_keep_counts: Optional[Dict[int, int]] = None
    best_frozen: Set[int] = set()
    best_p: Optional[float] = None

    for m in range(num_layers + 1):
        frozen = set(sorted_by_loss[:m])
        k_frozen = float(sum(layer_info[l]["size"] for l in frozen))

        if k_frozen > k_target_keep + 1e-6:
            # Even full-keep on frozen alone exceeds global keep budget
            break

        flex_layers = [l for l in range(num_layers) if l not in frozen]
        n_flex = int(sum(layer_info[l]["size"] for l in flex_layers))
        k_need_flex = k_target_keep - k_frozen

        keep_counts: Dict[int, int] = {}

        for l in frozen:
            keep_counts[l] = int(layer_info[l]["size"])

        if not flex_layers:
            k_total = k_frozen
            p_actual = 1.0 - float(k_total) / float(total_channels)
            err = abs(p_actual - p_target)
            if err < best_err:
                best_err = err
                best_keep_counts = keep_counts
                best_frozen = set(frozen)
                best_p = p_actual
            if err < tol:
                break
            continue

        if k_need_flex <= 0:
            for l in flex_layers:
                keep_counts[l] = 0
        elif k_need_flex >= float(n_flex):
            for l in flex_layers:
                keep_counts[l] = int(layer_info[l]["size"])
        else:
            sub_info = {i: layer_info[flex_layers[i]] for i in range(len(flex_layers))}
            sub_loss = torch.stack([loss_vec[flex_layers[i]] for i in range(len(flex_layers))])
            p_flex = 1.0 - float(k_need_flex) / float(n_flex)

            _s_star, keep_sub, _p_flex_actual = binary_search_s_for_target_prune(
                layer_info=sub_info,
                total_channels=n_flex,
                p_target=p_flex,
                layerwise_loss=sub_loss,
                max_iter=max_iter,
                tol=tol,
            )
            for i, l in enumerate(flex_layers):
                keep_counts[l] = int(keep_sub[i])

        k_total = float(sum(keep_counts[l] for l in range(num_layers)))
        p_actual = 1.0 - k_total / float(total_channels)
        err = abs(p_actual - p_target)

        if err < best_err:
            best_err = err
            best_keep_counts = keep_counts
            best_frozen = set(frozen)
            best_p = p_actual

        if err < tol:
            if verbose:
                _print(
                    f"coverage_binary_search_keep_plan_with_high_loss_full_keep: "
                    f"m={m} frozen_layers={sorted(best_frozen)} global_prune={p_actual:.6f} target={p_target:.6f}",
                    flush=True,
                )
            break

    if best_keep_counts is None or best_p is None:
        raise RuntimeError("coverage_binary_search_keep_plan_with_high_loss_full_keep: internal error, no plan")

    k_target_int = int(round(k_target_keep))
    k_min = _per_layer_min_keep_counts(
        layer_info, num_layers, min_keep_channels, min_keep_ratio
    )
    _relax_min_keep_to_fit_budget(
        k_min, layer_info, num_layers, total_channels, k_target_int, verbose
    )
    _apply_min_keep_then_match_total(
        best_keep_counts, layer_info, num_layers, k_min, k_target_int, verbose
    )

    k_total_after = float(sum(best_keep_counts[l] for l in range(num_layers)))
    best_p = 1.0 - k_total_after / float(total_channels)
    best_err = abs(best_p - p_target)

    if best_err >= tol and verbose:
        _print(
            f"[warn] coverage_binary_search_keep_plan_with_high_loss_full_keep: "
            f"|p - p_target|={best_err:.6f} >= tol={tol:.6g}; returning best-effort "
            f"(frozen_layers={sorted(best_frozen)}, p_actual={best_p:.6f})",
            flush=True,
        )

    if verbose:
        _print(
            f"coverage_binary_search_keep_plan_with_high_loss_full_keep done in "
            f"{(time() - st_time) * 1000:.2f}ms",
            flush=True,
        )

    return _keep_counts_to_ratio_tensor(num_layers, layer_info, best_keep_counts)


def coverage_binary_search_keep_plan(
    scores: Dict[int, torch.Tensor],
    p_target: float,
    layerwise_loss: torch.Tensor = None, 
    max_iter: int = 32,
    tol: float = None,  # tol: 容忍误差, 对于 1408 通道模型, 1 个通道是 7e-4, 1e-3 对应 1.4 个通道
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    主函数:
      输入:
        scores: dict[layer_idx] -> Tensor, 每层的 saliency (任意 shape, 会 flatten)
        p_target: 目标全局剪枝率, 比如 0.5 表示剪掉一半 channel
      输出:
        一个 dict, 包含:
          - "s_star": 找到的层内 saliency 覆盖率下界
          - "keep_ratio_per_layer": dict[layer] -> 本层剪枝率
          - "keep_count_per_layer": dict[layer] -> 保留多少 channel
          - "global_prune_ratio": 实际全局剪枝率
          - "global_saliency_coverage": 全局保留的 saliency 比例
          - "saliency_coverage_per_layer": dict[layer] -> 本层保留的 saliency 比例
    """
    st_time = time()
    if verbose:
        _print("start to compute global prune plan", flush=True)
    
    # 1) 预处理每层 saliency
    start_time = time()
    layer_info, total_channels, total_saliency = prepare_layer_info(scores)
    prepare_time = time() - start_time
    
    if tol is None:
        tol = 1 / layer_info[0]["size"]
    if verbose:
        _print(f"tolerance={tol:.6f}", flush=True)
    if total_channels == 0:
        raise ValueError("No channels found in scores. Check your input.")

    # 2) 二分搜索 s
    start_time = time()
    s_star, keep_counts, p_actual = binary_search_s_for_target_prune(
        layer_info=layer_info,
        total_channels=total_channels,
        p_target=p_target,
        layerwise_loss=layerwise_loss,
        max_iter=max_iter,
        tol=tol,
    )
    binary_search_time = time() - start_time

    # 3) 基于最终的 k_l(s_star) 计算每层剪枝率与 saliency 覆盖率
    keep_ratio_per_layer = torch.zeros(len(scores), dtype=torch.float32)
    saliency_coverage_per_layer = torch.zeros(len(scores), dtype=torch.float32)
    kept_saliency_total = 0.0

    for l, info in layer_info.items():
        d_l = info["size"]
        k_l = keep_counts.get(l, 0)
        if d_l > 0:
            keep_ratio_per_layer[l] = float(k_l) / float(d_l)  # 保留率
        else:
            raise ValueError(f"No channels found in layer {l}")

        total_l = info["total_saliency"]
        prefix = info["prefix"]

        if total_l > 0.0 and k_l > 0:
            kept_l = prefix[k_l - 1].item()
            cov_l = kept_l / (total_l + 1e-12)
        else:
            kept_l = 0.0
            cov_l = 0.0

        kept_saliency_total += kept_l
        saliency_coverage_per_layer[l] = float(cov_l)

    if total_saliency > 0.0:
        global_saliency_coverage = kept_saliency_total / (total_saliency + 1e-12)
    else:
        global_saliency_coverage = 0.0

    if verbose:
        _print("[Time Summary] layerwise_planning time summary")
        _print(f"\t layerwise_planning: prepare_layer_info time: {prepare_time * 1000:.2f}ms", flush=True)
        _print(f"\t layerwise_planning: binary_search_time: {binary_search_time * 1000:.2f}ms", flush=True)
        _print(f"\t layerwise_planning: compute global prune plan done in {(time() - st_time) * 1000:.2f}ms", flush=True)
        
        
    result = {
        "s_star": s_star.tolist(),
        "keep_ratio_per_layer": keep_ratio_per_layer,
        "keep_count_per_layer": keep_counts,
        "global_prune_ratio": float(p_actual),
        "global_saliency_coverage": float(global_saliency_coverage),
        "saliency_coverage_per_layer": saliency_coverage_per_layer,
        # "total_channels": int(total_channels),
        # "total_saliency": float(total_saliency),
    }
    
    keep_ratio_per_layer = result['keep_ratio_per_layer']
    diff = abs(result['global_prune_ratio'] - p_target)
    try:
        assert diff < tol, f"coverage_binary_search_keep_plan: actual prune diff={diff} is not close to tol={tol}"
    except:
        # import ipdb; ipdb.set_trace()
        exit()
    
    if verbose:
        _print("Probe of coverage_binary_search_keep_plan:")
        for k, v in result.items():
            _print(f"\t {k}: {v}", flush=True)
            
    return keep_ratio_per_layer