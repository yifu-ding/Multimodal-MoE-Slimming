"""Non-uniform pruning planners for Kimi-VL MoE channel pruning.

Two-level planning hierarchy
-----------------------------
Inter-layer — how much to keep per layer → keep_ratio[L]:

  uniform    : identical keep_ratio = 1-prune_ratio for every MoE layer

  coverage   : binary-search for saliency-coverage fraction s such that
               globally sum_l(k_l(s)) / N == (1-prune_ratio).
               Each layer l keeps the fewest channels needed to preserve s of
               its total saliency mass.  Layers with spiked score distributions
               are pruned more aggressively than flat ones.
               Optionally weighted by per-layer importance (layerwise_weights).

Intra-layer — how to distribute the per-layer budget across experts:

  expertwise : each expert independently top-k from its own scores;
               all experts in a layer keep the same k                [default]

  layerwise  : pool all E×I channel scores per layer; global top-k within
               the layer, then back-assigned to experts (experts may differ in k)

  global     : budget proportional to per-expert score mass, cross-layer

  coverage   : within each layer, expert e gets enough channels to cover
               (w_e * scale) fraction of its own saliency, where w_e is
               proportional to the expert's total score mass and scale is
               binary-searched to hit the per-layer channel budget.
               More-activated experts keep a larger share of their channels.

Ported and adapted from:
  LLM-Distillation/src/prune/generate/planners/inter_layer/algo/coverage.py
  LLM-Distillation/src/prune/generate/planners/intra_layer/algo/coverage.py
  LLM-Distillation/src/prune/generate/planners/intra_layer/algo/layerwise.py
  LLM-Distillation/src/prune/generate/planners/intra_layer/algo/expertwise.py
  LLM-Distillation/src/prune/generate/planners/intra_layer/algo/globally.py
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_expert_scores(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    layer_idx: int,
    eid: int,
    I: int,
) -> torch.Tensor:
    """Return float32 CPU score tensor for one expert; zeros if missing."""
    s = scores[layer_idx].get(eid, None)
    if s is None or s.numel() == 0:
        return torch.zeros(I, dtype=torch.float32)
    return s.float().cpu()


def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Bool mask of top-k positions in a 1-D float tensor."""
    k = max(1, min(k, scores.numel()))
    idx = torch.topk(scores.float().cpu(), k, largest=True).indices
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    mask[idx] = True
    return mask


def _channels_to_cover_fraction(
    prefix: torch.Tensor,   # sorted descending prefix-sum
    total: float,
    cov: float,             # target coverage fraction in [0, 1]
    max_k: int,
) -> int:
    """Minimum k such that prefix[k-1] >= cov * total."""
    if total <= 0.0 or cov <= 0.0:
        return 0
    target = total * min(1.0, max(0.0, cov))
    idx = int(torch.searchsorted(prefix, prefix.new_tensor(target), right=False).item())
    return min(max_k, max(1, idx + 1))


# ---------------------------------------------------------------------------
# Inter-layer planner: uniform
# ---------------------------------------------------------------------------

def plan_uniform(prune_ratio: float, L: int) -> List[float]:
    """Constant keep_ratio = 1 − prune_ratio for every layer."""
    return [1.0 - prune_ratio] * L


# ---------------------------------------------------------------------------
# Inter-layer planner: coverage (binary-search saliency coverage)
# ---------------------------------------------------------------------------

def _prepare_layer_info(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    sorted_layers: List[int],
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    For each layer, flatten all expert scores into one vector, sort descending,
    compute prefix-sum.  Returns (layer_info list, total_channels).

    layer_info[li] keys: size, sorted, prefix, total_saliency
    """
    layer_info: List[Dict[str, Any]] = []
    total_channels = 0
    for li, layer_idx in enumerate(sorted_layers):
        E = layer_to_num_experts[layer_idx]
        I = layer_to_num_channels[layer_idx]
        parts = [_get_expert_scores(scores, layer_idx, e, I).clamp_min(0.0)
                 for e in range(E)]
        flat = torch.cat(parts, dim=0)  # [E*I]
        d = flat.numel()
        total_channels += d
        sv, _ = torch.sort(flat, descending=True)
        prefix = torch.cumsum(sv, dim=0)
        layer_info.append({
            "layer_idx": layer_idx,
            "size": d,
            "sorted": sv,
            "prefix": prefix,
            "total_saliency": float(prefix[-1].item()) if d > 0 else 0.0,
        })
    return layer_info, total_channels


def _keep_count_for_coverage(info: Dict[str, Any], s: float) -> int:
    """Channels needed so that the layer covers fraction s of its saliency."""
    d = info["size"]
    if d == 0:
        return 0
    total = info["total_saliency"]
    prefix = info["prefix"]
    if total <= 0.0:
        return max(1, int(math.ceil(s * d)))
    return _channels_to_cover_fraction(prefix, total, s, d)


def _global_prune_ratio_for_s(
    layer_info: List[Dict[str, Any]],
    total_channels: int,
    s: float,
    layerwise_weights: Optional[torch.Tensor],
) -> Tuple[float, List[int]]:
    """Compute the global prune_ratio achieved when each layer targets coverage s."""
    keep_counts: List[int] = []
    K = 0
    for li, info in enumerate(layer_info):
        effective_s = s * (float(layerwise_weights[li].item()) if layerwise_weights is not None else 1.0)
        effective_s = max(0.0, min(1.0, effective_s))
        k = _keep_count_for_coverage(info, effective_s)
        keep_counts.append(k)
        K += k
    p = 1.0 - float(K) / float(total_channels)
    return p, keep_counts


def plan_coverage(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    sorted_layers: List[int],
    prune_ratio: float,
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
    layerwise_weights: Optional[torch.Tensor] = None,
    max_iter: int = 64,
    tol: Optional[float] = None,
) -> List[float]:
    """
    Binary-search for saliency coverage fraction s ∈ [0,1] such that globally
    the actual prune ratio ≈ p_target.

    Each layer l keeps the fewest channels needed so that their cumulative
    saliency ≥ s (× weight_l) × total_saliency_l.

    layerwise_weights : optional Tensor[L], positive.
        If provided, layers with higher weights need to cover a larger fraction
        of their saliency (i.e., they keep more channels), effectively shifting
        budget towards important layers.
        Typical source: per-layer loss increase when the layer is ablated.

    Returns
    -------
    List[float] of per-layer keep_ratio, length L.
    """
    layer_info, total_channels = _prepare_layer_info(
        scores, sorted_layers, layer_to_num_experts, layer_to_num_channels
    )
    L = len(sorted_layers)
    if tol is None:
        # 1 channel tolerance
        tol = 1.0 / max(total_channels, 1)

    # Normalize layerwise_weights so that mean == 1
    if layerwise_weights is not None:
        w = layerwise_weights.float().cpu()
        w_mean = w.mean().clamp_min(1e-12)
        layerwise_weights = w / w_mean

    low, high = 0.0, 1.0
    best_s, best_keep, best_err = 0.5, None, float("inf")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid, keep_mid = _global_prune_ratio_for_s(
            layer_info, total_channels, mid, layerwise_weights
        )
        err = abs(p_mid - prune_ratio)
        if err < best_err:
            best_err, best_s, best_keep = err, mid, keep_mid
        if err < tol:
            break
        # p(s) is monotone decreasing in s: larger s → more kept → lower p
        if p_mid > prune_ratio:
            low = mid   # prune too much → raise s → keep more
        else:
            high = mid  # prune too little → lower s → keep less

    # Convert keep_counts → keep_ratios
    keep_ratios: List[float] = []
    for li, info in enumerate(layer_info):
        d = info["size"]
        k = best_keep[li] if best_keep is not None else round((1.0 - prune_ratio) * d)
        keep_ratios.append(float(k) / float(d) if d > 0 else 1.0 - prune_ratio)
    return keep_ratios


# ---------------------------------------------------------------------------
# Intra-layer mask builders
# ---------------------------------------------------------------------------

def _build_masks_expertwise(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    sorted_layers: List[int],
    layer_keep_ratios: List[float],
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
) -> Dict[int, torch.Tensor]:
    """Each expert independently: top-k of its own scores.
    All experts in a layer keep k = round(I * keep_ratio[l])."""
    masks: Dict[int, torch.Tensor] = {}
    for li, layer_idx in enumerate(sorted_layers):
        E = layer_to_num_experts[layer_idx]
        I = layer_to_num_channels[layer_idx]
        k = max(1, round(I * layer_keep_ratios[li]))
        layer_mask = torch.zeros(E, I, dtype=torch.bool)
        for eid in range(E):
            s = _get_expert_scores(scores, layer_idx, eid, I)
            layer_mask[eid] = _topk_mask(s, k)
        masks[layer_idx] = layer_mask
    return masks


def _build_masks_layerwise(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    sorted_layers: List[int],
    layer_keep_ratios: List[float],
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
) -> Dict[int, torch.Tensor]:
    """Pool all E×I scores per layer; global top-k within the layer,
    back-assigned to experts.  k_layer = round(E * I * keep_ratio[l]).
    Guarantees at least 1 channel per expert."""
    masks: Dict[int, torch.Tensor] = {}
    for li, layer_idx in enumerate(sorted_layers):
        E = layer_to_num_experts[layer_idx]
        I = layer_to_num_channels[layer_idx]
        k_total = max(E, round(E * I * layer_keep_ratios[li]))

        expert_scores = [_get_expert_scores(scores, layer_idx, e, I) for e in range(E)]
        flat = torch.cat(expert_scores, dim=0)   # [E*I]

        top_global = torch.topk(flat, min(k_total, flat.numel()), largest=True).indices
        layer_mask = torch.zeros(E, I, dtype=torch.bool)
        layer_mask[top_global // I, top_global % I] = True

        # Guarantee at least 1 channel per expert
        for eid in range(E):
            if not layer_mask[eid].any():
                layer_mask[eid, int(expert_scores[eid].argmax().item())] = True

        masks[layer_idx] = layer_mask
    return masks


def _build_masks_global(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    sorted_layers: List[int],
    prune_ratio: float,
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
) -> Dict[int, torch.Tensor]:
    """Budget proportional to per-expert score mass across ALL layers.
    Ignores inter-layer keep_ratio; coverage = None."""
    score_mass: Dict[int, Dict[int, float]] = {}
    total_mass = 0.0
    total_channels = 0
    for layer_idx in sorted_layers:
        score_mass[layer_idx] = {}
        E = layer_to_num_experts[layer_idx]
        I = layer_to_num_channels[layer_idx]
        total_channels += E * I
        for eid in range(E):
            s = _get_expert_scores(scores, layer_idx, eid, I)
            m = float(s.sum().item())
            score_mass[layer_idx][eid] = m
            total_mass += m

    K_total = max(len(sorted_layers), round((1.0 - prune_ratio) * total_channels))

    masks: Dict[int, torch.Tensor] = {}
    for layer_idx in sorted_layers:
        E = layer_to_num_experts[layer_idx]
        I = layer_to_num_channels[layer_idx]
        layer_mask = torch.zeros(E, I, dtype=torch.bool)
        for eid in range(E):
            w = score_mass[layer_idx][eid] / (total_mass + 1e-12)
            k = max(1, min(I, round(w * K_total)))
            s = _get_expert_scores(scores, layer_idx, eid, I)
            layer_mask[eid] = _topk_mask(s, k)
        masks[layer_idx] = layer_mask
    return masks


def _build_masks_coverage(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    sorted_layers: List[int],
    layer_keep_ratios: List[float],
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
    expertwise_weights: Optional[torch.Tensor] = None,
    max_iter: int = 64,
) -> Dict[int, torch.Tensor]:
    """
    Coverage-based intra-layer channel selection.

    For each layer l with total budget k_layer = round(E * I * keep_ratio_l):

      1. Use an external per-expert anchor weight w_e within the layer.
         This determines the relative target coverage ratio across experts
         before binary-searching the shared scale factor.

      2. Binary-search for a scale factor t such that
             sum_e( k_e(w_e * t) ) ≈ k_layer
         where k_e(cov) = min k so that prefix_e[k-1] >= cov * total_e.

      3. Build each expert's mask from its coverage-driven k_e.

    Effect: an expert whose channels are highly concentrated (one channel
    dominates) needs fewer kept channels to maintain its saliency than an
    expert with a flat score distribution — more principled than top-k.

    If expertwise_weights is None, falls back to a uniform anchor across experts.

    Ported from:
      LLM-Distillation/src/prune/generate/planners/intra_layer/algo/coverage.py
    """
    masks: Dict[int, torch.Tensor] = {}

    for li, layer_idx in enumerate(sorted_layers):
        E = layer_to_num_experts[layer_idx]
        I = layer_to_num_channels[layer_idx]
        k_layer = max(E, round(E * I * layer_keep_ratios[li]))

        # Per-expert sorted scores and prefix sums
        expert_s_info: List[Dict[str, Any]] = []
        for eid in range(E):
            sv = _get_expert_scores(scores, layer_idx, eid, I).clamp_min(0.0)
            sv_sorted, _ = torch.sort(sv, descending=True)
            prefix = torch.cumsum(sv_sorted, dim=0)
            total_e = float(prefix[-1].item()) if sv.numel() > 0 else 0.0
            expert_s_info.append({
                "sorted": sv_sorted,
                "prefix": prefix,
                "total": total_e,
            })

        if expertwise_weights is None:
            cov_weights = torch.ones(E, dtype=torch.float32) / E
        else:
            cov_weights = expertwise_weights[li].float().cpu().clone()
            cov_weights = cov_weights.clamp_min(0.0)
            cov_weights = torch.sqrt(cov_weights)
            total_cov = float(cov_weights.sum().item())
            if total_cov < 1e-12:
                cov_weights = torch.ones(E, dtype=torch.float32) / E
            else:
                cov_weights /= total_cov

        # Binary search for scale t such that sum(k_e(w_e * t)) ≈ k_layer
        def _total_kept(t: float) -> Tuple[int, List[int]]:
            counts: List[int] = []
            total = 0
            for eid, info in enumerate(expert_s_info):
                cov = float(cov_weights[eid].item()) * t
                k = _channels_to_cover_fraction(info["prefix"], info["total"], cov, I)
                counts.append(k)
                total += k
            return total, counts

        low, high = 0.0, float(E)   # scale t range
        best_counts: Optional[List[int]] = None
        best_err = float("inf")

        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            total, counts = _total_kept(mid)
            err = abs(total - k_layer)
            if err < best_err:
                best_err, best_counts = err, counts
            if total == k_layer:
                break
            # Monotone: larger t → more coverage → more channels kept
            if total < k_layer:
                low = mid   # need more channels → raise t
            else:
                high = mid  # need fewer channels → lower t

        if best_counts is None:
            # Fallback: uniform expertwise
            best_counts = [max(1, round(I * layer_keep_ratios[li]))] * E

        # Fine-tune: adjust total to exactly k_layer by adding/removing
        # the marginal channel (highest unselected or lowest selected score)
        counts = list(best_counts)
        cur = sum(counts)
        # Greedily add channels with highest marginal score
        while cur < k_layer:
            best_eid, best_score = -1, float("-inf")
            for eid, info in enumerate(expert_s_info):
                k = counts[eid]
                if k < I:
                    score = float(info["sorted"][k].item()) if k < I else float("-inf")
                    if score > best_score:
                        best_score, best_eid = score, eid
            if best_eid == -1:
                break
            counts[best_eid] += 1
            cur += 1

        # Greedily drop channels with lowest marginal score
        while cur > k_layer:
            best_eid, worst_score = -1, float("inf")
            for eid, info in enumerate(expert_s_info):
                k = counts[eid]
                if k > 1:
                    score = float(info["sorted"][k - 1].item())
                    if score < worst_score:
                        worst_score, best_eid = score, eid
            if best_eid == -1:
                break
            counts[best_eid] -= 1
            cur -= 1

        # Build masks
        layer_mask = torch.zeros(E, I, dtype=torch.bool)
        for eid, info in enumerate(expert_s_info):
            k = max(1, counts[eid])
            raw_s = _get_expert_scores(scores, layer_idx, eid, I)
            layer_mask[eid] = _topk_mask(raw_s, k)
        masks[layer_idx] = layer_mask

    return masks


# ---------------------------------------------------------------------------
# Eviction adjuster  (post-planning: remove tiny experts, redistribute budget)
# ---------------------------------------------------------------------------

def _largest_remainder_alloc(
    ideals: torch.Tensor,
    caps: torch.Tensor,
    target: int,
) -> torch.Tensor:
    """Standard largest-remainder integer allocation with per-slot caps.

    Ported verbatim from:
      LLM-Distillation/src/prune/generate/adjusters/utils.py
    """
    assert ideals.dtype in (torch.float64, torch.float32)
    assert caps.dtype == torch.int64
    n = ideals.numel()
    base = torch.floor(ideals).to(torch.int64)
    base = torch.minimum(base, caps)
    s = int(base.sum().item())
    remain = min(target, int(caps.sum().item())) - s
    if remain <= 0:
        return base
    frac = ideals - base.to(ideals.dtype)
    eligible = base < caps
    scores_f = torch.where(eligible, frac, torch.full_like(frac, -1e9))
    order = torch.argsort(scores_f, descending=True)
    res = base.clone()
    j = 0
    while remain > 0 and j < n:
        idx = int(order[j].item())
        if res[idx] < caps[idx]:
            res[idx] += 1
            remain -= 1
        j += 1
    return res


def _rebuild_masks_from_K(
    K: torch.Tensor,
    sorted_layers: List[int],
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    layer_to_num_channels: Dict[int, int],
) -> Dict[int, torch.Tensor]:
    """Build bool keep-masks from a [L, E] integer keep-count tensor."""
    masks: Dict[int, torch.Tensor] = {}
    for li, layer_idx in enumerate(sorted_layers):
        I = layer_to_num_channels[layer_idx]
        E = K.shape[1]
        layer_mask = torch.zeros(E, I, dtype=torch.bool)
        for eid in range(E):
            k = int(K[li, eid].item())
            if k == 0:
                continue
            s = _get_expert_scores(scores, layer_idx, eid, I)
            layer_mask[eid] = _topk_mask(s, k)
        masks[layer_idx] = layer_mask
    return masks


def _evict_adjust_masks(
    masks: Dict[int, torch.Tensor],
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    sorted_layers: List[int],
    layer_to_num_channels: Dict[int, int],
    min_per_expert: int = 16,
) -> Dict[int, torch.Tensor]:
    """Post-planning eviction adjustment.

    Any expert assigned fewer than ``min_per_expert`` channels is fully evicted
    (kept count → 0). The freed budget is redistributed to surviving active
    experts within each layer, proportional to their remaining headroom (up to
    the original per-layer total-channel budget).  If freed budget is large
    enough to "re-activate" a previously evicted expert, it is restored at
    exactly ``min_per_expert`` channels.

    Total channel count across all layers is preserved at the level achievable
    under the min constraint (i.e., the global target is not exceeded).

    Ported and adapted from:
      LLM-Distillation/src/prune/generate/adjusters/evict.py
    """
    L = len(sorted_layers)

    # K[li, eid] = number of channels expert eid keeps in layer li
    K = torch.stack(
        [masks[sorted_layers[li]].sum(dim=1).to(torch.int64) for li in range(L)],
        dim=0,
    )  # [L, E]
    E = K.shape[1]

    # Each layer's original total-channel budget (preserved as a hard cap)
    layer_caps_vec = K.sum(dim=1).clone()  # [L]

    # Per-expert hard cap: at most I channels (the full intermediate size)
    layer_cap_e: List[torch.Tensor] = [
        torch.full(
            (masks[sorted_layers[li]].shape[0],),
            layer_to_num_channels[sorted_layers[li]],
            dtype=torch.int64,
        )
        for li in range(L)
    ]

    # 1) Evict experts with 0 < k < min_per_expert; accumulate freed budget
    freed_total = 0
    for li in range(L):
        evict = (K[li] > 0) & (K[li] < min_per_expert)
        freed_total += int(K[li][evict].sum().item())
        K[li][evict] = 0

    if freed_total == 0:
        return masks   # nothing changed — return original masks as-is

    # 2) Distribute freed_total back across layers proportional to headroom
    layer_used = K.sum(dim=1)
    layer_headroom = torch.clamp(layer_caps_vec - layer_used, min=torch.zeros_like(layer_caps_vec))
    total_headroom = int(layer_headroom.sum().item())

    if total_headroom == 0:
        return _rebuild_masks_from_K(K, sorted_layers, scores, layer_to_num_channels)

    layer_add = _largest_remainder_alloc(
        ideals=(layer_headroom.double() * (freed_total / max(total_headroom, 1))).to(torch.float64),
        caps=layer_headroom,
        target=freed_total,
    )

    for li in range(L):
        add_i = int(layer_add[li].item())
        if add_i <= 0:
            continue
        row = K[li].clone()
        caps_e = layer_cap_e[li]

        def _give_to_active(row, caps_e, budget):
            active = row >= min_per_expert
            if not active.any() or budget <= 0:
                return row, budget
            hr = torch.clamp(caps_e - row, min=torch.zeros_like(row))
            hr_a = torch.where(active, hr, torch.zeros_like(hr))
            total_hr = int(hr_a.sum().item())
            if total_hr <= 0:
                return row, budget
            give = _largest_remainder_alloc(
                ideals=(hr_a.double() * (budget / total_hr)).to(torch.float64),
                caps=hr_a,
                target=budget,
            )
            return row + give, budget - int(give.sum().item())

        # Step 1: give to already-active experts
        row, add_i = _give_to_active(row, caps_e, add_i)

        # Step 2: re-activate evicted experts if budget allows (sets them to min_per_expert)
        if add_i >= min_per_expert:
            zeros = row == 0
            if zeros.any():
                hr = torch.clamp(caps_e - row, min=torch.zeros_like(row))
                eligible_zeros = zeros & (hr >= min_per_expert)
                max_new = min(add_i // min_per_expert, int(eligible_zeros.sum().item()))
                if max_new > 0:
                    idx_zero = torch.where(eligible_zeros)[0][:max_new]
                    row[idx_zero] = min_per_expert
                    add_i -= max_new * min_per_expert

        # Step 3: any remainder goes back to active experts
        row, _ = _give_to_active(row, caps_e, add_i)

        K[li] = row

        # Safety: enforce layer cap
        s = int(K[li].sum().item())
        cap = int(layer_caps_vec[li].item())
        if s > cap:
            surplus = s - cap
            reducible = K[li].clone()
            real_cut = reducible.double() * (surplus / max(int(reducible.sum().item()), 1))
            base_cut = torch.minimum(torch.floor(real_cut).to(torch.int64), reducible)
            K[li] = K[li] - base_cut
            rem = surplus - int(base_cut.sum().item())
            if rem > 0:
                frac_arr = (real_cut - base_cut.double()).cpu().numpy()
                order = np.argsort(-frac_arr)
                j = 0
                while rem > 0 and j < E:
                    idx = int(order[j])
                    if K[li, idx] > 0:
                        K[li, idx] -= 1
                        rem -= 1
                    j += 1

    # Final cleanup: collapse any 1..min-1 stragglers to 0 (shouldn't occur, but be safe)
    for li in range(L):
        K[li][(K[li] > 0) & (K[li] < min_per_expert)] = 0

    return _rebuild_masks_from_K(K, sorted_layers, scores, layer_to_num_channels)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

INTER_LAYER_METHODS = ("uniform", "coverage", "global")
INTRA_LAYER_METHODS = ("expertwise", "layerwise", "coverage")


def generate_masks(
    scores: Dict[int, Dict[int, Optional[torch.Tensor]]],
    prune_ratio: float,
    layer_to_num_experts: Dict[int, int],
    layer_to_num_channels: Dict[int, int],
    inter_method: str = "uniform",
    intra_method: str = "expertwise",
    # coverage inter-layer kwargs
    layerwise_weights: Optional[torch.Tensor] = None,
    # coverage intra-layer kwargs
    expertwise_weights: Optional[torch.Tensor] = None,
    # eviction adjuster kwargs
    evict_min_channels: int = 0,
) -> Dict[int, torch.Tensor]:
    """Generate per-expert keep-masks.

    Parameters
    ----------
    scores : Dict[layer_idx, Dict[expert_idx, Tensor[I]]]
    prune_ratio : float
        Global fraction of channels to remove.
    inter_method : "uniform" | "coverage" | "global"
        Inter-layer budget allocation.
    intra_method : "expertwise" | "layerwise" | "global" | "coverage"
        Intra-layer channel selection.
        "global" ignores inter_method and allocates by score mass cross-layer.
    layerwise_weights : Tensor[L], optional
        Per-layer importance weights for the "coverage" inter-layer method.
        Typical source: per-layer ablation loss increase.
        If None, uses plain (unweighted) coverage binary search.
    expertwise_weights : Tensor[L, E], optional
        Per-expert anchor weights for the "coverage" intra-layer planner.
        If None, uses a uniform anchor within each layer.
    evict_min_channels : int, optional
        If > 0, run a post-planning eviction pass: any expert left with fewer
        than this many channels is fully evicted (set to 0), and the freed
        budget is redistributed to surviving active experts within each layer.
        Set to 0 (default) to skip eviction.

    Returns
    -------
    Dict[layer_idx, Tensor[E, I]] bool — True = keep this channel.
    """
    if inter_method not in INTER_LAYER_METHODS:
        raise ValueError(f"inter_method must be one of {INTER_LAYER_METHODS}")
    if intra_method not in INTRA_LAYER_METHODS:
        raise ValueError(f"intra_method must be one of {INTRA_LAYER_METHODS}")

    sorted_layers = sorted(scores.keys())

    # "global" bypasses inter-layer planning entirely
    if inter_method == "global":
        return _build_masks_global(
            scores, sorted_layers, prune_ratio,
            layer_to_num_experts, layer_to_num_channels,
        )

    # --- inter-layer plan ---
    if inter_method == "uniform":
        layer_keep_ratios = plan_uniform(prune_ratio, len(sorted_layers))
    else:  # coverage
        layer_keep_ratios = plan_coverage(
            scores, sorted_layers, prune_ratio,
            layer_to_num_experts, layer_to_num_channels,
            layerwise_weights=layerwise_weights,
        )

    # --- intra-layer masks ---
    if intra_method == "expertwise":  # expertwise uniform rank channels
        masks = _build_masks_expertwise(
            scores, sorted_layers, layer_keep_ratios,
            layer_to_num_experts, layer_to_num_channels,
        )
    elif intra_method == "layerwise":  # layerwise rank channels, with each expert non-uniform channels
        masks = _build_masks_layerwise(
            scores, sorted_layers, layer_keep_ratios,
            layer_to_num_experts, layer_to_num_channels,
        )
    else:  # coverage
        masks = _build_masks_coverage(
            scores, sorted_layers, layer_keep_ratios,
            layer_to_num_experts, layer_to_num_channels,
            expertwise_weights=expertwise_weights,
        )

    # --- optional eviction pass ---
    if evict_min_channels > 0:
        masks = _evict_adjust_masks(
            masks, scores, sorted_layers, layer_to_num_channels,
            min_per_expert=evict_min_channels,
        )

    return masks


# ---------------------------------------------------------------------------
# Mask statistics helpers (used by prune.py for config updates)
# ---------------------------------------------------------------------------

def masks_are_uniform(
    masks: Dict[int, torch.Tensor]
) -> Tuple[bool, Optional[int]]:
    """True + I' if every expert keeps the same number of channels."""
    ref: Optional[int] = None
    for m in masks.values():
        for eid in range(m.shape[0]):
            k = int(m[eid].sum().item())
            if ref is None:
                ref = k
            elif k != ref:
                return False, None
    return True, ref


def masks_are_layerwise_uniform(
    masks: Dict[int, torch.Tensor]
) -> Tuple[bool, Dict[int, int]]:
    """True + {layer: I'} if within each layer all experts keep the same I'."""
    layer_I: Dict[int, int] = {}
    for layer_idx, m in masks.items():
        counts = {int(m[e].sum().item()) for e in range(m.shape[0])}
        if len(counts) > 1:
            return False, {}
        layer_I[layer_idx] = counts.pop()
    return True, layer_I
