"""Sensitivity-aware EP4 width planning with cross-layer placement.

The planner deliberately keeps the expensive decision out of a global
``layer * expert * tier`` MILP:

1. Reuse the existing score-coverage binary searches to allocate an exact
   channel budget first across layers and then across experts in each layer.
2. Quantize those per-expert counts to four active widths plus zero while
   preserving the nearest globally reachable pruning budget.
3. Assign each layer's four active widths to EP ranks by enumerating its 24
   permutations.  A greedy pass plus local search balances cumulative rank
   load; an exact MILP remains available as an offline reference.

All tensors returned by this module are CPU tensors.  Width zero means that the
global expert remains in router space but has rank/local IDs equal to -1.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any, Dict, Sequence

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix

from src.generate_mask.planners.inter_layer.algo.coverage import (
    binary_search_s_for_target_prune,
    prepare_layer_info,
)
from src.generate_mask.planners.inter_layer.algo.loss_based import (
    smooth_layerwise_loss_with_fn,
)
from src.generate_mask.planners.intra_layer.algo.coverage import (
    _loss_to_layerwise_weights,
    _plan_layer_counts,
)


DEFAULT_WIDTHS = (0, 384, 512, 640, 768)


def _as_finite_float_tensor(value: torch.Tensor, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {tuple(value.shape)}")
    result = value.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _prepare_layer_weights(
    layer_sensitivity: torch.Tensor,
    *,
    smooth_times: int,
    smooth_fn: str,
) -> torch.Tensor:
    if layer_sensitivity.numel() == 1:
        weights = layer_sensitivity.clone().float().clamp_min(0.0)
    else:
        weights = smooth_layerwise_loss_with_fn(
            layer_sensitivity,
            smooth_times=smooth_times,
            smooth_fn=smooth_fn,
        ).clamp_min(0.0)
    if not bool(torch.isfinite(weights).all()) or float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    return weights / weights.mean().clamp_min(1e-12)


def _validate_inputs(
    layer_sensitivity: torch.Tensor,
    expert_sensitivity: torch.Tensor,
    scores: torch.Tensor,
    prune_ratio: float,
    widths: Sequence[int],
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], int]:
    scores = _as_finite_float_tensor(scores, "scores", 3)
    layer_sensitivity = _as_finite_float_tensor(
        layer_sensitivity, "layer_sensitivity", 1
    )
    expert_sensitivity = _as_finite_float_tensor(
        expert_sensitivity, "expert_sensitivity", 2
    )

    num_layers, num_experts, intermediate_size = scores.shape
    if layer_sensitivity.shape != (num_layers,):
        raise ValueError(
            "layer_sensitivity shape must match scores: "
            f"expected {(num_layers,)}, got {tuple(layer_sensitivity.shape)}"
        )
    if expert_sensitivity.shape != (num_layers, num_experts):
        raise ValueError(
            "expert_sensitivity shape must match scores: "
            f"expected {(num_layers, num_experts)}, got {tuple(expert_sensitivity.shape)}"
        )
    if not 0.0 <= float(prune_ratio) <= 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")

    normalized_widths = tuple(int(width) for width in widths)
    if len(set(normalized_widths)) != len(normalized_widths):
        raise ValueError(f"widths must not contain duplicates, got {normalized_widths}")
    if 0 not in normalized_widths:
        raise ValueError("widths must contain zero for removed experts")
    if any(width < 0 or width > intermediate_size for width in normalized_widths):
        raise ValueError(
            f"every width must be in [0, {intermediate_size}], got {normalized_widths}"
        )

    active_widths = tuple(sorted((width for width in normalized_widths if width > 0), reverse=True))
    if len(active_widths) != ep_size:
        raise ValueError(
            f"EP{ep_size} requires exactly {ep_size} active widths, got {active_widths}"
        )
    if active_widths[0] != intermediate_size:
        raise ValueError(
            "the largest active width must equal scores.shape[-1]: "
            f"expected {intermediate_size}, got {active_widths[0]}"
        )
    if num_experts < len(active_widths):
        raise ValueError(
            f"each layer needs at least {len(active_widths)} experts, got {num_experts}"
        )

    unit = 0
    for width in active_widths:
        unit = math.gcd(unit, width)
    if unit <= 0:
        raise ValueError(f"cannot derive an integer width unit from {active_widths}")

    return (
        layer_sensitivity,
        expert_sensitivity,
        scores,
        normalized_widths,
        active_widths,
        unit,
    )


def _match_layer_counts_to_exact_total(
    keep_counts: Dict[int, int],
    layer_info: Dict[int, Any],
    layer_weights: torch.Tensor,
    target_keep: int,
) -> torch.Tensor:
    """Repair the discrete binary-search result using marginal channel scores."""
    num_layers = len(layer_info)
    counts = torch.tensor(
        [int(keep_counts.get(layer, 0)) for layer in range(num_layers)],
        dtype=torch.int64,
    )
    current = int(counts.sum().item())

    if current < target_keep:
        heap: list[tuple[float, int]] = []
        for layer in range(num_layers):
            size = int(layer_info[layer]["size"])
            count = int(counts[layer].item())
            if count < size:
                score = float(layer_info[layer]["sorted"][count].item())
                weighted = score * float(layer_weights[layer].item())
                heapq.heappush(heap, (-weighted, layer))
        while current < target_keep:
            if not heap:
                raise RuntimeError("cannot increase layer counts to the requested total")
            _, layer = heapq.heappop(heap)
            counts[layer] += 1
            current += 1
            count = int(counts[layer].item())
            size = int(layer_info[layer]["size"])
            if count < size:
                score = float(layer_info[layer]["sorted"][count].item())
                weighted = score * float(layer_weights[layer].item())
                heapq.heappush(heap, (-weighted, layer))

    elif current > target_keep:
        heap = []
        for layer in range(num_layers):
            count = int(counts[layer].item())
            if count > 0:
                score = float(layer_info[layer]["sorted"][count - 1].item())
                weighted = score * float(layer_weights[layer].item())
                heapq.heappush(heap, (weighted, layer))
        while current > target_keep:
            if not heap:
                raise RuntimeError("cannot decrease layer counts to the requested total")
            _, layer = heapq.heappop(heap)
            counts[layer] -= 1
            current -= 1
            count = int(counts[layer].item())
            if count > 0:
                score = float(layer_info[layer]["sorted"][count - 1].item())
                weighted = score * float(layer_weights[layer].item())
                heapq.heappush(heap, (weighted, layer))

    if int(counts.sum().item()) != target_keep:
        raise RuntimeError("failed to repair the layer channel budget exactly")
    return counts


def _allocate_raw_channel_counts(
    scores: torch.Tensor,
    layer_sensitivity: torch.Tensor,
    expert_sensitivity: torch.Tensor,
    prune_ratio: float,
    *,
    layer_smooth_times: int,
    layer_smooth_fn: str,
    binary_search_max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the existing layer and expert score-coverage binary searches."""
    num_layers, num_experts, intermediate_size = scores.shape
    total_channels = int(scores.numel())
    target_keep = int(round((1.0 - float(prune_ratio)) * total_channels))

    layer_weights = _prepare_layer_weights(
        layer_sensitivity,
        smooth_times=layer_smooth_times,
        smooth_fn=layer_smooth_fn,
    )

    layer_info, _, _ = prepare_layer_info(scores)
    coverage_targets, keep_counts_dict, _ = binary_search_s_for_target_prune(
        layer_info=layer_info,
        total_channels=total_channels,
        p_target=float(prune_ratio),
        layerwise_loss=layer_weights,
        max_iter=binary_search_max_iter,
        tol=0.0,
    )
    layer_keep_counts = _match_layer_counts_to_exact_total(
        keep_counts=keep_counts_dict,
        layer_info=layer_info,
        layer_weights=layer_weights,
        target_keep=target_keep,
    )
    layer_keep_ratios = layer_keep_counts.float() / float(num_experts * intermediate_size)

    expert_weights = _loss_to_layerwise_weights(expert_sensitivity)
    sorted_indices = torch.argsort(scores, dim=-1, descending=True, stable=True)
    sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices).clamp_min(0.0)
    prefix = torch.cumsum(sorted_scores, dim=-1)
    totals = prefix[:, :, -1]
    raw_counts = torch.zeros((num_layers, num_experts), dtype=torch.int64)

    for layer in range(num_layers):
        raw_counts[layer] = _plan_layer_counts(
            coverage_weight=expert_weights[layer],
            prefix=prefix[layer],
            totals=totals[layer],
            sorted_vals=sorted_scores[layer],
            target_keep=int(layer_keep_counts[layer].item()),
            max_iter=binary_search_max_iter,
        )

    if int(raw_counts.sum().item()) != target_keep:
        raise RuntimeError(
            "expert coverage allocation did not preserve the exact layer budgets: "
            f"expected {target_keep}, got {int(raw_counts.sum().item())}"
        )

    if isinstance(coverage_targets, torch.Tensor):
        coverage_targets = coverage_targets.detach().cpu().float()
    else:
        coverage_targets = torch.full((num_layers,), float(coverage_targets))
    return (
        raw_counts,
        layer_keep_counts,
        layer_keep_ratios,
        coverage_targets,
        sorted_indices,
    )


def _prefix_value(prefix: torch.Tensor, layer: int, expert: int, width: int) -> float:
    if width <= 0:
        return 0.0
    return float(prefix[layer, expert, width - 1].item())


def _quantize_balanced_widths_to_budget(
    raw_counts: torch.Tensor,
    active_widths: tuple[int, ...],
    unit: int,
    target_keep_channels: int,
) -> tuple[torch.Tensor, int]:
    """Build an exact-budget plan with balanced active-tier expert counts.

    This mode is intended for performance experiments where balanced fused-MoE
    group sizes matter more than preserving the score-derived width histogram.
    Expert identities are still assigned monotonically by their raw keep counts.
    """
    num_layers, num_experts = raw_counts.shape
    active_ascending = tuple(sorted(active_widths))
    tier_units = tuple(width // unit for width in active_ascending)
    if any(width % unit != 0 for width in active_ascending):
        raise ValueError(f"active widths must be divisible by width unit {unit}")
    if target_keep_channels % unit != 0:
        raise ValueError(
            f"target_keep_channels={target_keep_channels} is not divisible by width unit {unit}"
        )

    target_units = target_keep_channels // unit
    num_tiers = len(active_ascending)
    total_experts = num_layers * num_experts
    best_counts: tuple[int, ...] | None = None
    best_key: tuple[int, int] | None = None
    minimum_active = num_layers * num_tiers

    # For a fixed active-expert total, the most balanced tier counts are q or
    # q+1. Enumerating which tiers receive the remainder gives the exact global
    # channel budget without a large integer program.
    for active_total in range(minimum_active, total_experts + 1):
        base_count, remainder = divmod(active_total, num_tiers)
        for extra_tiers in itertools.combinations(range(num_tiers), remainder):
            counts = [base_count] * num_tiers
            for tier in extra_tiers:
                counts[tier] += 1
            if min(counts) < num_layers:
                continue
            keep_units = sum(count * width for count, width in zip(counts, tier_units))
            if keep_units != target_units:
                continue
            key = (max(counts) - min(counts), -active_total)
            if best_key is None or key < best_key:
                best_key = key
                best_counts = tuple(counts)

    if best_counts is None:
        raise ValueError(
            "the pruning target cannot be represented with globally balanced active tiers: "
            f"target_keep={target_keep_channels}, layers={num_layers}, experts={num_experts}, "
            f"active_widths={active_ascending}"
        )

    # Spread each tier's remainder across layers while keeping the total number
    # of active experts per layer balanced. Raw layer budgets only break ties.
    layer_tier_counts = torch.tensor(
        [[count // num_layers for count in best_counts] for _ in range(num_layers)],
        dtype=torch.int64,
    )
    row_extras = torch.zeros(num_layers, dtype=torch.int64)
    raw_layer_units = raw_counts.sum(dim=1, dtype=torch.int64).float() / float(unit)
    base_layer_units = (layer_tier_counts * torch.tensor(tier_units)).sum(dim=1).float()
    residual_units = raw_layer_units - base_layer_units
    tier_order = sorted(
        range(num_tiers),
        key=lambda tier: (-(best_counts[tier] % num_layers), -tier_units[tier], tier),
    )
    for tier in tier_order:
        remainder = best_counts[tier] % num_layers
        candidates = sorted(
            range(num_layers),
            key=lambda layer: (
                int(row_extras[layer].item()),
                -float(residual_units[layer].item()),
                layer,
            ),
        )
        for layer in candidates[:remainder]:
            layer_tier_counts[layer, tier] += 1
            row_extras[layer] += 1
            residual_units[layer] -= tier_units[tier]

    expected_global = torch.tensor(best_counts, dtype=torch.int64)
    if not layer_tier_counts.sum(dim=0).equal(expected_global):
        raise RuntimeError("balanced tier scheduler did not preserve global tier counts")
    per_layer_spread = layer_tier_counts.max(dim=1).values - layer_tier_counts.min(dim=1).values
    if int(per_layer_spread.max().item()) > 1:
        raise RuntimeError("balanced tier scheduler produced a per-layer tier-count spread above 1")

    widths = torch.zeros_like(raw_counts)
    for layer in range(num_layers):
        slots: list[int] = []
        for tier, width in enumerate(active_ascending):
            slots.extend([width] * int(layer_tier_counts[layer, tier].item()))
        slots.extend([0] * (num_experts - len(slots)))
        if len(slots) != num_experts:
            raise RuntimeError(f"balanced tier scheduler overfilled layer {layer}")
        expert_order = torch.argsort(raw_counts[layer], stable=True)
        widths[layer, expert_order] = torch.tensor(sorted(slots), dtype=widths.dtype)

    final_keep = int(widths.sum().item())
    if final_keep != target_keep_channels:
        raise RuntimeError(
            f"balanced tier budget mismatch: expected {target_keep_channels}, got {final_keep}"
        )
    return widths, final_keep


def _quantize_widths_to_budget(
    raw_counts: torch.Tensor,
    sorted_scores: torch.Tensor,
    layer_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    active_widths: tuple[int, ...],
    unit: int,
    target_keep_channels: int,
    balance_tier_counts: bool = False,
) -> tuple[torch.Tensor, int]:
    """Quantize by mandatory-tier anchors followed by marginal-cost upgrades."""
    if balance_tier_counts:
        return _quantize_balanced_widths_to_budget(
            raw_counts=raw_counts,
            active_widths=active_widths,
            unit=unit,
            target_keep_channels=target_keep_channels,
        )

    num_layers, num_experts = raw_counts.shape
    active_ascending = tuple(sorted(active_widths))
    upgrade_widths = (0,) + active_ascending
    width_to_upgrade_index = {width: index for index, width in enumerate(upgrade_widths)}

    min_keep = num_layers * sum(active_widths)
    max_keep = num_layers * (
        sum(active_widths) + (num_experts - len(active_widths)) * max(active_widths)
    )
    if not min_keep <= target_keep_channels <= max_keep:
        raise ValueError(
            "the pruning target is infeasible when every active tier must appear in every layer: "
            f"target_keep={target_keep_channels}, feasible=[{min_keep}, {max_keep}]"
        )

    widths = torch.zeros_like(raw_counts)
    anchors = torch.zeros_like(raw_counts, dtype=torch.bool)
    for layer in range(num_layers):
        raw = raw_counts[layer].numpy().astype(np.float64, copy=False)
        anchor_cost = np.abs(
            np.asarray(active_ascending, dtype=np.float64)[:, None] - raw[None, :]
        )
        row_ids, expert_ids = linear_sum_assignment(anchor_cost)
        for row_id, expert_id in zip(row_ids.tolist(), expert_ids.tolist()):
            widths[layer, expert_id] = active_ascending[row_id]
            anchors[layer, expert_id] = True

    current_keep = int(widths.sum().item())
    remaining_units = (target_keep_channels - current_keep) // unit
    if current_keep + remaining_units * unit != target_keep_channels:
        raise ValueError(
            f"target_keep_channels={target_keep_channels} is not divisible by width unit {unit}"
        )
    min_expert_units = min(active_widths) // unit
    if 0 < remaining_units < min_expert_units:
        raise ValueError(
            "the requested discrete budget is unreachable: "
            f"{remaining_units} extra width units cannot create an additional expert "
            f"whose minimum active width is {min_expert_units} units"
        )

    prefix = torch.cumsum(sorted_scores.clamp_min(0.0), dim=-1)
    heap: list[tuple[float, float, int, int, int]] = []

    def push_next(layer: int, expert: int) -> None:
        old_width = int(widths[layer, expert].item())
        old_index = width_to_upgrade_index[old_width]
        if old_index + 1 >= len(upgrade_widths):
            return
        new_width = upgrade_widths[old_index + 1]
        delta = new_width - old_width
        raw = int(raw_counts[layer, expert].item())
        marginal_distance = abs(new_width - raw) - abs(old_width - raw)
        added_score = (
            _prefix_value(prefix, layer, expert, new_width)
            - _prefix_value(prefix, layer, expert, old_width)
        )
        sensitivity_scale = max(
            float(layer_weights[layer].item()) * float(expert_weights[layer, expert].item()),
            1e-12,
        )
        heapq.heappush(
            heap,
            (
                float(marginal_distance) / float(delta),
                -added_score * sensitivity_scale / float(delta),
                layer,
                expert,
                new_width,
            ),
        )

    for layer in range(num_layers):
        for expert in range(num_experts):
            if not bool(anchors[layer, expert]):
                push_next(layer, expert)

    while remaining_units > 0:
        deferred: list[tuple[float, float, int, int, int]] = []
        selected = None
        while heap:
            candidate = heapq.heappop(heap)
            _, _, layer, expert, new_width = candidate
            old_width = int(widths[layer, expert].item())
            delta_units = (new_width - old_width) // unit
            if delta_units <= remaining_units:
                selected = candidate
                break
            deferred.append(candidate)
        for candidate in deferred:
            heapq.heappush(heap, candidate)
        if selected is None:
            raise RuntimeError(
                "tier quantization could not reach the exact discrete budget; "
                f"remaining_units={remaining_units}"
            )

        _, _, layer, expert, new_width = selected
        old_width = int(widths[layer, expert].item())
        delta_units = (new_width - old_width) // unit
        widths[layer, expert] = new_width
        remaining_units -= delta_units
        push_next(layer, expert)

    final_keep = int(widths.sum().item())
    if final_keep != target_keep_channels:
        raise RuntimeError(
            f"tier quantization budget mismatch: expected {target_keep_channels}, got {final_keep}"
        )
    for layer in range(num_layers):
        for width in active_widths:
            if int((widths[layer] == width).sum().item()) < 1:
                raise RuntimeError(f"layer {layer} has no expert in mandatory width tier {width}")
    return widths, final_keep


def _merge_sparse_width_tiers(
    expert_widths: torch.Tensor,
    raw_counts: torch.Tensor,
    active_widths: Sequence[int],
    *,
    max_experts: int = 5,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Remove per-layer width tiers that contain at most ``max_experts``.

    Sparse tiers are merged into the nearest surviving adjacent tier according
    to each expert's pre-quantization channel count.  Zero is a valid lower
    destination.  The exact discrete budget is intentionally allowed to drift:
    avoiding tiny kernels is more important than preserving a handful of width
    units after quantization.
    """
    if max_experts < 0:
        raise ValueError(f"max_experts must be non-negative, got {max_experts}")
    widths = expert_widths.detach().cpu().to(torch.int64).clone()
    raw = raw_counts.detach().cpu().to(torch.int64)
    if widths.shape != raw.shape:
        raise ValueError("expert_widths and raw_counts must have the same shape")
    if max_experts == 0:
        return widths, []

    tiers = tuple(sorted(int(width) for width in active_widths))
    diagnostics: list[dict[str, Any]] = []
    for layer in range(widths.shape[0]):
        counts = {width: int((widths[layer] == width).sum()) for width in tiers}
        present = [width for width in tiers if counts[width] > 0]
        surviving = [width for width in present if counts[width] > max_experts]
        if not present or len(surviving) == len(present):
            continue
        if not surviving:
            # Degenerate synthetic/small-model fallback: retain the tier with
            # the most work so the layer still has active experts to place.
            surviving = [max(present, key=lambda width: (counts[width] * width, width))]

        destinations = (0, *surviving)
        for width in present:
            if width in surviving:
                continue
            expert_ids = torch.where(widths[layer] == width)[0].tolist()
            moved: dict[int, int] = {}
            lower = max((candidate for candidate in destinations if candidate < width), default=None)
            higher = min((candidate for candidate in destinations if candidate > width), default=None)
            for expert in expert_ids:
                raw_width = int(raw[layer, expert])
                if lower is None:
                    destination = int(higher)
                elif higher is None:
                    destination = int(lower)
                else:
                    lower_distance = abs(raw_width - lower)
                    higher_distance = abs(raw_width - higher)
                    if lower_distance == higher_distance:
                        destination = higher if raw_width >= width else lower
                    else:
                        destination = lower if lower_distance < higher_distance else higher
                widths[layer, expert] = destination
                moved[destination] = moved.get(destination, 0) + 1
            diagnostics.append(
                {
                    "layer": layer,
                    "removed_width": width,
                    "removed_count": len(expert_ids),
                    "destinations": moved,
                }
            )

    return widths, diagnostics


def _build_layer_placement_groups(
    expert_widths: torch.Tensor,
    active_widths: Sequence[int],
    *,
    ep_size: int,
) -> Dict[str, Any]:
    """Create exactly ``ep_size`` non-empty groups per layer.

    Each present width starts as one group.  When a layer has fewer width tiers
    than EP ranks, the group with the largest ``count * width`` is split as
    evenly as possible by expert count.  This lets multiple ranks execute the
    same width without leaving a rank idle.
    """
    widths = expert_widths.detach().cpu().to(torch.int64)
    allowed = {int(width) for width in active_widths}
    num_layers, num_experts = widths.shape
    group_widths = torch.empty((num_layers, ep_size), dtype=torch.int64)
    group_counts = torch.empty((num_layers, ep_size), dtype=torch.int64)
    expert_to_group = torch.full((num_layers, num_experts), -1, dtype=torch.int64)
    layer_groups: list[list[list[int]]] = []

    for layer in range(num_layers):
        groups = [
            {
                "width": width,
                "experts": torch.where(widths[layer] == width)[0].tolist(),
            }
            for width in sorted(allowed, reverse=True)
            if bool((widths[layer] == width).any())
        ]
        while len(groups) < ep_size:
            splittable = [index for index, group in enumerate(groups) if len(group["experts"]) >= 2]
            if not splittable:
                raise ValueError(
                    f"layer {layer} has only {sum(len(group['experts']) for group in groups)} "
                    f"active experts and cannot populate {ep_size} EP ranks"
                )
            split_index = max(
                splittable,
                key=lambda index: (
                    len(groups[index]["experts"]) * int(groups[index]["width"]),
                    len(groups[index]["experts"]),
                    int(groups[index]["width"]),
                    -index,
                ),
            )
            group = groups.pop(split_index)
            expert_ids = list(group["experts"])
            midpoint = (len(expert_ids) + 1) // 2
            groups.insert(split_index, {"width": group["width"], "experts": expert_ids[:midpoint]})
            groups.insert(split_index + 1, {"width": group["width"], "experts": expert_ids[midpoint:]})

        if len(groups) != ep_size:
            raise ValueError(
                f"layer {layer} has {len(groups)} active width tiers, exceeding EP size {ep_size}"
            )
        layer_group_ids: list[list[int]] = []
        for group_index, group in enumerate(groups):
            expert_ids = list(group["experts"])
            group_widths[layer, group_index] = int(group["width"])
            group_counts[layer, group_index] = len(expert_ids)
            expert_to_group[layer, expert_ids] = group_index
            layer_group_ids.append(expert_ids)
        layer_groups.append(layer_group_ids)

    return {
        "placement_group_widths": group_widths,
        "placement_group_counts": group_counts,
        "expert_to_placement_group": expert_to_group,
        "placement_groups": layer_groups,
    }


def _solve_placement_groups_milp(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
) -> Dict[str, Any]:
    """Assign arbitrary per-layer placement groups using an exact MILP."""
    counts = group_counts.detach().cpu().to(torch.int64)
    widths = group_widths.detach().cpu().to(torch.int64)
    if counts.ndim != 2 or widths.shape != counts.shape:
        raise ValueError("group_counts and group_widths must have the same 2D shape")
    num_layers, ep_size = counts.shape
    if ep_size <= 0 or ep_size > 7:
        raise ValueError(f"unsupported EP size: {ep_size}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if bool((counts <= 0).any()) or bool((widths <= 0).any()):
        raise ValueError("every placement group must have positive count and width")

    permutations = tuple(itertools.permutations(range(ep_size)))
    num_permutations = len(permutations)
    num_binary = num_layers * num_permutations
    deviation_index = num_binary
    num_variables = num_binary + 1

    group_loads = (counts * widths).numpy()
    layer_rank_load = np.zeros((num_layers, num_permutations, ep_size), dtype=np.float64)
    for layer in range(num_layers):
        for permutation_id, permutation in enumerate(permutations):
            for rank, group_index in enumerate(permutation):
                layer_rank_load[layer, permutation_id, rank] = group_loads[layer, group_index]

    total_load = float(group_loads.sum())
    mean_load = total_load / float(ep_size)

    constraint_rows = num_layers + 2 * ep_size
    matrix = lil_matrix((constraint_rows, num_variables), dtype=np.float64)
    lower = np.full(constraint_rows, -np.inf, dtype=np.float64)
    upper = np.full(constraint_rows, np.inf, dtype=np.float64)

    row = 0
    for layer in range(num_layers):
        start = layer * num_permutations
        matrix[row, start : start + num_permutations] = 1.0
        lower[row] = 1.0
        upper[row] = 1.0
        row += 1

    for rank in range(ep_size):
        for layer in range(num_layers):
            start = layer * num_permutations
            matrix[row, start : start + num_permutations] = layer_rank_load[layer, :, rank]
        matrix[row, deviation_index] = -1.0
        upper[row] = mean_load
        row += 1

        for layer in range(num_layers):
            start = layer * num_permutations
            matrix[row, start : start + num_permutations] = -layer_rank_load[layer, :, rank]
        matrix[row, deviation_index] = -1.0
        upper[row] = -mean_load
        row += 1

    bounds_lower = np.zeros(num_variables, dtype=np.float64)
    bounds_upper = np.ones(num_variables, dtype=np.float64)
    bounds_upper[deviation_index] = np.inf
    if fix_first_layer and num_layers > 0:
        identity_id = permutations.index(tuple(range(ep_size)))
        bounds_lower[:num_permutations] = 0.0
        bounds_upper[:num_permutations] = 0.0
        bounds_lower[identity_id] = 1.0
        bounds_upper[identity_id] = 1.0

    objective = np.zeros(num_variables, dtype=np.float64)
    objective[deviation_index] = 1.0
    integrality = np.ones(num_variables, dtype=np.int8)
    integrality[deviation_index] = 0

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"cross-layer placement MILP failed: status={result.status}, message={result.message}"
        )

    permutation_ids = []
    rank_widths = torch.empty_like(widths)
    group_to_rank = torch.empty_like(counts)
    rank_group_indices = torch.empty_like(counts)
    for layer in range(num_layers):
        start = layer * num_permutations
        values = result.x[start : start + num_permutations]
        permutation_id = int(np.argmax(values))
        if values[permutation_id] < 0.5:
            raise RuntimeError(f"layer {layer} has no integral placement permutation")
        permutation_ids.append(permutation_id)
        permutation = permutations[permutation_id]
        for rank, group_index in enumerate(permutation):
            rank_group_indices[layer, rank] = group_index
            rank_widths[layer, rank] = widths[layer, group_index]
            group_to_rank[layer, group_index] = rank

    rank_loads = torch.zeros(ep_size, dtype=torch.int64)
    for layer in range(num_layers):
        for rank in range(ep_size):
            group_index = int(rank_group_indices[layer, rank])
            rank_loads[rank] += counts[layer, group_index] * widths[layer, group_index]

    mean = float(rank_loads.double().mean().item())
    max_deviation = float((rank_loads.double() - mean).abs().max().item())
    relative_deviation = max_deviation / mean if mean > 0.0 else 0.0
    return {
        "rank_widths": rank_widths,
        "group_to_rank": group_to_rank,
        "rank_group_indices": rank_group_indices,
        "rank_weight_loads": rank_loads,
        "mean_rank_weight_load": mean,
        "max_rank_weight_deviation": max_deviation,
        "relative_max_rank_weight_deviation": relative_deviation,
        "tolerance": float(tolerance),
        "tolerance_satisfied": relative_deviation <= float(tolerance) + 1e-12,
        "permutation_ids": torch.tensor(permutation_ids, dtype=torch.int64),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_objective": float(result.fun),
    }


def _solve_cross_layer_placement_milp(
    width_counts: torch.Tensor,
    active_widths: Sequence[int],
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
) -> Dict[str, Any]:
    """Assign active width groups to EP ranks with an exact placement MILP."""
    if not isinstance(width_counts, torch.Tensor) or width_counts.ndim != 2:
        raise ValueError("width_counts must be a [L, num_active_widths] tensor")
    counts = width_counts.detach().cpu().to(torch.int64)
    widths = tuple(int(width) for width in active_widths)
    num_layers, num_widths = counts.shape
    if num_widths != len(widths):
        raise ValueError(
            f"width_counts has {num_widths} columns but active_widths has {len(widths)}"
        )
    if bool((counts <= 0).any()):
        raise ValueError("every active width tier must contain at least one expert per layer")
    result = _solve_placement_groups_milp(
        counts,
        torch.tensor(widths, dtype=torch.int64).expand(num_layers, -1),
        tolerance=tolerance,
        fix_first_layer=fix_first_layer,
    )
    result["tier_to_rank"] = result["group_to_rank"]
    return result


def _placement_objective(rank_loads: np.ndarray) -> tuple[float, float]:
    centered = rank_loads - rank_loads.mean()
    return float(np.abs(centered).max()), float(np.dot(centered, centered))


def _solve_placement_groups_greedy(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    max_local_search_passes: int = 100,
) -> Dict[str, Any]:
    """Balance per-layer placement groups across a fixed number of EP ranks."""
    counts = group_counts.detach().cpu().to(torch.int64)
    widths = group_widths.detach().cpu().to(torch.int64)
    if counts.ndim != 2 or widths.shape != counts.shape:
        raise ValueError("group_counts and group_widths must have the same 2D shape")
    if bool((counts <= 0).any()) or bool((widths <= 0).any()):
        raise ValueError("every placement group must have positive count and width")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if max_local_search_passes < 0:
        raise ValueError(
            "max_local_search_passes must be non-negative, got "
            f"{max_local_search_passes}"
        )

    num_layers, ep_size = counts.shape
    permutations = tuple(itertools.permutations(range(ep_size)))
    identity_id = permutations.index(tuple(range(ep_size)))
    group_loads = (counts * widths).numpy()
    layer_rank_load = np.stack(
        [group_loads[:, permutation] for permutation in permutations], axis=1
    )

    permutation_ids = np.full(num_layers, -1, dtype=np.int64)
    rank_loads = np.zeros(ep_size, dtype=np.int64)
    first_unfixed_layer = 0
    if fix_first_layer and num_layers > 0:
        permutation_ids[0] = identity_id
        rank_loads += layer_rank_load[0, identity_id]
        first_unfixed_layer = 1

    layer_order = sorted(
        range(first_unfixed_layer, num_layers),
        key=lambda layer: (-int(np.ptp(group_loads[layer])), layer),
    )
    for layer in layer_order:
        best_id = min(
            range(len(permutations)),
            key=lambda permutation_id: (
                *_placement_objective(rank_loads + layer_rank_load[layer, permutation_id]),
                permutation_id,
            ),
        )
        permutation_ids[layer] = best_id
        rank_loads += layer_rank_load[layer, best_id]

    local_search_passes = 0
    for _ in range(max_local_search_passes):
        improved = False
        for layer in range(first_unfixed_layer, num_layers):
            current_id = int(permutation_ids[layer])
            base_loads = rank_loads - layer_rank_load[layer, current_id]
            best_id = current_id
            best_objective = _placement_objective(rank_loads)
            for candidate_id in range(len(permutations)):
                candidate_objective = _placement_objective(
                    base_loads + layer_rank_load[layer, candidate_id]
                )
                if candidate_objective < best_objective:
                    best_id = candidate_id
                    best_objective = candidate_objective
            if best_id != current_id:
                rank_loads = base_loads + layer_rank_load[layer, best_id]
                permutation_ids[layer] = best_id
                improved = True
        local_search_passes += 1
        if not improved:
            break

    rank_widths = torch.empty_like(widths)
    group_to_rank = torch.empty_like(counts)
    rank_group_indices = torch.empty_like(counts)
    for layer, permutation_id in enumerate(permutation_ids.tolist()):
        permutation = permutations[permutation_id]
        for rank, group_index in enumerate(permutation):
            rank_group_indices[layer, rank] = group_index
            rank_widths[layer, rank] = widths[layer, group_index]
            group_to_rank[layer, group_index] = rank

    rank_loads_tensor = torch.from_numpy(rank_loads.copy())
    mean = float(rank_loads_tensor.double().mean().item())
    max_deviation = float((rank_loads_tensor.double() - mean).abs().max().item())
    relative_deviation = max_deviation / mean if mean > 0.0 else 0.0
    return {
        "rank_widths": rank_widths,
        "group_to_rank": group_to_rank,
        "rank_group_indices": rank_group_indices,
        "rank_weight_loads": rank_loads_tensor,
        "mean_rank_weight_load": mean,
        "max_rank_weight_deviation": max_deviation,
        "relative_max_rank_weight_deviation": relative_deviation,
        "tolerance": float(tolerance),
        "tolerance_satisfied": relative_deviation <= float(tolerance) + 1e-12,
        "permutation_ids": torch.from_numpy(permutation_ids.copy()),
        "solver_status": 0,
        "solver_message": "greedy placement-group assignment with coordinate-descent local search",
        "solver_objective": max_deviation,
        "placement_method": "greedy",
        "local_search_passes": local_search_passes,
    }


def _solve_cross_layer_placement_greedy(
    width_counts: torch.Tensor,
    active_widths: Sequence[int],
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    max_local_search_passes: int = 100,
) -> Dict[str, Any]:
    """Balance EP rank loads using 24-way greedy placement and local search."""
    if not isinstance(width_counts, torch.Tensor) or width_counts.ndim != 2:
        raise ValueError("width_counts must be a [L, num_active_widths] tensor")
    counts = width_counts.detach().cpu().to(torch.int64)
    widths = tuple(int(width) for width in active_widths)
    num_layers, num_widths = counts.shape
    if num_widths != len(widths):
        raise ValueError(
            f"width_counts has {num_widths} columns but active_widths has {len(widths)}"
        )
    if num_widths <= 0 or num_widths > 7:
        raise ValueError(f"unsupported number of active widths: {num_widths}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if max_local_search_passes < 0:
        raise ValueError(
            "max_local_search_passes must be non-negative, got "
            f"{max_local_search_passes}"
        )
    if bool((counts <= 0).any()):
        raise ValueError("every active width tier must contain at least one expert per layer")

    result = _solve_placement_groups_greedy(
        counts,
        torch.tensor(widths, dtype=torch.int64).expand(num_layers, -1),
        tolerance=tolerance,
        fix_first_layer=fix_first_layer,
        max_local_search_passes=max_local_search_passes,
    )
    result["tier_to_rank"] = result["group_to_rank"]
    result["solver_message"] = "greedy placement with coordinate-descent local search"
    return result


def solve_cross_layer_placement(
    width_counts: torch.Tensor,
    active_widths: Sequence[int],
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    method: str = "greedy",
    max_local_search_passes: int = 100,
) -> Dict[str, Any]:
    """Assign each layer's width groups to EP ranks.

    ``greedy`` is the production default.  ``milp`` is retained for small
    instances and optimality comparisons because its runtime grows quickly.
    """
    normalized_method = method.lower()
    if normalized_method == "greedy":
        return _solve_cross_layer_placement_greedy(
            width_counts,
            active_widths,
            tolerance=tolerance,
            fix_first_layer=fix_first_layer,
            max_local_search_passes=max_local_search_passes,
        )
    if normalized_method == "milp":
        result = _solve_cross_layer_placement_milp(
            width_counts,
            active_widths,
            tolerance=tolerance,
            fix_first_layer=fix_first_layer,
        )
        result["placement_method"] = "milp"
        return result
    raise ValueError(
        f"unsupported placement method {method!r}; expected 'greedy' or 'milp'"
    )


def _build_masks_and_mappings(
    expert_widths: torch.Tensor,
    sorted_indices: torch.Tensor,
    expert_to_group: torch.Tensor,
    group_to_rank: torch.Tensor,
    ep_size: int,
) -> Dict[str, Any]:
    num_layers, num_experts = expert_widths.shape
    intermediate_size = sorted_indices.shape[-1]
    masks = torch.zeros(
        (num_layers, num_experts, intermediate_size), dtype=torch.bool
    )
    expert_to_rank = torch.full((num_layers, num_experts), -1, dtype=torch.int64)
    expert_to_local_id = torch.full((num_layers, num_experts), -1, dtype=torch.int64)
    local_to_global: list[list[list[int]]] = []

    for layer in range(num_layers):
        layer_local_to_global: list[list[int]] = [list() for _ in range(ep_size)]
        for expert in range(num_experts):
            width = int(expert_widths[layer, expert].item())
            if width <= 0:
                continue
            chosen = sorted_indices[layer, expert, :width]
            masks[layer, expert].index_fill_(0, chosen, True)
            group = int(expert_to_group[layer, expert].item())
            if group < 0:
                raise RuntimeError(f"active expert {expert} in layer {layer} has no placement group")
            rank = int(group_to_rank[layer, group].item())
            local_id = len(layer_local_to_global[rank])
            expert_to_rank[layer, expert] = rank
            expert_to_local_id[layer, expert] = local_id
            layer_local_to_global[rank].append(expert)
        local_to_global.append(layer_local_to_global)

    return {
        "intermediate_masks": masks,
        "expert_to_rank": expert_to_rank,
        "expert_to_local_id": expert_to_local_id,
        "local_to_global": local_to_global,
    }


def _finalize_ep4_structure(
    expert_widths: torch.Tensor,
    raw_counts: torch.Tensor,
    sorted_indices: torch.Tensor,
    active_widths: tuple[int, ...],
    *,
    ep_size: int,
    sparse_tier_max_experts: int,
    placement_tolerance: float,
    fix_first_layer_placement: bool,
    placement_method: str,
    placement_local_search_passes: int,
) -> tuple[torch.Tensor, int, list[dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    expert_widths, tier_merges = _merge_sparse_width_tiers(
        expert_widths,
        raw_counts,
        active_widths,
        max_experts=sparse_tier_max_experts,
    )
    actual_keep_channels = int(expert_widths.sum().item())
    groups = _build_layer_placement_groups(
        expert_widths,
        active_widths,
        ep_size=ep_size,
    )
    normalized_method = placement_method.lower()
    if normalized_method == "greedy":
        placement = _solve_placement_groups_greedy(
            groups["placement_group_counts"],
            groups["placement_group_widths"],
            tolerance=placement_tolerance,
            fix_first_layer=fix_first_layer_placement,
            max_local_search_passes=placement_local_search_passes,
        )
    elif normalized_method == "milp":
        placement = _solve_placement_groups_milp(
            groups["placement_group_counts"],
            groups["placement_group_widths"],
            tolerance=placement_tolerance,
            fix_first_layer=fix_first_layer_placement,
        )
        placement["placement_method"] = "milp"
    else:
        raise ValueError(
            f"unsupported placement method {placement_method!r}; expected 'greedy' or 'milp'"
        )

    tier_to_rank = torch.full(
        (expert_widths.shape[0], len(active_widths)), -1, dtype=torch.int64
    )
    tier_to_ranks: list[list[list[int]]] = []
    for layer in range(expert_widths.shape[0]):
        layer_ranks: list[list[int]] = []
        for tier_index, width in enumerate(active_widths):
            ranks = torch.where(placement["rank_widths"][layer] == width)[0].tolist()
            if bool((expert_widths[layer] == width).any()) and not ranks:
                raise RuntimeError(f"layer {layer} width {width} has no assigned EP rank")
            if ranks:
                tier_to_rank[layer, tier_index] = ranks[0]
            layer_ranks.append(ranks)
        tier_to_ranks.append(layer_ranks)
    placement["tier_to_rank"] = tier_to_rank
    placement["tier_to_ranks"] = tier_to_ranks

    mapping = _build_masks_and_mappings(
        expert_widths=expert_widths,
        sorted_indices=sorted_indices,
        expert_to_group=groups["expert_to_placement_group"],
        group_to_rank=placement["group_to_rank"],
        ep_size=ep_size,
    )
    return expert_widths, actual_keep_channels, tier_merges, groups, placement, mapping


@torch.no_grad()
def plan_ep4_intplan(
    layer_sensitivity: torch.Tensor,
    expert_sensitivity: torch.Tensor,
    scores: torch.Tensor,
    prune_ratio: float,
    *,
    widths: Sequence[int] = DEFAULT_WIDTHS,
    ep_size: int = 4,
    placement_tolerance: float = 0.01,
    strict_placement_tolerance: bool = False,
    placement_method: str = "greedy",
    placement_local_search_passes: int = 100,
    layer_smooth_times: int = 2,
    layer_smooth_fn: str = "sqrt",
    binary_search_max_iter: int = 32,
    fix_first_layer_placement: bool = True,
    balance_tier_counts: bool = False,
    sparse_tier_max_experts: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Build an EP4 width and placement plan.

    Args:
        layer_sensitivity: One scalar per layer, shape ``[L]``.
        expert_sensitivity: One scalar per layer/expert, shape ``[L, E]``.
        scores: Per-channel importance, shape ``[L, E, I]``.
        prune_ratio: Fraction of expert intermediate channels to remove.  This
            is not a keep ratio; ``keep_ratio = 1 - prune_ratio``.
        widths: Four active widths plus zero.  The default is
            ``(0, 384, 512, 640, 768)``.
        placement_tolerance: Allowed relative deviation from mean cumulative
            rank weight load.
        strict_placement_tolerance: Raise if the best placement exceeds the
            tolerance.  The default returns the best plan plus a false status.
        placement_method: ``"greedy"`` for fast production planning or
            ``"milp"`` for an exact small-instance/offline reference.
        placement_local_search_passes: Maximum coordinate-descent passes used
            by the greedy placement method.
        balance_tier_counts: Force the four active width tiers to contain as
            close to the same number of experts as the exact budget permits.
            This is useful for performance-only fused-MoE experiments.
        sparse_tier_max_experts: Merge a nonzero per-layer width tier when it
            contains at most this many experts. Set to zero to retain the old
            mandatory-four-tier behavior.

    Returns:
        A dictionary containing raw/quantized channel counts, masks, per-layer
        rank permutations, global/local mappings, and budget/load diagnostics.
    """
    (
        layer_sensitivity,
        expert_sensitivity,
        scores,
        normalized_widths,
        active_widths,
        unit,
    ) = _validate_inputs(
        layer_sensitivity=layer_sensitivity,
        expert_sensitivity=expert_sensitivity,
        scores=scores,
        prune_ratio=prune_ratio,
        widths=widths,
        ep_size=ep_size,
    )
    num_layers, num_experts, intermediate_size = scores.shape

    layer_weights = _prepare_layer_weights(
        layer_sensitivity,
        smooth_times=layer_smooth_times,
        smooth_fn=layer_smooth_fn,
    )
    expert_weights = _loss_to_layerwise_weights(expert_sensitivity)

    (
        raw_counts,
        layer_keep_counts,
        layer_keep_ratios,
        layer_coverage_targets,
        sorted_indices,
    ) = _allocate_raw_channel_counts(
        scores=scores,
        layer_sensitivity=layer_sensitivity,
        expert_sensitivity=expert_sensitivity,
        prune_ratio=prune_ratio,
        layer_smooth_times=layer_smooth_times,
        layer_smooth_fn=layer_smooth_fn,
        binary_search_max_iter=binary_search_max_iter,
    )

    sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices).clamp_min(0.0)
    total_channels = num_layers * num_experts * intermediate_size
    continuous_target_keep = (1.0 - float(prune_ratio)) * total_channels
    target_keep_units = int(round(continuous_target_keep / unit))
    target_keep_channels = target_keep_units * unit
    expert_widths, _ = _quantize_widths_to_budget(
        raw_counts=raw_counts,
        sorted_scores=sorted_scores,
        layer_weights=layer_weights,
        expert_weights=expert_weights,
        active_widths=active_widths,
        unit=unit,
        target_keep_channels=target_keep_channels,
        balance_tier_counts=balance_tier_counts,
    )

    (
        expert_widths,
        actual_keep_channels,
        tier_merges,
        placement_groups,
        placement,
        mapping,
    ) = _finalize_ep4_structure(
        expert_widths,
        raw_counts,
        sorted_indices,
        active_widths,
        ep_size=ep_size,
        sparse_tier_max_experts=sparse_tier_max_experts,
        placement_tolerance=placement_tolerance,
        fix_first_layer_placement=fix_first_layer_placement,
        placement_method=placement_method,
        placement_local_search_passes=placement_local_search_passes,
    )

    all_widths = tuple(sorted(normalized_widths, reverse=True))
    width_counts = torch.stack(
        [
            torch.stack(
                [(expert_widths[layer] == width).sum() for width in all_widths]
            )
            for layer in range(num_layers)
        ]
    ).to(torch.int64)
    active_columns = [all_widths.index(width) for width in active_widths]
    active_width_counts = width_counts[:, active_columns]

    if strict_placement_tolerance and not placement["tolerance_satisfied"]:
        raise RuntimeError(
            "best cross-layer placement exceeds tolerance: "
            f"relative_deviation={placement['relative_max_rank_weight_deviation']:.6f}, "
            f"tolerance={placement_tolerance:.6f}"
        )

    actual_prune_ratio = 1.0 - float(actual_keep_channels) / float(total_channels)

    result: Dict[str, Any] = {
        "prune_ratio": float(prune_ratio),
        "keep_ratio": 1.0 - float(prune_ratio),
        "actual_prune_ratio": actual_prune_ratio,
        "actual_keep_ratio": 1.0 - actual_prune_ratio,
        "target_keep_channels": int(target_keep_channels),
        "actual_keep_channels": int(actual_keep_channels),
        "width_unit": int(unit),
        "widths": all_widths,
        "active_widths": active_widths,
        "raw_K_E_inter": raw_counts,
        "K_E_inter": expert_widths,
        "expert_widths": expert_widths,
        "layerwise_keep_counts": layer_keep_counts,
        "layerwise_keep_plan": layer_keep_ratios,
        "layer_coverage_targets": layer_coverage_targets,
        "width_counts": width_counts,
        "active_width_counts": active_width_counts,
        "layer_sensitivity_weights": layer_weights,
        "expert_sensitivity_weights": expert_weights,
        "tier_count_balance": bool(balance_tier_counts),
        "sparse_tier_max_experts": int(sparse_tier_max_experts),
        "sparse_tier_merges": tier_merges,
        "post_merge_budget_delta": int(actual_keep_channels - target_keep_channels),
    }
    result.update(placement_groups)
    result.update(placement)
    result.update(mapping)

    if verbose:
        print(
            "[EP4 intplan] "
            f"target_prune={float(prune_ratio):.6f}, "
            f"actual_prune={actual_prune_ratio:.6f}, "
            f"rank_loads={placement['rank_weight_loads'].tolist()}, "
            f"relative_deviation={placement['relative_max_rank_weight_deviation']:.6f}, "
            f"tolerance_satisfied={placement['tolerance_satisfied']}"
        )
    return result


@torch.no_grad()
def plan_ep4_from_masks(
    intermediate_masks: torch.Tensor,
    layer_sensitivity: torch.Tensor,
    expert_sensitivity: torch.Tensor,
    scores: torch.Tensor,
    prune_ratio: float,
    *,
    widths: Sequence[int] = DEFAULT_WIDTHS,
    ep_size: int = 4,
    placement_tolerance: float = 0.01,
    strict_placement_tolerance: bool = False,
    placement_method: str = "greedy",
    placement_local_search_passes: int = 100,
    layer_smooth_times: int = 2,
    layer_smooth_fn: str = "sqrt",
    fix_first_layer_placement: bool = True,
    balance_tier_counts: bool = False,
    sparse_tier_max_experts: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Quantize an existing pruning mask into an executable EP4 plan.

    The input mask is produced by the regular MAES pipeline, including its
    modality-aware budgeting. Its per-expert channel counts are treated as the
    continuous allocation target. Within each expert, channels selected by the
    input mask remain ahead of unselected channels, with ``scores`` breaking
    ties inside both groups.
    """
    if not isinstance(intermediate_masks, torch.Tensor) or intermediate_masks.ndim != 3:
        raise ValueError(
            "intermediate_masks must be a [layers, experts, channels] tensor"
        )
    masks = intermediate_masks.detach().to(device="cpu", dtype=torch.bool)
    (
        layer_sensitivity,
        expert_sensitivity,
        scores,
        normalized_widths,
        active_widths,
        unit,
    ) = _validate_inputs(
        layer_sensitivity=layer_sensitivity,
        expert_sensitivity=expert_sensitivity,
        scores=scores,
        prune_ratio=prune_ratio,
        widths=widths,
        ep_size=ep_size,
    )
    if masks.shape != scores.shape:
        raise ValueError(
            "intermediate_masks shape must equal scores shape: "
            f"expected {tuple(scores.shape)}, got {tuple(masks.shape)}"
        )

    num_layers, num_experts, intermediate_size = scores.shape
    raw_counts = masks.sum(dim=-1, dtype=torch.int64)

    # Stable two-pass ordering: score descending inside each group, followed
    # by selected channels before unselected channels.
    score_order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    selected_in_score_order = torch.gather(masks, dim=-1, index=score_order)
    selected_first = torch.argsort(
        (~selected_in_score_order).to(torch.int8), dim=-1, stable=True
    )
    sorted_indices = torch.gather(score_order, dim=-1, index=selected_first)
    sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices).clamp_min(0.0)

    layer_weights = _prepare_layer_weights(
        layer_sensitivity,
        smooth_times=layer_smooth_times,
        smooth_fn=layer_smooth_fn,
    )
    expert_weights = _loss_to_layerwise_weights(expert_sensitivity)
    total_channels = num_layers * num_experts * intermediate_size
    continuous_target_keep = (1.0 - float(prune_ratio)) * total_channels
    target_keep_units = int(round(continuous_target_keep / unit))
    target_keep_channels = target_keep_units * unit
    expert_widths, _ = _quantize_widths_to_budget(
        raw_counts=raw_counts,
        sorted_scores=sorted_scores,
        layer_weights=layer_weights,
        expert_weights=expert_weights,
        active_widths=active_widths,
        unit=unit,
        target_keep_channels=target_keep_channels,
        balance_tier_counts=balance_tier_counts,
    )
    (
        expert_widths,
        actual_keep_channels,
        tier_merges,
        placement_groups,
        placement,
        mapping,
    ) = _finalize_ep4_structure(
        expert_widths,
        raw_counts,
        sorted_indices,
        active_widths,
        ep_size=ep_size,
        sparse_tier_max_experts=sparse_tier_max_experts,
        placement_tolerance=placement_tolerance,
        fix_first_layer_placement=fix_first_layer_placement,
        placement_method=placement_method,
        placement_local_search_passes=placement_local_search_passes,
    )

    all_widths = tuple(sorted(normalized_widths, reverse=True))
    width_counts = torch.stack(
        [
            torch.stack(
                [(expert_widths[layer] == width).sum() for width in all_widths]
            )
            for layer in range(num_layers)
        ]
    ).to(torch.int64)
    active_columns = [all_widths.index(width) for width in active_widths]
    active_width_counts = width_counts[:, active_columns]
    if strict_placement_tolerance and not placement["tolerance_satisfied"]:
        raise RuntimeError(
            "best cross-layer placement exceeds tolerance: "
            f"relative_deviation={placement['relative_max_rank_weight_deviation']:.6f}, "
            f"tolerance={placement_tolerance:.6f}"
        )

    actual_prune_ratio = 1.0 - float(actual_keep_channels) / float(total_channels)
    result: Dict[str, Any] = {
        "plan_source": "maes_intermediate_masks",
        "prune_ratio": float(prune_ratio),
        "keep_ratio": 1.0 - float(prune_ratio),
        "actual_prune_ratio": actual_prune_ratio,
        "actual_keep_ratio": 1.0 - actual_prune_ratio,
        "target_keep_channels": int(target_keep_channels),
        "actual_keep_channels": int(actual_keep_channels),
        "width_unit": int(unit),
        "widths": all_widths,
        "active_widths": active_widths,
        "base_K_E_inter": raw_counts,
        "K_E_inter": expert_widths,
        "expert_widths": expert_widths,
        "width_counts": width_counts,
        "active_width_counts": active_width_counts,
        "layer_sensitivity_weights": layer_weights,
        "expert_sensitivity_weights": expert_weights,
        "tier_count_balance": bool(balance_tier_counts),
        "sparse_tier_max_experts": int(sparse_tier_max_experts),
        "sparse_tier_merges": tier_merges,
        "post_merge_budget_delta": int(actual_keep_channels - target_keep_channels),
    }
    result.update(placement_groups)
    result.update(placement)
    result.update(mapping)
    if verbose:
        print(
            "[EP4 mask plan] "
            f"target_prune={float(prune_ratio):.6f}, "
            f"actual_prune={actual_prune_ratio:.6f}, "
            f"rank_loads={placement['rank_weight_loads'].tolist()}, "
            f"relative_deviation={placement['relative_max_rank_weight_deviation']:.6f}, "
            f"tolerance_satisfied={placement['tolerance_satisfied']}"
        )
    return result


__all__ = [
    "DEFAULT_WIDTHS",
    "plan_ep4_from_masks",
    "plan_ep4_intplan",
    "solve_cross_layer_placement",
]
