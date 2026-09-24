"""Sensitivity-aware EP4 width planning with cross-layer placement.

The planner deliberately keeps the expensive decision out of a global
``layer * expert * tier`` MILP:

1. Reuse the existing score-coverage binary searches to allocate an exact
   channel budget first across layers and then across experts in each layer.
2. Quantize those per-expert counts to four active widths plus zero while
   preserving the nearest globally reachable pruning budget.
3. Assign each layer's active groups to EP ranks with the sorting-based greedy
   initialization and pairwise-swap refinement from Algorithm 2.  An exact
   assignment MILP remains available as an offline reference.

All tensors returned by this module are CPU tensors.  Width zero means that the
global expert remains in router space but has rank/local IDs equal to -1.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
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


class PlacementMilpNoIncumbentError(RuntimeError):
    def __init__(self, result: Any):
        super().__init__(
            "cross-layer placement MILP produced no incumbent: "
            f"status={result.status}, message={result.message}"
        )
        dual_bound = getattr(result, "mip_dual_bound", None)
        mip_gap = getattr(result, "mip_gap", None)
        node_count = getattr(result, "mip_node_count", None)
        self.diagnostics = {
            "solver_status": int(result.status),
            "solver_success": bool(result.success),
            "solver_optimal": False,
            "solver_message": str(result.message),
            "solver_objective": None,
            "mip_dual_bound": None if dual_bound is None else float(dual_bound),
            "mip_gap": None if mip_gap is None else float(mip_gap),
            "mip_node_count": None if node_count is None else int(node_count),
        }


def _as_finite_float_tensor(value: torch.Tensor, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got shape {tuple(value.shape)}"
        )
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], int
]:
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

    active_widths = tuple(
        sorted((width for width in normalized_widths if width > 0), reverse=True)
    )
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
                raise RuntimeError(
                    "cannot increase layer counts to the requested total"
                )
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
                raise RuntimeError(
                    "cannot decrease layer counts to the requested total"
                )
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
    layer_keep_ratios = layer_keep_counts.float() / float(
        num_experts * intermediate_size
    )

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
        raise RuntimeError(
            "balanced tier scheduler did not preserve global tier counts"
        )
    per_layer_spread = (
        layer_tier_counts.max(dim=1).values - layer_tier_counts.min(dim=1).values
    )
    if int(per_layer_spread.max().item()) > 1:
        raise RuntimeError(
            "balanced tier scheduler produced a per-layer tier-count spread above 1"
        )

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
    width_to_upgrade_index = {
        width: index for index, width in enumerate(upgrade_widths)
    }

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
        added_score = _prefix_value(prefix, layer, expert, new_width) - _prefix_value(
            prefix, layer, expert, old_width
        )
        sensitivity_scale = max(
            float(layer_weights[layer].item())
            * float(expert_weights[layer, expert].item()),
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
                raise RuntimeError(
                    f"layer {layer} has no expert in mandatory width tier {width}"
                )
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
            lower = max(
                (candidate for candidate in destinations if candidate < width),
                default=None,
            )
            higher = min(
                (candidate for candidate in destinations if candidate > width),
                default=None,
            )
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
                        destination = (
                            lower if lower_distance < higher_distance else higher
                        )
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
            splittable = [
                index
                for index, group in enumerate(groups)
                if len(group["experts"]) >= 2
            ]
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
            groups.insert(
                split_index, {"width": group["width"], "experts": expert_ids[:midpoint]}
            )
            groups.insert(
                split_index + 1,
                {"width": group["width"], "experts": expert_ids[midpoint:]},
            )

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


def _placement_objective(rank_loads: np.ndarray) -> tuple[float, float]:
    spread = float(rank_loads.max() - rank_loads.min())
    centered = rank_loads - rank_loads.mean()
    return spread, float(np.dot(centered, centered))


def _placement_metrics(rank_loads: torch.Tensor, tolerance: float) -> Dict[str, Any]:
    loads = rank_loads.double()
    mean = float(loads.mean().item())
    spread = float((loads.max() - loads.min()).item())
    relative_spread = spread / mean if mean > 0.0 else 0.0
    return {
        "mean_rank_weight_load": mean,
        "max_rank_weight_deviation": spread,
        "relative_max_rank_weight_deviation": relative_spread,
        "rank_weight_spread": spread,
        "relative_rank_weight_spread": relative_spread,
        "tolerance": float(tolerance),
        "tolerance_satisfied": relative_spread <= float(tolerance) + 1e-12,
    }


def _validate_placement_groups(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = group_counts.detach().cpu().to(torch.int64)
    widths = group_widths.detach().cpu().to(torch.int64)
    if counts.ndim != 2 or widths.shape != counts.shape:
        raise ValueError("group_counts and group_widths must have the same 2D shape")
    if counts.shape[1] <= 0:
        raise ValueError("placement must contain at least one group")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if bool((counts <= 0).any()) or bool((widths <= 0).any()):
        raise ValueError("every placement group must have positive count and width")
    return counts, widths


def _placement_spread_arithmetic_lower_bound(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
) -> Dict[str, int]:
    """Return the divisibility lower bound for the integral rank-load spread."""
    counts, widths = _validate_placement_groups(group_counts, group_widths, 0.0)
    quantum = 0
    for width in widths.flatten().tolist():
        quantum = math.gcd(quantum, int(width))
    if quantum <= 0:
        raise ValueError("placement width quantum must be positive")

    total_load = int((counts * widths).sum().item())
    if total_load % quantum:
        raise RuntimeError(
            f"total placement load {total_load} is not divisible by quantum {quantum}"
        )
    ep_size = int(counts.shape[1])
    total_quanta = total_load // quantum
    lower_bound = 0 if total_quanta % ep_size == 0 else quantum
    return {
        "arithmetic_quantum": quantum,
        "total_rank_weight_load": total_load,
        "total_load_quanta": total_quanta,
        "arithmetic_spread_lower_bound": lower_bound,
    }


def _decode_assignment(
    values: np.ndarray,
    counts: torch.Tensor,
    widths: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    num_layers, ep_size = counts.shape
    assignment = values[: num_layers * ep_size * ep_size].reshape(
        num_layers, ep_size, ep_size
    )
    rank_group_indices = torch.empty_like(counts)
    group_to_rank = torch.empty_like(counts)
    rank_widths = torch.empty_like(widths)
    for layer in range(num_layers):
        selected_groups = []
        for rank in range(ep_size):
            group = int(np.argmax(assignment[layer, :, rank]))
            if assignment[layer, group, rank] < 0.5:
                raise RuntimeError(
                    f"layer {layer}, rank {rank} has no integral group assignment"
                )
            rank_group_indices[layer, rank] = group
            group_to_rank[layer, group] = rank
            rank_widths[layer, rank] = widths[layer, group]
            selected_groups.append(group)
        if sorted(selected_groups) != list(range(ep_size)):
            raise RuntimeError(f"layer {layer} incumbent is not a permutation")
    rank_loads = torch.zeros(ep_size, dtype=torch.int64)
    for layer in range(num_layers):
        for rank in range(ep_size):
            group = int(rank_group_indices[layer, rank])
            rank_loads[rank] += counts[layer, group] * widths[layer, group]
    return {
        "rank_widths": rank_widths,
        "group_to_rank": group_to_rank,
        "rank_group_indices": rank_group_indices,
        "rank_weight_loads": rank_loads,
    }


def _solve_placement_groups_milp(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    time_limit: float | None = None,
    spread_upper_bound: float | None = None,
    feasibility_only: bool = False,
) -> Dict[str, Any]:
    """Solve placement with an assignment MILP or a bounded feasibility model."""
    counts, widths = _validate_placement_groups(group_counts, group_widths, tolerance)
    if feasibility_only and spread_upper_bound is None:
        raise ValueError("feasibility_only requires spread_upper_bound")
    if spread_upper_bound is not None and spread_upper_bound < 0.0:
        raise ValueError(
            f"spread_upper_bound must be non-negative, got {spread_upper_bound}"
        )
    arithmetic = _placement_spread_arithmetic_lower_bound(counts, widths)
    num_layers, ep_size = counts.shape
    group_loads = (counts * widths).numpy().astype(np.float64, copy=False)
    num_binary = num_layers * ep_size * ep_size
    max_index = num_binary
    min_index = num_binary + 1
    num_variables = num_binary + 2

    assignment_rows = 2 * num_layers * ep_size
    load_rows = 2 * ep_size
    valid_inequality_rows = 2 + int(spread_upper_bound is not None)
    constraint_rows = assignment_rows + load_rows + valid_inequality_rows
    matrix = lil_matrix((constraint_rows, num_variables), dtype=np.float64)
    lower = np.full(constraint_rows, -np.inf, dtype=np.float64)
    upper = np.full(constraint_rows, np.inf, dtype=np.float64)

    def variable_index(layer: int, group: int, rank: int) -> int:
        return (layer * ep_size + group) * ep_size + rank

    row = 0
    for layer in range(num_layers):
        for group in range(ep_size):
            for rank in range(ep_size):
                matrix[row, variable_index(layer, group, rank)] = 1.0
            lower[row] = upper[row] = 1.0
            row += 1
        for rank in range(ep_size):
            for group in range(ep_size):
                matrix[row, variable_index(layer, group, rank)] = 1.0
            lower[row] = upper[row] = 1.0
            row += 1

    for rank in range(ep_size):
        for layer in range(num_layers):
            for group in range(ep_size):
                matrix[row, variable_index(layer, group, rank)] = group_loads[
                    layer, group
                ]
        matrix[row, max_index] = -1.0
        upper[row] = 0.0
        row += 1

        for layer in range(num_layers):
            for group in range(ep_size):
                matrix[row, variable_index(layer, group, rank)] = -group_loads[
                    layer, group
                ]
        matrix[row, min_index] = 1.0
        upper[row] = 0.0
        row += 1

    total_load = float(group_loads.sum())
    mean_load = total_load / float(ep_size)
    matrix[row, max_index] = 1.0
    lower[row] = mean_load
    row += 1
    matrix[row, min_index] = 1.0
    upper[row] = mean_load
    row += 1
    if spread_upper_bound is not None:
        matrix[row, max_index] = 1.0
        matrix[row, min_index] = -1.0
        upper[row] = float(spread_upper_bound)

    bounds_lower = np.zeros(num_variables, dtype=np.float64)
    bounds_upper = np.ones(num_variables, dtype=np.float64)
    bounds_upper[max_index:] = total_load
    if fix_first_layer and num_layers > 0:
        for group in range(ep_size):
            for rank in range(ep_size):
                index = variable_index(0, group, rank)
                fixed_value = float(group == rank)
                bounds_lower[index] = fixed_value
                bounds_upper[index] = fixed_value

    objective = np.zeros(num_variables, dtype=np.float64)
    if not feasibility_only:
        objective[max_index] = 1.0
        objective[min_index] = -1.0
    integrality = np.ones(num_variables, dtype=np.int8)
    options: Dict[str, Any] = {"presolve": True, "mip_rel_gap": 0.0}
    if time_limit is not None:
        if time_limit <= 0.0:
            raise ValueError(f"time_limit must be positive, got {time_limit}")
        options["time_limit"] = float(time_limit)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options=options,
    )
    if result.x is None:
        error = PlacementMilpNoIncumbentError(result)
        error.diagnostics.update(arithmetic)
        error.diagnostics["milp_binary_variables"] = int(num_binary)
        error.diagnostics["milp_time_limit"] = (
            None if time_limit is None else float(time_limit)
        )
        error.diagnostics["milp_mode"] = (
            "feasibility" if feasibility_only else "optimization"
        )
        error.diagnostics["spread_upper_bound"] = (
            None if spread_upper_bound is None else float(spread_upper_bound)
        )
        raise error

    placement = _decode_assignment(result.x, counts, widths)
    placement.update(_placement_metrics(placement["rank_weight_loads"], tolerance))
    arithmetic_optimal = (
        placement["rank_weight_spread"]
        <= float(arithmetic["arithmetic_spread_lower_bound"]) + 1e-6
    )
    highs_model_optimal = int(result.status) == 0
    highs_optimal = highs_model_optimal and not feasibility_only
    dual_bound = getattr(result, "mip_dual_bound", None)
    mip_gap = getattr(result, "mip_gap", None)
    node_count = getattr(result, "mip_node_count", None)
    placement.update(
        {
            "solver_status": int(result.status),
            "solver_success": bool(result.success),
            "solver_optimal": highs_optimal or arithmetic_optimal,
            "highs_optimal": highs_optimal,
            "highs_model_optimal": highs_model_optimal,
            "feasibility_proven": feasibility_only and highs_model_optimal,
            "arithmetic_optimal": arithmetic_optimal,
            "optimality_proof": (
                "highs"
                if highs_optimal
                else "arithmetic_lower_bound"
                if arithmetic_optimal
                else None
            ),
            "solver_message": str(result.message),
            "solver_objective": placement["rank_weight_spread"],
            "solver_reported_objective": float(result.fun),
            "mip_dual_bound": np.nan if dual_bound is None else float(dual_bound),
            "mip_gap": np.nan if mip_gap is None else float(mip_gap),
            "mip_node_count": -1 if node_count is None else int(node_count),
            "milp_encoding": "assignment",
            "milp_mode": "feasibility" if feasibility_only else "optimization",
            "milp_binary_variables": int(num_binary),
            "milp_time_limit": None if time_limit is None else float(time_limit),
            "spread_upper_bound": (
                None if spread_upper_bound is None else float(spread_upper_bound)
            ),
            **arithmetic,
        }
    )
    return placement


def _solve_placement_groups_milp_permutation_reference(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
) -> Dict[str, Any]:
    """Exact permutation MILP retained to validate the assignment encoding."""
    counts, widths = _validate_placement_groups(group_counts, group_widths, tolerance)
    num_layers, ep_size = counts.shape
    if ep_size > 7:
        raise ValueError(f"unsupported reference EP size: {ep_size}")
    permutations = tuple(itertools.permutations(range(ep_size)))
    num_permutations = len(permutations)
    num_binary = num_layers * num_permutations
    max_index = num_binary
    min_index = num_binary + 1
    num_variables = num_binary + 2
    group_loads = (counts * widths).numpy()
    layer_rank_load = np.stack(
        [group_loads[:, permutation] for permutation in permutations], axis=1
    ).astype(np.float64, copy=False)

    matrix = lil_matrix((num_layers + 2 * ep_size, num_variables), dtype=np.float64)
    lower = np.full(matrix.shape[0], -np.inf, dtype=np.float64)
    upper = np.full(matrix.shape[0], np.inf, dtype=np.float64)
    row = 0
    for layer in range(num_layers):
        start = layer * num_permutations
        matrix[row, start : start + num_permutations] = 1.0
        lower[row] = upper[row] = 1.0
        row += 1
    for rank in range(ep_size):
        for layer in range(num_layers):
            start = layer * num_permutations
            matrix[row, start : start + num_permutations] = layer_rank_load[
                layer, :, rank
            ]
        matrix[row, max_index] = -1.0
        upper[row] = 0.0
        row += 1
        for layer in range(num_layers):
            start = layer * num_permutations
            matrix[row, start : start + num_permutations] = -layer_rank_load[
                layer, :, rank
            ]
        matrix[row, min_index] = 1.0
        upper[row] = 0.0
        row += 1

    total_load = float(group_loads.sum())
    bounds_lower = np.zeros(num_variables, dtype=np.float64)
    bounds_upper = np.ones(num_variables, dtype=np.float64)
    bounds_upper[max_index:] = total_load
    if fix_first_layer and num_layers > 0:
        identity_id = permutations.index(tuple(range(ep_size)))
        bounds_lower[:num_permutations] = 0.0
        bounds_upper[:num_permutations] = 0.0
        bounds_lower[identity_id] = bounds_upper[identity_id] = 1.0
    objective = np.zeros(num_variables, dtype=np.float64)
    objective[max_index] = 1.0
    objective[min_index] = -1.0
    result = milp(
        c=objective,
        integrality=np.ones(num_variables, dtype=np.int8),
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"reference placement MILP failed: status={result.status}, message={result.message}"
        )

    rank_group_indices = torch.empty_like(counts)
    group_to_rank = torch.empty_like(counts)
    rank_widths = torch.empty_like(widths)
    permutation_ids = []
    for layer in range(num_layers):
        start = layer * num_permutations
        permutation_id = int(np.argmax(result.x[start : start + num_permutations]))
        permutation_ids.append(permutation_id)
        for rank, group in enumerate(permutations[permutation_id]):
            rank_group_indices[layer, rank] = group
            group_to_rank[layer, group] = rank
            rank_widths[layer, rank] = widths[layer, group]
    rank_loads = torch.zeros(ep_size, dtype=torch.int64)
    for layer in range(num_layers):
        for rank in range(ep_size):
            group = int(rank_group_indices[layer, rank])
            rank_loads[rank] += counts[layer, group] * widths[layer, group]
    placement: Dict[str, Any] = {
        "rank_widths": rank_widths,
        "group_to_rank": group_to_rank,
        "rank_group_indices": rank_group_indices,
        "rank_weight_loads": rank_loads,
        "permutation_ids": torch.tensor(permutation_ids, dtype=torch.int64),
        "solver_status": int(result.status),
        "solver_success": bool(result.success),
        "solver_optimal": True,
        "solver_message": str(result.message),
        "solver_objective": float(result.fun),
        "mip_dual_bound": float(getattr(result, "mip_dual_bound", np.nan)),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", -1)),
        "milp_encoding": "permutation_reference",
        "milp_binary_variables": int(num_binary),
    }
    placement.update(_placement_metrics(rank_loads, tolerance))
    return placement


def _solve_cross_layer_placement_milp(
    width_counts: torch.Tensor,
    active_widths: Sequence[int],
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    time_limit: float | None = None,
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
        raise ValueError(
            "every active width tier must contain at least one expert per layer"
        )
    result = _solve_placement_groups_milp(
        counts,
        torch.tensor(widths, dtype=torch.int64).expand(num_layers, -1),
        tolerance=tolerance,
        fix_first_layer=fix_first_layer,
        time_limit=time_limit,
    )
    result["tier_to_rank"] = result["group_to_rank"]
    return result


def _solve_placement_groups_greedy_permutation_reference(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    max_local_search_passes: int = 100,
) -> Dict[str, Any]:
    """Original full-permutation greedy retained for small reference cases."""
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
    if ep_size > 8:
        raise ValueError(f"unsupported reference EP size: {ep_size}")
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
                *_placement_objective(
                    rank_loads + layer_rank_load[layer, permutation_id]
                ),
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
    placement: Dict[str, Any] = {
        "rank_widths": rank_widths,
        "group_to_rank": group_to_rank,
        "rank_group_indices": rank_group_indices,
        "rank_weight_loads": rank_loads_tensor,
        "permutation_ids": torch.from_numpy(permutation_ids.copy()),
        "solver_status": 0,
        "solver_message": "full-permutation greedy reference",
        "solver_objective": _placement_objective(rank_loads)[0],
        "placement_method": "greedy_permutation_reference",
        "refinement_neighborhood": "full_bijection",
        "local_search_passes": local_search_passes,
    }
    placement.update(_placement_metrics(rank_loads_tensor, tolerance))
    return placement


def _permutation_id(permutation: Sequence[int]) -> int:
    remaining = list(range(len(permutation)))
    result = 0
    for index, value in enumerate(permutation):
        position = remaining.index(int(value))
        result += position * math.factorial(len(permutation) - index - 1)
        remaining.pop(position)
    return result


def _lpt_sort_initialize(
    group_loads: np.ndarray,
    ep_size: int,
    num_layers: int,
    *,
    fix_first_layer: bool,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, float]:
    """Algorithm 2's sort initialization, shared by every Refine operator.

    Heaviest-extreme layers are placed first; within a layer the heaviest
    group goes to the currently lightest rank (LPT).
    """
    rank_group_indices = np.full((num_layers, ep_size), -1, dtype=np.int64)
    rank_loads = np.zeros(ep_size, dtype=np.int64)
    first_unfixed_layer = 0

    started = time.perf_counter()
    if fix_first_layer and num_layers > 0:
        rank_group_indices[0] = np.arange(ep_size, dtype=np.int64)
        rank_loads += group_loads[0]
        first_unfixed_layer = 1
    layer_order = sorted(
        range(first_unfixed_layer, num_layers),
        key=lambda layer: (-int(np.ptp(group_loads[layer])), layer),
    )
    for layer in layer_order:
        rank_order = sorted(range(ep_size), key=lambda rank: (rank_loads[rank], rank))
        group_order = sorted(
            range(ep_size), key=lambda group: (-group_loads[layer, group], group)
        )
        for rank, group in zip(rank_order, group_order):
            rank_group_indices[layer, rank] = group
            rank_loads[rank] += group_loads[layer, group]
    elapsed = time.perf_counter() - started
    return (
        rank_group_indices,
        rank_loads,
        first_unfixed_layer,
        rank_loads.copy(),
        elapsed,
    )


def _width_quantum(widths: torch.Tensor) -> int:
    quantum = 0
    for width in widths.flatten().tolist():
        quantum = math.gcd(quantum, int(width))
    if quantum <= 0:
        raise ValueError("placement width quantum must be positive")
    return quantum


def _finalize_refine_result(
    rank_group_indices: np.ndarray,
    rank_loads: np.ndarray,
    initial_rank_loads: np.ndarray,
    counts: torch.Tensor,
    widths: torch.Tensor,
    tolerance: float,
    *,
    placement_method: str,
    refinement_neighborhood: str,
    initialization_seconds: float,
    refinement_seconds: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Shared wrap-up for the SA/Tabu/Beam Refine operators."""
    num_layers, ep_size = counts.shape
    rank_group_indices_tensor = torch.from_numpy(rank_group_indices.copy())
    group_to_rank = torch.empty_like(counts)
    rank_widths = torch.empty_like(widths)
    for layer in range(num_layers):
        for rank, group in enumerate(rank_group_indices[layer].tolist()):
            group_to_rank[layer, group] = rank
            rank_widths[layer, rank] = widths[layer, group]
    rank_loads_tensor = torch.from_numpy(rank_loads.copy())
    initial_rank_loads_tensor = torch.from_numpy(initial_rank_loads.copy())
    placement: Dict[str, Any] = {
        "rank_widths": rank_widths,
        "group_to_rank": group_to_rank,
        "rank_group_indices": rank_group_indices_tensor,
        "rank_weight_loads": rank_loads_tensor,
        "initial_rank_weight_loads": initial_rank_loads_tensor,
        "solver_status": 0,
        "solver_objective": _placement_objective(rank_loads)[0],
        "placement_method": placement_method,
        "refinement_neighborhood": refinement_neighborhood,
        "initialization_time_seconds": initialization_seconds,
        "refinement_time_seconds": refinement_seconds,
    }
    placement.update(extra)
    placement.update(_placement_metrics(rank_loads_tensor, tolerance))
    initial_metrics = _placement_metrics(initial_rank_loads_tensor, tolerance)
    placement["initial_rank_weight_spread"] = initial_metrics["rank_weight_spread"]
    placement["initial_relative_rank_weight_spread"] = initial_metrics[
        "relative_rank_weight_spread"
    ]
    return placement


def _solve_placement_groups_simulated_annealing(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    seed: int = 0,
    cooling_rate: float = 0.99,
    max_iterations: int | None = None,
    max_seconds: float | None = 30.0,
) -> Dict[str, Any]:
    """Pairwise-swap neighbourhood with simulated-annealing acceptance.

    Anneals on the primary objective ΔΦ (scaled by the width quantum); when a
    candidate ties on ΔΦ, the same schedule anneals the tie-break S so the
    algorithm can still escape the ΔΦ-plateaus that trap greedy local search.
    """
    counts, widths = _validate_placement_groups(group_counts, group_widths, tolerance)
    if cooling_rate <= 0.0 or cooling_rate >= 1.0:
        raise ValueError(f"cooling_rate must be in (0, 1), got {cooling_rate}")
    num_layers, ep_size = counts.shape
    quantum = _width_quantum(widths)
    group_loads = (counts * widths).numpy()

    (
        rank_group_indices,
        rank_loads,
        first_unfixed_layer,
        initial_rank_loads,
        initialization_seconds,
    ) = _lpt_sort_initialize(
        group_loads, ep_size, num_layers, fix_first_layer=fix_first_layer
    )

    if max_iterations is None:
        max_iterations = max(1, 100 * num_layers * ep_size)
    initial_temperature = 1.44 * quantum
    min_temperature = max(quantum / 100.0, 1e-9)

    rng = np.random.default_rng(seed)
    current_indices = rank_group_indices.copy()
    current_loads = rank_loads.copy()
    current_objective = _placement_objective(current_loads)
    best_indices = current_indices.copy()
    best_loads = current_loads.copy()
    best_objective = current_objective
    accepted_worse = 0
    temperature = initial_temperature
    iterations_run = 0
    hit_time_cap = False

    refinement_started = time.perf_counter()
    if num_layers > first_unfixed_layer and ep_size > 1:
        for iterations_run in range(1, max_iterations + 1):
            if temperature < min_temperature:
                break
            if (
                max_seconds is not None
                and iterations_run % 2000 == 0
                and time.perf_counter() - refinement_started > max_seconds
            ):
                hit_time_cap = True
                break
            layer = int(rng.integers(first_unfixed_layer, num_layers))
            left, right = (int(value) for value in rng.choice(ep_size, size=2, replace=False))
            candidate = current_indices[layer].copy()
            candidate[left], candidate[right] = candidate[right], candidate[left]
            candidate_loads = (
                current_loads
                - group_loads[layer, current_indices[layer]]
                + group_loads[layer, candidate]
            )
            candidate_objective = _placement_objective(candidate_loads)
            delta_spread = candidate_objective[0] - current_objective[0]
            if delta_spread != 0.0:
                loss = delta_spread / quantum
            else:
                loss = (candidate_objective[1] - current_objective[1]) / (quantum * quantum)
            if loss <= 0.0:
                accept = True
                is_worse = False
            else:
                probability = math.exp(-loss / max(temperature, 1e-9))
                accept = bool(rng.random() < probability)
                is_worse = True
            if accept:
                if is_worse:
                    accepted_worse += 1
                current_indices[layer] = candidate
                current_loads = candidate_loads
                current_objective = candidate_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_indices = current_indices.copy()
                    best_loads = current_loads.copy()
            temperature *= cooling_rate
    refinement_seconds = time.perf_counter() - refinement_started

    return _finalize_refine_result(
        best_indices,
        best_loads,
        initial_rank_loads,
        counts,
        widths,
        tolerance,
        placement_method="simulated_annealing",
        refinement_neighborhood="pairwise_swap",
        initialization_seconds=initialization_seconds,
        refinement_seconds=refinement_seconds,
        extra={
            "solver_message": "sort initialization with simulated-annealing refinement",
            "random_seed": seed,
            "iterations": iterations_run,
            "max_iterations": max_iterations,
            "accepted_worse_count": accepted_worse,
            "initial_temperature": initial_temperature,
            "min_temperature": min_temperature,
            "cooling_rate": cooling_rate,
            "hit_time_cap": hit_time_cap,
        },
    )


def _solve_placement_groups_tabu(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    tabu_tenure: int | None = None,
    max_rounds: int | None = None,
    stall_rounds: int | None = None,
    seed: int = 0,
    max_seconds: float | None = 30.0,
) -> Dict[str, Any]:
    """Pairwise-swap neighbourhood, best non-tabu candidate per round.

    Each round scans every (layer, rank-pair) swap, takes the best one that
    is either not tabu or improves the best-so-far (aspiration), and always
    executes it -- even when it worsens the current solution -- which is how
    tabu search escapes the ΔΦ-plateaus untouched by greedy local search.
    Only one layer's assignment changes per round, so the round budget scales
    with the number of layers: a single sweep is not enough to touch every
    layer once. The tabu tenure is jittered (Glover's standard fix) because a
    fixed tenure otherwise lets the search settle into a period-``tenure``
    cycle of moves and their exact reverses.
    """
    counts, widths = _validate_placement_groups(group_counts, group_widths, tolerance)
    if max_rounds is not None and max_rounds < 0:
        raise ValueError(f"max_rounds must be non-negative, got {max_rounds}")
    num_layers, ep_size = counts.shape
    tenure = tabu_tenure if tabu_tenure is not None else max(1, ep_size)
    if max_rounds is None:
        max_rounds = max(100, 20 * num_layers)
    if stall_rounds is None:
        stall_rounds = max_rounds
    rng = np.random.default_rng(seed)
    group_loads = (counts * widths).numpy()

    (
        rank_group_indices,
        rank_loads,
        first_unfixed_layer,
        initial_rank_loads,
        initialization_seconds,
    ) = _lpt_sort_initialize(
        group_loads, ep_size, num_layers, fix_first_layer=fix_first_layer
    )

    current_indices = rank_group_indices.copy()
    current_loads = rank_loads.copy()
    best_indices = current_indices.copy()
    best_loads = current_loads.copy()
    best_objective = _placement_objective(best_loads)
    accepted_worse = 0
    tabu_until: dict[tuple[int, int, int], int] = {}
    rounds_without_improvement = 0
    rounds_run = 0
    hit_time_cap = False

    refinement_started = time.perf_counter()
    if ep_size > 1 and num_layers > first_unfixed_layer:
        for round_index in range(1, max_rounds + 1):
            if (
                max_seconds is not None
                and time.perf_counter() - refinement_started > max_seconds
            ):
                hit_time_cap = True
                break
            rounds_run = round_index
            current_objective = _placement_objective(current_loads)
            best_candidate = None
            best_candidate_objective = None
            best_candidate_key = None
            for layer in range(first_unfixed_layer, num_layers):
                base_loads = current_loads - group_loads[layer, current_indices[layer]]
                for left in range(ep_size):
                    for right in range(left + 1, ep_size):
                        candidate = current_indices[layer].copy()
                        candidate[left], candidate[right] = (
                            candidate[right],
                            candidate[left],
                        )
                        candidate_loads = base_loads + group_loads[layer, candidate]
                        candidate_objective = _placement_objective(candidate_loads)
                        key = (layer, left, right)
                        aspires = candidate_objective < best_objective
                        if tabu_until.get(key, 0) > round_index and not aspires:
                            continue
                        if (
                            best_candidate_objective is None
                            or candidate_objective < best_candidate_objective
                        ):
                            best_candidate_objective = candidate_objective
                            best_candidate = (layer, candidate, candidate_loads)
                            best_candidate_key = key
            if best_candidate is None:
                break
            layer, candidate, candidate_loads = best_candidate
            if best_candidate_objective > current_objective:
                accepted_worse += 1
            current_indices[layer] = candidate
            current_loads = candidate_loads
            jittered_tenure = int(rng.integers(tenure, 2 * tenure + 1))
            tabu_until[best_candidate_key] = round_index + jittered_tenure
            if best_candidate_objective < best_objective:
                best_objective = best_candidate_objective
                best_indices = current_indices.copy()
                best_loads = current_loads.copy()
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            if rounds_without_improvement >= stall_rounds:
                break
    refinement_seconds = time.perf_counter() - refinement_started

    return _finalize_refine_result(
        best_indices,
        best_loads,
        initial_rank_loads,
        counts,
        widths,
        tolerance,
        placement_method="tabu_search",
        refinement_neighborhood="pairwise_swap",
        initialization_seconds=initialization_seconds,
        refinement_seconds=refinement_seconds,
        extra={
            "solver_message": "sort initialization with tabu-search refinement",
            "iterations": rounds_run,
            "max_rounds": max_rounds,
            "tabu_tenure": tenure,
            "accepted_worse_count": accepted_worse,
            "random_seed": seed,
            "hit_time_cap": hit_time_cap,
        },
    )


def _solve_placement_groups_beam(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    beam_width: int = 8,
    max_rounds: int | None = None,
    stall_rounds: int | None = None,
    max_seconds: float | None = 30.0,
) -> Dict[str, Any]:
    """Pairwise-swap neighbourhood, keeping the ``beam_width`` best states.

    Every round expands every beam member by every pairwise swap in every
    layer, then keeps the top ``beam_width`` distinct states by (ΔΦ, S).
    Parallel paths let it cross plateaus that trap a single-point search.
    """
    counts, widths = _validate_placement_groups(group_counts, group_widths, tolerance)
    if beam_width <= 0:
        raise ValueError(f"beam_width must be positive, got {beam_width}")
    if max_rounds is not None and max_rounds < 0:
        raise ValueError(f"max_rounds must be non-negative, got {max_rounds}")
    num_layers, ep_size = counts.shape
    if max_rounds is None:
        max_rounds = max(100, 20 * num_layers)
    if stall_rounds is None:
        stall_rounds = max_rounds
    group_loads = (counts * widths).numpy()

    (
        rank_group_indices,
        rank_loads,
        first_unfixed_layer,
        initial_rank_loads,
        initialization_seconds,
    ) = _lpt_sort_initialize(
        group_loads, ep_size, num_layers, fix_first_layer=fix_first_layer
    )

    beam: list[tuple[np.ndarray, np.ndarray]] = [
        (rank_group_indices.copy(), rank_loads.copy())
    ]
    best_indices = rank_group_indices.copy()
    best_loads = rank_loads.copy()
    best_objective = _placement_objective(best_loads)
    rounds_without_improvement = 0
    rounds_run = 0
    hit_time_cap = False

    refinement_started = time.perf_counter()
    if ep_size > 1 and num_layers > first_unfixed_layer:
        for round_index in range(1, max_rounds + 1):
            if (
                max_seconds is not None
                and time.perf_counter() - refinement_started > max_seconds
            ):
                hit_time_cap = True
                break
            rounds_run = round_index
            seen_this_round: set[bytes] = set()
            candidates: list[tuple[tuple[float, float], np.ndarray, np.ndarray]] = []
            for indices, loads in beam:
                for layer in range(first_unfixed_layer, num_layers):
                    base_loads = loads - group_loads[layer, indices[layer]]
                    for left in range(ep_size):
                        for right in range(left + 1, ep_size):
                            candidate_layer = indices[layer].copy()
                            candidate_layer[left], candidate_layer[right] = (
                                candidate_layer[right],
                                candidate_layer[left],
                            )
                            candidate_loads = (
                                base_loads + group_loads[layer, candidate_layer]
                            )
                            new_indices = indices.copy()
                            new_indices[layer] = candidate_layer
                            key = new_indices.tobytes()
                            if key in seen_this_round:
                                continue
                            seen_this_round.add(key)
                            candidates.append(
                                (
                                    _placement_objective(candidate_loads),
                                    new_indices,
                                    candidate_loads,
                                )
                            )
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            beam = [
                (indices, loads) for _, indices, loads in candidates[:beam_width]
            ]
            round_best_objective, round_best_indices, round_best_loads = candidates[0]
            if round_best_objective < best_objective:
                best_objective = round_best_objective
                best_indices = round_best_indices.copy()
                best_loads = round_best_loads.copy()
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            if rounds_without_improvement >= stall_rounds:
                break
    refinement_seconds = time.perf_counter() - refinement_started

    return _finalize_refine_result(
        best_indices,
        best_loads,
        initial_rank_loads,
        counts,
        widths,
        tolerance,
        placement_method="beam_search",
        refinement_neighborhood="pairwise_swap",
        initialization_seconds=initialization_seconds,
        refinement_seconds=refinement_seconds,
        extra={
            "solver_message": "sort initialization with beam-search refinement",
            "iterations": rounds_run,
            "max_rounds": max_rounds,
            "beam_width": beam_width,
            "hit_time_cap": hit_time_cap,
        },
    )


def _solve_placement_groups_greedy(
    group_counts: torch.Tensor,
    group_widths: torch.Tensor,
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    max_local_search_passes: int = 100,
    refinement_neighborhood: str = "auto",
) -> Dict[str, Any]:
    """Balance rank loads with sort initialization and local refinement."""
    counts, widths = _validate_placement_groups(group_counts, group_widths, tolerance)
    if max_local_search_passes < 0:
        raise ValueError(
            "max_local_search_passes must be non-negative, got "
            f"{max_local_search_passes}"
        )
    if refinement_neighborhood not in {"auto", "pairwise_swap", "full_bijection"}:
        raise ValueError(
            "refinement_neighborhood must be 'auto', 'pairwise_swap', or "
            "'full_bijection', "
            f"got {refinement_neighborhood!r}"
        )

    num_layers, ep_size = counts.shape
    if refinement_neighborhood == "auto":
        refinement_neighborhood = "full_bijection" if ep_size == 4 else "pairwise_swap"
    if refinement_neighborhood == "full_bijection" and ep_size > 8:
        raise ValueError("full-bijection refinement is restricted to EP size <= 8")
    group_loads = (counts * widths).numpy()
    (
        rank_group_indices,
        rank_loads,
        first_unfixed_layer,
        initial_rank_loads,
        initialization_seconds,
    ) = _lpt_sort_initialize(
        group_loads, ep_size, num_layers, fix_first_layer=fix_first_layer
    )

    refinement_started = time.perf_counter()
    local_search_passes = 0
    for _ in range(max_local_search_passes):
        improved = False
        for layer in range(first_unfixed_layer, num_layers):
            current = rank_group_indices[layer].copy()
            base_loads = rank_loads - group_loads[layer, current]
            best = current
            best_loads = rank_loads
            best_objective = _placement_objective(rank_loads)
            if refinement_neighborhood == "pairwise_swap":
                candidates = []
                for left in range(ep_size):
                    for right in range(left + 1, ep_size):
                        candidate = current.copy()
                        candidate[left], candidate[right] = (
                            candidate[right],
                            candidate[left],
                        )
                        candidates.append(candidate)
            else:
                candidates = itertools.permutations(range(ep_size))
            for candidate_value in candidates:
                candidate = np.asarray(candidate_value, dtype=np.int64)
                candidate_loads = base_loads + group_loads[layer, candidate]
                candidate_objective = _placement_objective(candidate_loads)
                if candidate_objective < best_objective:
                    best = candidate.copy()
                    best_loads = candidate_loads
                    best_objective = candidate_objective
            if not np.array_equal(best, current):
                rank_group_indices[layer] = best
                rank_loads = best_loads
                improved = True
        local_search_passes += 1
        if not improved:
            break
    refinement_seconds = time.perf_counter() - refinement_started

    rank_group_indices_tensor = torch.from_numpy(rank_group_indices.copy())
    group_to_rank = torch.empty_like(counts)
    rank_widths = torch.empty_like(widths)
    permutation_ids = (
        torch.empty(num_layers, dtype=torch.int64) if ep_size <= 20 else None
    )
    for layer in range(num_layers):
        permutation = rank_group_indices[layer].tolist()
        if permutation_ids is not None:
            permutation_ids[layer] = _permutation_id(permutation)
        for rank, group in enumerate(permutation):
            group_to_rank[layer, group] = rank
            rank_widths[layer, rank] = widths[layer, group]

    rank_loads_tensor = torch.from_numpy(rank_loads.copy())
    initial_rank_loads_tensor = torch.from_numpy(initial_rank_loads.copy())
    placement: Dict[str, Any] = {
        "rank_widths": rank_widths,
        "group_to_rank": group_to_rank,
        "rank_group_indices": rank_group_indices_tensor,
        "rank_weight_loads": rank_loads_tensor,
        "initial_rank_weight_loads": initial_rank_loads_tensor,
        "permutation_ids": permutation_ids,
        "solver_status": 0,
        "solver_message": "sort initialization with local refinement",
        "solver_objective": _placement_objective(rank_loads)[0],
        "placement_method": "greedy",
        "refinement_neighborhood": refinement_neighborhood,
        "local_search_passes": local_search_passes,
        "initialization_time_seconds": initialization_seconds,
        "refinement_time_seconds": refinement_seconds,
    }
    placement.update(_placement_metrics(rank_loads_tensor, tolerance))
    initial_metrics = _placement_metrics(initial_rank_loads_tensor, tolerance)
    placement["initial_rank_weight_spread"] = initial_metrics["rank_weight_spread"]
    placement["initial_relative_rank_weight_spread"] = initial_metrics[
        "relative_rank_weight_spread"
    ]
    return placement


def _solve_cross_layer_placement_greedy(
    width_counts: torch.Tensor,
    active_widths: Sequence[int],
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    max_local_search_passes: int = 100,
    refinement_neighborhood: str = "auto",
) -> Dict[str, Any]:
    """Balance EP rank loads with Algorithm 2 sorting and local search."""
    if not isinstance(width_counts, torch.Tensor) or width_counts.ndim != 2:
        raise ValueError("width_counts must be a [L, num_active_widths] tensor")
    counts = width_counts.detach().cpu().to(torch.int64)
    widths = tuple(int(width) for width in active_widths)
    num_layers, num_widths = counts.shape
    if num_widths != len(widths):
        raise ValueError(
            f"width_counts has {num_widths} columns but active_widths has {len(widths)}"
        )
    if num_widths <= 0:
        raise ValueError(f"unsupported number of active widths: {num_widths}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if max_local_search_passes < 0:
        raise ValueError(
            "max_local_search_passes must be non-negative, got "
            f"{max_local_search_passes}"
        )
    if bool((counts <= 0).any()):
        raise ValueError(
            "every active width tier must contain at least one expert per layer"
        )

    result = _solve_placement_groups_greedy(
        counts,
        torch.tensor(widths, dtype=torch.int64).expand(num_layers, -1),
        tolerance=tolerance,
        fix_first_layer=fix_first_layer,
        max_local_search_passes=max_local_search_passes,
        refinement_neighborhood=refinement_neighborhood,
    )
    result["tier_to_rank"] = result["group_to_rank"]
    result["solver_message"] = "sort-based greedy placement with local refinement"
    return result


def solve_cross_layer_placement(
    width_counts: torch.Tensor,
    active_widths: Sequence[int],
    *,
    tolerance: float = 0.01,
    fix_first_layer: bool = True,
    method: str = "greedy",
    max_local_search_passes: int = 100,
    refinement_neighborhood: str = "auto",
    milp_time_limit: float | None = None,
) -> Dict[str, Any]:
    """Assign each layer's width groups to EP ranks.

    ``greedy`` is the production default. ``milp`` uses an exact assignment
    formulation for optimality comparisons.
    """
    normalized_method = method.lower()
    if normalized_method == "greedy":
        return _solve_cross_layer_placement_greedy(
            width_counts,
            active_widths,
            tolerance=tolerance,
            fix_first_layer=fix_first_layer,
            max_local_search_passes=max_local_search_passes,
            refinement_neighborhood=refinement_neighborhood,
        )
    if normalized_method == "milp":
        result = _solve_cross_layer_placement_milp(
            width_counts,
            active_widths,
            tolerance=tolerance,
            fix_first_layer=fix_first_layer,
            time_limit=milp_time_limit,
        )
        result["placement_method"] = "milp"
        return result
    if normalized_method == "greedy_permutation_reference":
        widths = torch.tensor(
            tuple(int(width) for width in active_widths), dtype=torch.int64
        )
        result = _solve_placement_groups_greedy_permutation_reference(
            width_counts,
            widths.expand(width_counts.shape[0], -1),
            tolerance=tolerance,
            fix_first_layer=fix_first_layer,
            max_local_search_passes=max_local_search_passes,
        )
        result["tier_to_rank"] = result["group_to_rank"]
        return result
    if normalized_method == "milp_permutation_reference":
        widths = torch.tensor(
            tuple(int(width) for width in active_widths), dtype=torch.int64
        )
        result = _solve_placement_groups_milp_permutation_reference(
            width_counts,
            widths.expand(width_counts.shape[0], -1),
            tolerance=tolerance,
            fix_first_layer=fix_first_layer,
        )
        result["tier_to_rank"] = result["group_to_rank"]
        result["placement_method"] = "milp_permutation_reference"
        return result
    raise ValueError(
        f"unsupported placement method {method!r}; expected 'greedy', 'milp', "
        "'greedy_permutation_reference', or 'milp_permutation_reference'"
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
    masks = torch.zeros((num_layers, num_experts, intermediate_size), dtype=torch.bool)
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
                raise RuntimeError(
                    f"active expert {expert} in layer {layer} has no placement group"
                )
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
) -> tuple[
    torch.Tensor,
    int,
    list[dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
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
                raise RuntimeError(
                    f"layer {layer} width {width} has no assigned EP rank"
                )
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
        placement_tolerance: Allowed relative max-min spread of cumulative
            rank weight load.
        strict_placement_tolerance: Raise if the best placement exceeds the
            tolerance.  The default returns the best plan plus a false status.
        placement_method: ``"greedy"`` for fast production planning or
            ``"milp"`` for an exact assignment-MILP reference.
        placement_local_search_passes: Maximum local-refinement passes used by
            the greedy placement method.
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
            torch.stack([(expert_widths[layer] == width).sum() for width in all_widths])
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
            torch.stack([(expert_widths[layer] == width).sum() for width in all_widths])
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
