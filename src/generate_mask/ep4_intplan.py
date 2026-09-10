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


DEFAULT_WIDTHS = (768, 640, 512, 384, 0)


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


def _quantize_widths_to_budget(
    raw_counts: torch.Tensor,
    sorted_scores: torch.Tensor,
    layer_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    active_widths: tuple[int, ...],
    unit: int,
    target_keep_channels: int,
) -> tuple[torch.Tensor, int]:
    """Quantize by mandatory-tier anchors followed by marginal-cost upgrades."""
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
    if num_widths <= 0 or num_widths > 7:
        raise ValueError(f"unsupported number of active widths: {num_widths}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if bool((counts <= 0).any()):
        raise ValueError("every active width tier must contain at least one expert per layer")

    permutations = tuple(itertools.permutations(range(num_widths)))
    num_permutations = len(permutations)
    num_binary = num_layers * num_permutations
    deviation_index = num_binary
    num_variables = num_binary + 1

    layer_rank_load = np.zeros(
        (num_layers, num_permutations, num_widths), dtype=np.float64
    )
    counts_np = counts.numpy()
    for layer in range(num_layers):
        for permutation_id, permutation in enumerate(permutations):
            for rank, width_index in enumerate(permutation):
                layer_rank_load[layer, permutation_id, rank] = (
                    widths[width_index] * counts_np[layer, width_index]
                )

    total_load = float(sum(widths[index] * int(counts[:, index].sum()) for index in range(num_widths)))
    mean_load = total_load / float(num_widths)

    constraint_rows = num_layers + 2 * num_widths
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

    for rank in range(num_widths):
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
        identity_id = permutations.index(tuple(range(num_widths)))
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
    rank_widths = torch.empty((num_layers, num_widths), dtype=torch.int64)
    tier_to_rank = torch.empty((num_layers, num_widths), dtype=torch.int64)
    for layer in range(num_layers):
        start = layer * num_permutations
        values = result.x[start : start + num_permutations]
        permutation_id = int(np.argmax(values))
        if values[permutation_id] < 0.5:
            raise RuntimeError(f"layer {layer} has no integral placement permutation")
        permutation_ids.append(permutation_id)
        permutation = permutations[permutation_id]
        for rank, width_index in enumerate(permutation):
            rank_widths[layer, rank] = widths[width_index]
            tier_to_rank[layer, width_index] = rank

    rank_loads = torch.zeros(num_widths, dtype=torch.int64)
    for layer in range(num_layers):
        for rank in range(num_widths):
            width = int(rank_widths[layer, rank].item())
            width_index = widths.index(width)
            rank_loads[rank] += width * counts[layer, width_index]

    mean = float(rank_loads.double().mean().item())
    max_deviation = float((rank_loads.double() - mean).abs().max().item())
    relative_deviation = max_deviation / mean if mean > 0.0 else 0.0
    return {
        "rank_widths": rank_widths,
        "tier_to_rank": tier_to_rank,
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


def _placement_objective(rank_loads: np.ndarray) -> tuple[float, float]:
    centered = rank_loads - rank_loads.mean()
    return float(np.abs(centered).max()), float(np.dot(centered, centered))


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

    permutations = tuple(itertools.permutations(range(num_widths)))
    identity_id = permutations.index(tuple(range(num_widths)))
    counts_np = counts.numpy()
    widths_np = np.asarray(widths, dtype=np.int64)
    tier_loads = counts_np * widths_np[None, :]
    layer_rank_load = np.stack(
        [tier_loads[:, permutation] for permutation in permutations], axis=1
    )

    permutation_ids = np.full(num_layers, -1, dtype=np.int64)
    rank_loads = np.zeros(num_widths, dtype=np.int64)
    first_unfixed_layer = 0
    if fix_first_layer and num_layers > 0:
        permutation_ids[0] = identity_id
        rank_loads += layer_rank_load[0, identity_id]
        first_unfixed_layer = 1

    # Place the layers with the largest within-layer load spread first.  At
    # each step, balance the partial cumulative load; objective ties use the
    # squared deviation and then permutation ID for deterministic output.
    layer_order = sorted(
        range(first_unfixed_layer, num_layers),
        key=lambda layer: (-int(np.ptp(tier_loads[layer])), layer),
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

    # Coordinate descent fixes choices that were locally good during the
    # partial-load pass but are suboptimal after all layers have been placed.
    local_search_passes = 0
    for _ in range(max_local_search_passes):
        improved = False
        for layer in range(first_unfixed_layer, num_layers):
            current_id = int(permutation_ids[layer])
            base_loads = rank_loads - layer_rank_load[layer, current_id]
            current_objective = _placement_objective(rank_loads)
            best_id = current_id
            best_objective = current_objective
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

    rank_widths = torch.empty((num_layers, num_widths), dtype=torch.int64)
    tier_to_rank = torch.empty((num_layers, num_widths), dtype=torch.int64)
    for layer, permutation_id in enumerate(permutation_ids.tolist()):
        permutation = permutations[permutation_id]
        for rank, width_index in enumerate(permutation):
            rank_widths[layer, rank] = widths[width_index]
            tier_to_rank[layer, width_index] = rank

    rank_loads_tensor = torch.from_numpy(rank_loads.copy())
    mean = float(rank_loads_tensor.double().mean().item())
    max_deviation = float(
        (rank_loads_tensor.double() - mean).abs().max().item()
    )
    relative_deviation = max_deviation / mean if mean > 0.0 else 0.0
    return {
        "rank_widths": rank_widths,
        "tier_to_rank": tier_to_rank,
        "rank_weight_loads": rank_loads_tensor,
        "mean_rank_weight_load": mean,
        "max_rank_weight_deviation": max_deviation,
        "relative_max_rank_weight_deviation": relative_deviation,
        "tolerance": float(tolerance),
        "tolerance_satisfied": relative_deviation <= float(tolerance) + 1e-12,
        "permutation_ids": torch.from_numpy(permutation_ids.copy()),
        "solver_status": 0,
        "solver_message": "greedy placement with coordinate-descent local search",
        "solver_objective": max_deviation,
        "placement_method": "greedy",
        "local_search_passes": local_search_passes,
    }


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
    active_widths: tuple[int, ...],
    tier_to_rank: torch.Tensor,
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
        width_to_rank = {
            width: int(tier_to_rank[layer, width_index].item())
            for width_index, width in enumerate(active_widths)
        }
        layer_local_to_global: list[list[int]] = [list() for _ in active_widths]
        for expert in range(num_experts):
            width = int(expert_widths[layer, expert].item())
            if width <= 0:
                continue
            chosen = sorted_indices[layer, expert, :width]
            masks[layer, expert].index_fill_(0, chosen, True)
            rank = width_to_rank[width]
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
            ``(768, 640, 512, 384, 0)``.  Passing
            ``(768, 512, 384, 256, 0)`` reproduces the original tier proposal.
        placement_tolerance: Allowed relative deviation from mean cumulative
            rank weight load.
        strict_placement_tolerance: Raise if the best placement exceeds the
            tolerance.  The default returns the best plan plus a false status.
        placement_method: ``"greedy"`` for fast production planning or
            ``"milp"`` for an exact small-instance/offline reference.
        placement_local_search_passes: Maximum coordinate-descent passes used
            by the greedy placement method.

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
    expert_widths, actual_keep_channels = _quantize_widths_to_budget(
        raw_counts=raw_counts,
        sorted_scores=sorted_scores,
        layer_weights=layer_weights,
        expert_weights=expert_weights,
        active_widths=active_widths,
        unit=unit,
        target_keep_channels=target_keep_channels,
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

    placement = solve_cross_layer_placement(
        width_counts=active_width_counts,
        active_widths=active_widths,
        tolerance=placement_tolerance,
        fix_first_layer=fix_first_layer_placement,
        method=placement_method,
        max_local_search_passes=placement_local_search_passes,
    )
    if strict_placement_tolerance and not placement["tolerance_satisfied"]:
        raise RuntimeError(
            "best cross-layer placement exceeds tolerance: "
            f"relative_deviation={placement['relative_max_rank_weight_deviation']:.6f}, "
            f"tolerance={placement_tolerance:.6f}"
        )

    mapping = _build_masks_and_mappings(
        expert_widths=expert_widths,
        sorted_indices=sorted_indices,
        active_widths=active_widths,
        tier_to_rank=placement["tier_to_rank"],
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
    }
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


__all__ = [
    "DEFAULT_WIDTHS",
    "plan_ep4_intplan",
    "solve_cross_layer_placement",
]
