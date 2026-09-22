import torch

from scripts.run_placement_e0_followup import solve_feasibility_ladder
from src.generate_mask.ep4_intplan import (
    DEFAULT_WIDTHS,
    _build_layer_placement_groups,
    _merge_sparse_width_tiers,
    _quantize_balanced_widths_to_budget,
    _solve_placement_groups_greedy,
    _solve_placement_groups_greedy_permutation_reference,
    _solve_placement_groups_milp,
    _solve_placement_groups_milp_permutation_reference,
    _placement_spread_arithmetic_lower_bound,
    plan_ep4_from_masks,
    plan_ep4_intplan,
    solve_cross_layer_placement,
)


def _synthetic_inputs():
    generator = torch.Generator().manual_seed(7)
    scores = torch.rand((4, 8, 768), generator=generator)
    layer_sensitivity = torch.tensor([0.4, 1.2, 0.8, 2.0])
    expert_sensitivity = torch.rand((4, 8), generator=generator) + 0.1
    return layer_sensitivity, expert_sensitivity, scores


def test_ep4_plan_preserves_discrete_budget_and_mapping_invariants():
    assert DEFAULT_WIDTHS == (0, 384, 512, 640, 768)
    layer_sensitivity, expert_sensitivity, scores = _synthetic_inputs()
    result = plan_ep4_intplan(
        layer_sensitivity,
        expert_sensitivity,
        scores,
        prune_ratio=0.30,
        placement_tolerance=0.20,
        sparse_tier_max_experts=0,
    )

    widths = result["expert_widths"]
    allowed = torch.tensor([0, 384, 512, 640, 768])
    assert torch.isin(widths, allowed).all()
    assert int(widths.sum().item()) == result["target_keep_channels"]
    assert result["actual_keep_channels"] == result["target_keep_channels"]
    assert result["intermediate_masks"].sum(dim=-1).equal(widths)
    assert int(result["raw_K_E_inter"].sum().item()) == round(0.70 * scores.numel())

    for layer in range(widths.shape[0]):
        for width in (768, 640, 512, 384):
            assert int((widths[layer] == width).sum().item()) >= 1
        assert sorted(result["rank_widths"][layer].tolist()) == [384, 512, 640, 768]

    removed = widths == 0
    assert (result["expert_to_rank"][removed] == -1).all()
    assert (result["expert_to_local_id"][removed] == -1).all()
    for layer in range(widths.shape[0]):
        for rank, global_ids in enumerate(result["local_to_global"][layer]):
            for local_id, expert_id in enumerate(global_ids):
                assert result["expert_to_rank"][layer, expert_id].item() == rank
                assert result["expert_to_local_id"][layer, expert_id].item() == local_id


def test_cross_layer_placement_rotates_tiers_and_balances_symmetric_case():
    counts = torch.tensor(
        [
            [8, 4, 2, 1],
            [8, 4, 2, 1],
            [8, 4, 2, 1],
            [8, 4, 2, 1],
        ],
        dtype=torch.int64,
    )
    result = solve_cross_layer_placement(
        counts,
        active_widths=(768, 640, 512, 384),
        tolerance=0.0,
    )

    assert result["rank_weight_loads"].unique().numel() == 1
    assert result["relative_max_rank_weight_deviation"] == 0.0
    assert result["tolerance_satisfied"]
    assert torch.unique(result["rank_widths"], dim=0).shape[0] > 1


def test_prune_ratio_is_not_interpreted_as_keep_ratio():
    layer_sensitivity, expert_sensitivity, scores = _synthetic_inputs()
    result = plan_ep4_intplan(
        layer_sensitivity,
        expert_sensitivity,
        scores,
        prune_ratio=0.30,
        sparse_tier_max_experts=0,
    )
    expected_units = round(0.70 * scores.numel() / 128)
    assert result["target_keep_channels"] == expected_units * 128
    assert abs(result["actual_prune_ratio"] - 0.30) < 128 / scores.numel()


def test_balanced_performance_tiers_hit_qwen_global_budgets():
    raw_counts = torch.arange(48 * 128, dtype=torch.int64).reshape(48, 128) % 769
    total_channels = 48 * 128 * 768
    expected_global_counts = {
        0.30: [1433, 1433, 1434, 1434],
        0.50: [1024, 1024, 1024, 1024],
    }

    for prune_ratio, expected_counts in expected_global_counts.items():
        target = round((1.0 - prune_ratio) * total_channels / 128) * 128
        widths, actual = _quantize_balanced_widths_to_budget(
            raw_counts=raw_counts,
            active_widths=(768, 640, 512, 384),
            unit=128,
            target_keep_channels=target,
        )
        active_counts = torch.stack(
            [(widths == width).sum() for width in (384, 512, 640, 768)]
        )
        per_layer_counts = torch.stack(
            [(widths == width).sum(dim=1) for width in (384, 512, 640, 768)],
            dim=1,
        )

        assert actual == target
        assert active_counts.tolist() == expected_counts
        assert int((active_counts.max() - active_counts.min()).item()) <= 1
        per_layer_spread = (
            per_layer_counts.max(dim=1).values - per_layer_counts.min(dim=1).values
        )
        assert int(per_layer_spread.max()) <= 1
        assert int((widths == 0).sum().item()) > 0


def test_infeasible_budget_is_rejected_when_all_active_tiers_are_mandatory():
    generator = torch.Generator().manual_seed(11)
    scores = torch.rand((1, 4, 768), generator=generator)
    try:
        plan_ep4_intplan(
            torch.ones(1),
            torch.ones((1, 4)),
            scores,
            prune_ratio=0.50,
        )
    except ValueError as error:
        assert "infeasible" in str(error)
    else:
        raise AssertionError("expected an infeasible mandatory-tier budget error")


def test_strict_placement_tolerance_rejects_unbalanced_single_layer():
    generator = torch.Generator().manual_seed(13)
    scores = torch.rand((1, 4, 768), generator=generator)
    try:
        plan_ep4_intplan(
            torch.ones(1),
            torch.ones((1, 4)),
            scores,
            prune_ratio=0.25,
            placement_tolerance=0.0,
            strict_placement_tolerance=True,
            sparse_tier_max_experts=0,
        )
    except RuntimeError as error:
        assert "exceeds tolerance" in str(error)
    else:
        raise AssertionError("expected strict placement tolerance to reject the plan")


def test_plan_from_masks_preserves_mask_priority_and_mapping_invariants():
    generator = torch.Generator().manual_seed(19)
    scores = torch.rand((4, 8, 768), generator=generator)
    base_masks = torch.zeros_like(scores, dtype=torch.bool)
    base_masks[..., :538] = True
    result = plan_ep4_from_masks(
        base_masks,
        torch.tensor([0.4, 1.2, 0.8, 2.0]),
        torch.rand((4, 8), generator=generator) + 0.1,
        scores,
        prune_ratio=0.30,
        placement_tolerance=0.20,
        sparse_tier_max_experts=0,
    )

    assert result["plan_source"] == "maes_intermediate_masks"
    assert int(result["expert_widths"].sum()) == result["target_keep_channels"]
    assert result["intermediate_masks"].sum(dim=-1).equal(result["expert_widths"])
    removed = result["expert_widths"] == 0
    assert (result["expert_to_rank"][removed] == -1).all()
    assert (result["expert_to_local_id"][removed] == -1).all()

    # Any expert quantized below the base keep count must contain only channels
    # that were already selected by the source mask.
    shrunk = result["expert_widths"] <= 512
    selected_outside_base = result["intermediate_masks"][..., 538:].any(dim=-1)
    assert not bool((shrunk & selected_outside_base).any())


def test_sparse_tier_merge_and_heaviest_group_split():
    expert_widths = torch.tensor(
        [[384] * 20 + [512] * 20 + [640] * 4 + [768] * 20], dtype=torch.int64
    )
    raw_counts = expert_widths.clone()
    raw_counts[0, 40:42] = 520
    raw_counts[0, 42:44] = 750

    merged, diagnostics = _merge_sparse_width_tiers(
        expert_widths,
        raw_counts,
        active_widths=(768, 640, 512, 384),
        max_experts=5,
    )

    assert int((merged == 640).sum()) == 0
    assert int((merged == 512).sum()) == 22
    assert int((merged == 768).sum()) == 22
    assert diagnostics == [
        {
            "layer": 0,
            "removed_width": 640,
            "removed_count": 4,
            "destinations": {512: 2, 768: 2},
        }
    ]

    groups = _build_layer_placement_groups(
        merged,
        active_widths=(768, 640, 512, 384),
        ep_size=4,
    )
    group_widths = groups["placement_group_widths"][0].tolist()
    group_counts = groups["placement_group_counts"][0].tolist()
    assert group_widths == [768, 768, 512, 384]
    assert group_counts == [11, 11, 22, 20]
    assert all(count > 0 for count in group_counts)


def test_default_plan_allows_duplicate_rank_widths_without_idle_ranks():
    layer_sensitivity, expert_sensitivity, scores = _synthetic_inputs()
    result = plan_ep4_intplan(
        layer_sensitivity,
        expert_sensitivity,
        scores,
        prune_ratio=0.30,
        placement_tolerance=1.0,
    )

    assert result["sparse_tier_max_experts"] == 5
    assert result["post_merge_budget_delta"] == (
        result["actual_keep_channels"] - result["target_keep_channels"]
    )
    for layer_groups in result["local_to_global"]:
        assert len(layer_groups) == 4
        assert all(layer_group for layer_group in layer_groups)


def test_sort_initialization_matches_full_permutation_reference():
    generator = torch.Generator().manual_seed(29)
    counts = torch.randint(1, 20, (8, 4), generator=generator)
    widths = torch.tensor([768, 640, 512, 384]).expand_as(counts)

    sorted_result = _solve_placement_groups_greedy(
        counts,
        widths,
        max_local_search_passes=0,
    )
    reference = _solve_placement_groups_greedy_permutation_reference(
        counts,
        widths,
        max_local_search_passes=0,
    )

    assert sorted_result["rank_weight_spread"] == reference["rank_weight_spread"]
    assert sorted_result["rank_weight_loads"].equal(reference["rank_weight_loads"])


def test_pairwise_and_full_bijection_refinement_are_selectable():
    counts = torch.tensor(
        [[3, 2, 12, 2], [14, 7, 11, 1], [14, 6, 4, 2]], dtype=torch.int64
    )
    widths = torch.tensor([768, 640, 512, 384]).expand_as(counts)

    pairwise = _solve_placement_groups_greedy(
        counts,
        widths,
        refinement_neighborhood="pairwise_swap",
    )
    full = _solve_placement_groups_greedy(
        counts,
        widths,
        refinement_neighborhood="full_bijection",
    )

    assert pairwise["refinement_neighborhood"] == "pairwise_swap"
    assert full["refinement_neighborhood"] == "full_bijection"
    assert full["rank_weight_spread"] <= pairwise["rank_weight_spread"]
    assert pairwise["initial_rank_weight_spread"] >= pairwise["rank_weight_spread"]


def test_assignment_milp_matches_permutation_milp_and_reports_dual_bound():
    counts = torch.tensor(
        [[8, 4, 2, 1], [7, 5, 3, 2], [6, 4, 3, 1], [8, 3, 2, 1]],
        dtype=torch.int64,
    )
    widths = torch.tensor([768, 640, 512, 384]).expand_as(counts)

    assignment = _solve_placement_groups_milp(counts, widths)
    permutation = _solve_placement_groups_milp_permutation_reference(counts, widths)

    assert assignment["solver_optimal"]
    assert assignment["rank_weight_spread"] == permutation["rank_weight_spread"]
    assert abs(assignment["mip_dual_bound"] - assignment["solver_objective"]) < 1e-6
    assert assignment["mip_gap"] == 0.0
    assert assignment["milp_binary_variables"] == counts.shape[0] * counts.shape[1] ** 2


def test_arithmetic_spread_lower_bound_uses_width_quantum_and_divisibility():
    widths = torch.tensor([768, 640, 512, 384]).expand(2, -1)
    divisible = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    indivisible = torch.tensor([[2, 1, 1, 1], [1, 1, 1, 1]])

    zero_floor = _placement_spread_arithmetic_lower_bound(divisible, widths)
    nonzero_floor = _placement_spread_arithmetic_lower_bound(indivisible, widths)

    assert zero_floor["arithmetic_quantum"] == 128
    assert zero_floor["arithmetic_spread_lower_bound"] == 0
    assert nonzero_floor["arithmetic_spread_lower_bound"] == 128


def test_milp_feasibility_at_arithmetic_floor_certifies_optimality():
    counts = torch.ones((2, 4), dtype=torch.int64)
    widths = torch.tensor([768, 640, 512, 384]).expand_as(counts)
    floor = _placement_spread_arithmetic_lower_bound(counts, widths)[
        "arithmetic_spread_lower_bound"
    ]

    result = _solve_placement_groups_milp(
        counts,
        widths,
        spread_upper_bound=floor,
        feasibility_only=True,
    )

    assert result["rank_weight_spread"] == floor
    assert result["arithmetic_optimal"]
    assert result["solver_optimal"]
    assert result["highs_model_optimal"]
    assert result["feasibility_proven"]
    assert not result["highs_optimal"]
    assert result["milp_mode"] == "feasibility"


def test_feasibility_ladder_retries_until_first_feasible_quantum():
    counts = torch.ones((1, 4), dtype=torch.int64)
    widths = torch.tensor([[768, 640, 512, 384]], dtype=torch.int64)

    result = solve_feasibility_ladder(counts, widths, total_time_limit=10.0)

    assert [attempt["target_spread"] for attempt in result["attempts"]] == [
        128.0,
        256.0,
        384.0,
    ]
    assert [attempt["solver_status"] for attempt in result["attempts"]] == [2, 2, 0]
    assert result["retry_count"] == 2
    assert result["spread"] == 384.0
    assert result["optimality_proven"]
    assert result["optimality_proof"] == "quantum_feasibility_ladder"


def test_greedy_and_milp_fix_the_same_first_layer():
    counts = torch.tensor(
        [[8, 4, 2, 1], [7, 5, 3, 2], [6, 4, 3, 1], [8, 3, 2, 1]],
        dtype=torch.int64,
    )
    widths = torch.tensor([768, 640, 512, 384]).expand_as(counts)

    greedy = _solve_placement_groups_greedy(counts, widths)
    milp = _solve_placement_groups_milp(counts, widths)

    expected = list(range(counts.shape[1]))
    assert greedy["rank_group_indices"][0].tolist() == expected
    assert milp["rank_group_indices"][0].tolist() == expected


def test_sort_greedy_supports_sixteen_ranks_without_permutation_enumeration():
    generator = torch.Generator().manual_seed(31)
    counts = torch.randint(1, 9, (6, 16), generator=generator)
    widths = torch.arange(16, 0, -1, dtype=torch.int64).mul(64).expand_as(counts)

    result = _solve_placement_groups_greedy(counts, widths)

    assert result["rank_weight_loads"].shape == (16,)
    assert sorted(result["rank_group_indices"][0].tolist()) == list(range(16))
