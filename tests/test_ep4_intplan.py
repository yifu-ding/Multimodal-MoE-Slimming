import torch

from src.generate_mask.ep4_intplan import (
    DEFAULT_WIDTHS,
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
    )
    expected_units = round(0.70 * scores.numel() / 128)
    assert result["target_keep_channels"] == expected_units * 128
    assert abs(result["actual_prune_ratio"] - 0.30) < 128 / scores.numel()


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
        )
    except RuntimeError as error:
        assert "exceeds tolerance" in str(error)
    else:
        raise AssertionError("expected strict placement tolerance to reject the plan")
