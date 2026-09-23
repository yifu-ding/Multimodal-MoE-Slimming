import torch

from src.generate_mask.ep4_intplan import plan_ep4_intplan
from src.vllm_ep4_plan import SCHEMA_VERSION, validate_ep4_plan
from src.vllm_ep4_runtime import (
    _rank_expert_map,
    _single_width_layer_plan,
    _slice_expert_weight,
)


def _valid_plan():
    generator = torch.Generator().manual_seed(23)
    result = plan_ep4_intplan(
        torch.ones(4),
        torch.rand((4, 8), generator=generator) + 0.1,
        torch.rand((4, 8, 768), generator=generator),
        prune_ratio=0.30,
        placement_tolerance=0.20,
    )
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "model_layer_ids": [0, 1, 2, 3],
        }
    )
    return result


def test_plan_validation_and_per_rank_maps_skip_removed_experts():
    plan = validate_ep4_plan(_valid_plan())
    for layer in range(plan["expert_widths"].shape[0]):
        removed = plan["expert_widths"][layer] == 0
        for rank in range(4):
            expert_map = _rank_expert_map(
                plan["expert_to_rank"][layer],
                plan["expert_to_local_id"][layer],
                rank,
            )
            assert (expert_map[removed] == -1).all()
            assigned = plan["expert_to_rank"][layer] == rank
            assert (expert_map[~assigned] == -1).all()
            assert expert_map[assigned].tolist() == list(range(int(assigned.sum())))


def test_weight_slicing_uses_same_channels_for_gate_up_and_down():
    channels = torch.tensor([0, 3, 6, 7])
    gate = torch.arange(8 * 3).reshape(8, 3)
    down = torch.arange(3 * 8).reshape(3, 8)
    assert torch.equal(
        _slice_expert_weight(gate, "w1", channels, 8), gate[channels]
    )
    assert torch.equal(
        _slice_expert_weight(gate, "w3", channels, 8), gate[channels]
    )
    assert torch.equal(
        _slice_expert_weight(down, "w2", channels, 8), down[:, channels]
    )


def test_single_width_pins_one_tier_per_rank_across_all_layers():
    plan = validate_ep4_plan(_valid_plan())
    active_widths = sorted(int(value) for value in plan["active_widths"])
    num_layers = plan["expert_widths"].shape[0]
    for position in range(num_layers):
        layer_widths = plan["expert_widths"][position]
        seen_experts: set[int] = set()
        for rank in range(4):
            active = _single_width_layer_plan(
                plan, position, model_layer_id=position, ep_rank=rank
            )
            # The naive baseline must never rotate: rank `r` always owns the
            # r-th smallest active width, in every single layer.
            assert active.rank_width == active_widths[rank]
            for expert_id in active.local_to_global:
                assert int(layer_widths[expert_id]) == active_widths[rank]
                assert expert_id not in seen_experts
                seen_experts.add(expert_id)
        expected = {
            int(expert_id)
            for expert_id in range(int(layer_widths.shape[0]))
            if int(layer_widths[expert_id]) != 0
        }
        assert seen_experts == expected


def test_validator_rejects_removed_expert_with_live_mapping():
    plan = _valid_plan()
    removed = torch.nonzero(plan["expert_widths"] == 0, as_tuple=False)
    if removed.numel() == 0:
        # Force a structurally invalid mapping independent of the planner's
        # exact tier allocation for this small synthetic input.
        plan["expert_widths"][0, 0] = 0
        plan["intermediate_masks"][0, 0] = False
        removed = torch.tensor([[0, 0]])
    layer, expert = removed[0].tolist()
    plan["expert_to_rank"][layer, expert] = 0
    try:
        validate_ep4_plan(plan)
    except ValueError as error:
        assert "removed experts" in str(error)
    else:
        raise AssertionError("expected an invalid removed-expert mapping error")
