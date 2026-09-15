import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.calibration.block_forward import _build_fixed_score_mask, _teacher_forward_inputs
from src.calibration.collect_scores_main import _load_selection_manifest
from src.calibration.collector import loop_2_helpers
from src.calibration.collector.loop_1_helpers import compute_activation_I_masked
from src.calibration.collector.loop_1_score_collector import loop_1_score_collector
from src.calibration.collector.utils import (
    safe_update_running_stat,
    update_fused_running_stats,
)
from src.calibration.helpers.hooks import register_teacher_block_hook
from src.calibration.prepare_mixed_calibration import (
    _allocate_candidate_counts,
    _audit_token_lengths,
    allocate_total_score_tokens,
    balanced_farthest_point_sample,
    ensure_token_budget_capacity,
    farthest_point_sample,
)
from src.calibration.score_accumulator import _modality_affinity_from_counts
from tasks.video_mmmu import process_media


class RunningStatTest(unittest.TestCase):
    def test_mean_is_order_independent_and_preserves_scale(self):
        expected = torch.tensor([4.0, 5.0])
        values = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([8.0, 9.0]),
        ]
        results = []
        for observations in (values, list(reversed(values))):
            expert = nn.Module()
            for value in observations:
                safe_update_running_stat(expert, value, key="score", aggregation="mean")
            results.append(expert.score)
        torch.testing.assert_close(results[0], expected)
        torch.testing.assert_close(results[1], expected)

    def test_fused_mean_counts_each_expert_independently(self):
        experts = nn.Module()
        update_fused_running_stats(
            experts,
            "score",
            {0: torch.tensor(1.0), 1: torch.tensor(9.0)},
            num_experts=2,
            device=torch.device("cpu"),
        )
        update_fused_running_stats(
            experts,
            "score",
            {0: torch.tensor(3.0)},
            num_experts=2,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(experts.score, torch.tensor([2.0, 9.0]))

    def test_fused_token_counts_are_summed(self):
        experts = nn.Module()
        for value in (3.0, 5.0):
            update_fused_running_stats(
                experts,
                "token_count_text",
                {0: torch.tensor(value)},
                num_experts=2,
                device=torch.device("cpu"),
                accumulate_sum=True,
            )

        torch.testing.assert_close(experts.token_count_text, torch.tensor([8.0, 0.0]))


class TeacherBlockHookTest(unittest.TestCase):
    def test_captured_target_is_insulated_from_inplace_postprocessing(self):
        block = nn.Identity()
        state = {}
        handle = register_teacher_block_hook(block, state)
        try:
            output = block(torch.tensor([[1.0, 2.0]]))
            output.add_(10.0)
        finally:
            handle.remove()

        torch.testing.assert_close(state["output"], torch.tensor([[1.0, 2.0]]))


class TeacherForwardInputsTest(unittest.TestCase):
    def test_limits_logits_when_model_supports_logits_to_keep(self):
        class Model(nn.Module):
            def forward(self, input_ids=None, logits_to_keep=0):
                return input_ids

        input_ids = torch.tensor([[1, 2]])
        result = _teacher_forward_inputs(Model(), {"input_ids": input_ids})
        self.assertEqual(result["logits_to_keep"], 1)
        torch.testing.assert_close(result["input_ids"], input_ids)

    def test_does_not_add_unsupported_logit_limit(self):
        class Model(nn.Module):
            def forward(self, input_ids=None):
                return input_ids

        result = _teacher_forward_inputs(Model(), {"input_ids": torch.tensor([[1]])})
        self.assertNotIn("logits_to_keep", result)
        self.assertNotIn("num_logits_to_keep", result)


class VideoDecodeFallbackTest(unittest.TestCase):
    def test_process_media_retries_before_corrupt_video_tail(self):
        class _VideoReader:
            def __init__(self):
                self.requested_indices = []

            def __len__(self):
                return 100

        # Use the real Decord exception type while keeping all decoding in memory.
        import decord

        reader = _VideoReader()

        def get_batch(indices):
            reader.requested_indices.append(indices)
            if len(reader.requested_indices) < 3:
                raise decord.DECORDError("broken tail")
            return mock.Mock(asnumpy=lambda: np.zeros((8, 2, 2, 3), dtype=np.uint8))

        reader.get_batch = get_batch
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle, mock.patch(
            "tasks.video_mmmu.decord.VideoReader", return_value=reader
        ), self.assertWarnsRegex(RuntimeWarning, "through 95%"):
            frames, count = process_media(handle.name, max_frames=8)

        self.assertEqual(count, 8)
        self.assertEqual(len(frames), 8)
        self.assertEqual([indices[-1] for indices in reader.requested_indices], [99, 97, 94])


class SecondOrderAutotuneTest(unittest.TestCase):
    def setUp(self):
        loop_2_helpers.reset_second_order_chunk_autotune()

    def tearDown(self):
        loop_2_helpers.reset_second_order_chunk_autotune()

    def test_auto_chunk_halves_on_oom_and_reuses_success(self):
        context = {"teacher_target": torch.zeros(8, 2, 4)}
        attempted_sizes = []

        def fake_compute(**kwargs):
            chunk_size = kwargs["chunk_size"]
            attempted_sizes.append(chunk_size)
            if chunk_size > 2:
                raise RuntimeError("CUDA out of memory")
            return torch.ones(8)

        block = mock.Mock()
        env = {
            "SECOND_ORDER_CHUNK_SIZE": "auto",
            "SECOND_ORDER_MAX_CHUNK_SIZE": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            loop_2_helpers, "get_block_eval_context", return_value=context
        ), mock.patch.object(
            loop_2_helpers, "_compute_expert_second_order_batched_once", side_effect=fake_compute
        ), mock.patch.object(
            loop_2_helpers.torch.cuda, "is_available", return_value=False
        ):
            result = loop_2_helpers.compute_expert_second_order_batched(
                block, mock.Mock(), [True] * 8, {}
            )
            second_result = loop_2_helpers.compute_expert_second_order_batched(
                block, mock.Mock(), [True] * 8, {}
            )

        self.assertEqual(attempted_sizes, [8, 4, 2, 2])
        torch.testing.assert_close(result, torch.ones(8))
        torch.testing.assert_close(second_result, torch.ones(8))

    def test_explicit_chunk_does_not_retry(self):
        context = {"teacher_target": torch.zeros(8, 2, 4)}
        with mock.patch.dict(
            os.environ, {"SECOND_ORDER_CHUNK_SIZE": "4"}, clear=False
        ), mock.patch.object(
            loop_2_helpers, "get_block_eval_context", return_value=context
        ), mock.patch.object(
            loop_2_helpers,
            "_compute_expert_second_order_batched_once",
            side_effect=RuntimeError("CUDA out of memory"),
        ) as compute:
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                loop_2_helpers.compute_expert_second_order_batched(
                    mock.Mock(), mock.Mock(), [True] * 8, {}
                )

        self.assertEqual(compute.call_count, 1)


class HessianProbeTest(unittest.TestCase):
    class _Expert(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))

        def forward(self, hidden_states):
            return hidden_states * self.weight

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList(
                [
                    HessianProbeTest._Expert([1.0, 2.0]),
                    HessianProbeTest._Expert([3.0, -1.0]),
                ]
            )

        def forward(self, hidden_states):
            return self.experts[0](hidden_states) + self.experts[1](hidden_states)

    def test_full_hessian_and_direct_removals_match_exact_quadratic(self):
        block = self._Block()
        hidden_states = torch.tensor([[[1.0, 2.0], [-2.0, 0.5]]])
        teacher_target = block(hidden_states).detach()
        kwargs = {
            "block_in_args": (hidden_states,),
            "block_in_kwargs": {},
            "teacher_target": teacher_target,
            "attn_mask": torch.ones(1, 2, dtype=torch.bool),
            "loss_fn": "l2",
            "autocast_dtype": None,
            "autocast_device_type": "cpu",
            "hessian_probe_enabled": True,
            "hessian_probe_pair": (0, 1),
            "hessian_probe_validate": True,
        }

        result = loop_2_helpers._compute_expert_second_order_batched_once(
            cnt_block=block,
            experts=block.experts,
            has_activation_mask=[True, True],
            _kwargs=kwargs,
            chunk_size=1,
        )
        probe = kwargs["_hessian_probe_batch"]

        contributions = [expert(hidden_states).detach() for expert in block.experts]
        expected_hessian = torch.empty(2, 2)
        for row in range(2):
            for column in range(2):
                expected_hessian[row, column] = 2.0 * (
                    contributions[row] * contributions[column]
                ).mean(dim=-1).sum()

        torch.testing.assert_close(probe["gradient"], torch.zeros(2))
        torch.testing.assert_close(probe["hessian"], expected_hessian)
        torch.testing.assert_close(result, 0.5 * expected_hessian.diag())
        self.assertAlmostEqual(
            probe["ablation_losses"]["remove_e"],
            0.5 * expected_hessian[0, 0].item(),
        )
        self.assertAlmostEqual(
            probe["ablation_losses"]["remove_f"],
            0.5 * expected_hessian[1, 1].item(),
        )
        self.assertAlmostEqual(
            probe["ablation_losses"]["remove_ef"],
            0.5
            * (
                expected_hessian[0, 0]
                + expected_hessian[1, 1]
                + 2.0 * expected_hessian[0, 1]
            ).item(),
        )


class MixedCalibrationSelectionTest(unittest.TestCase):
    def test_balanced_fps_respects_source_quotas(self):
        points = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [8.0, 0.0], [9.0, 0.0], [10.0, 0.0]]
        )
        sources = ["a", "a", "a", "b", "b", "b"]
        selected, distances = balanced_farthest_point_sample(
            points, sources, {"a": 2, "b": 2}
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(len(distances), 4)
        self.assertEqual(
            {source: sum(sources[index] == source for index in selected) for source in ("a", "b")},
            {"a": 2, "b": 2},
        )

    def test_token_capacity_repair_preserves_sources(self):
        selected, replacements = ensure_token_budget_capacity(
            [0, 2],
            candidate_indices=[10, 11, 12, 13],
            candidate_sources=["a", "a", "b", "b"],
            candidate_capacities=[10, 100, 10, 200],
            target_tokens=250,
        )

        self.assertEqual(selected, [1, 3])
        self.assertEqual(replacements, 2)

    def test_total_token_allocation_is_exact_and_has_no_per_sample_cap(self):
        allocation = allocate_total_score_tokens([100, 500, 6000], 5000)

        self.assertEqual(sum(allocation), 5000)
        self.assertEqual(allocation[0], 100)
        self.assertGreater(max(allocation), 2048)
        self.assertTrue(all(value <= cap for value, cap in zip(allocation, [100, 500, 6000])))

    def test_token_audit_skips_undecodable_videommmu_candidate(self):
        class _Dataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                if index == 0:
                    raise OSError("broken video stream")
                return {"sample_id": "gqa-ok"}

        references = [
            {"dataset_name": "video_mmmu", "sample_id": "bad-video"},
            {"dataset_name": "gqa", "sample_id": "gqa-ok"},
        ]
        prepared = {"attention_mask": torch.ones(1, 12, dtype=torch.long)}
        with mock.patch(
            "src.calibration.prepare_mixed_calibration.prepare_raw_batch_inputs",
            return_value=prepared,
        ):
            qualified = _audit_token_lengths(
                _Dataset(), references, bundle=mock.Mock(), min_sample_tokens=10
            )

        self.assertEqual(qualified, [1])
        self.assertFalse(references[0]["token_qualified"])
        self.assertEqual(references[0]["exclusion_reason"], "video_decode_error")
        self.assertEqual(references[0]["exclusion_error_type"], "OSError")
        self.assertTrue(references[1]["token_qualified"])

    def test_token_audit_does_not_hide_non_video_io_errors(self):
        class _Dataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise OSError("broken image")

        references = [{"dataset_name": "gqa", "sample_id": "bad-image"}]
        with self.assertRaisesRegex(OSError, "broken image"):
            _audit_token_lengths(
                _Dataset(), references, bundle=mock.Mock(), min_sample_tokens=10
            )

    def test_candidate_allocation_is_balanced_and_redistributes_shortfall(self):
        allocation = _allocate_candidate_counts(
            10,
            {"gqa": 10, "coco": 10, "m4_instruct": 1, "video_mmmu": 10},
            ["gqa", "coco", "m4_instruct", "video_mmmu"],
        )

        self.assertEqual(sum(allocation.values()), 10)
        self.assertEqual(allocation["m4_instruct"], 1)
        remaining = [allocation[name] for name in ("gqa", "coco", "video_mmmu")]
        self.assertLessEqual(max(remaining) - min(remaining), 1)

    def test_fps_is_deterministic_and_starts_near_global_centroid(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [10.0, 0.0]])
        selected, distances = farthest_point_sample(points, 3)

        self.assertEqual(selected, [2, 3, 0])
        self.assertEqual(len(distances), 3)

    def test_fixed_score_mask_preserves_modality_ratio_and_spans_sequence(self):
        bundle = mock.Mock()
        bundle.model.config.image_token_id = 99
        bundle.model.config.video_token_id = None
        bundle.model.config.media_placeholder_token_id = None
        inputs = {
            "input_ids": torch.tensor(
                [[99, 99, 99, 99, 99, 99, 1, 2, 3, 4, 5, 6]]
            ),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
        }

        score_mask = _build_fixed_score_mask(bundle, inputs, tokens_per_sample=6)

        self.assertEqual(int(score_mask.sum().item()), 6)
        self.assertEqual(score_mask[0, :6].nonzero(as_tuple=True)[0].tolist(), [0, 2, 5])
        self.assertEqual(score_mask[0, 6:].nonzero(as_tuple=True)[0].tolist(), [0, 2, 5])

    def test_fixed_score_mask_accepts_variable_row_quotas(self):
        bundle = mock.Mock()
        bundle.model.config.image_token_id = None
        bundle.model.config.video_token_id = None
        bundle.model.config.media_placeholder_token_id = None
        inputs = {
            "input_ids": torch.arange(20).view(2, 10),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

        score_mask = _build_fixed_score_mask(bundle, inputs, tokens_per_sample=[3, 7])

        self.assertEqual(score_mask.sum(dim=1).tolist(), [3, 7])

    def test_affinity_uses_modality_normalized_routing_rates(self):
        # Expert 0 receives the same fraction (75%) of each modality even though
        # the calibration set contains ten times as many visual routing events.
        affinity = _modality_affinity_from_counts(
            torch.tensor([75.0, 25.0]),
            torch.tensor([750.0, 250.0]),
        )

        torch.testing.assert_close(affinity, torch.zeros(2))

    def test_manifest_loader_sorts_frozen_selection_rank(self):
        payload = {
            "schema_version": 1,
            "score_tokens_per_sample": 2048,
            "samples": [
                {
                    "selection_rank": 1,
                    "dataset_name": "coco",
                    "dataset_index": 9,
                    "sample_id": "b",
                },
                {
                    "selection_rank": 0,
                    "dataset_name": "gqa",
                    "dataset_index": 3,
                    "sample_id": "a",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            loaded, digest = _load_selection_manifest(handle.name)

        self.assertEqual(
            [sample["sample_id"] for sample in loaded["samples"]], ["a", "b"]
        )
        self.assertEqual(len(digest), 64)

    def test_manifest_loader_validates_variable_total_token_budget(self):
        payload = {
            "schema_version": 2,
            "score_tokens_per_sample": None,
            "score_token_budget": 12,
            "score_token_counts_variable": True,
            "samples": [
                {
                    "selection_rank": 0,
                    "dataset_name": "gqa",
                    "dataset_index": 3,
                    "sample_id": "a",
                    "model_token_count": 10,
                    "score_token_count": 5,
                },
                {
                    "selection_rank": 1,
                    "dataset_name": "coco",
                    "dataset_index": 9,
                    "sample_id": "b",
                    "model_token_count": 12,
                    "score_token_count": 7,
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            loaded, _ = _load_selection_manifest(handle.name)

        self.assertEqual(sum(item["score_token_count"] for item in loaded["samples"]), 12)

    def test_manifest_loader_rejects_variable_budget_with_fixed_quota(self):
        payload = {
            "schema_version": 2,
            "score_tokens_per_sample": 6,
            "score_token_budget": 12,
            "score_token_counts_variable": True,
            "samples": [
                {
                    "selection_rank": idx,
                    "dataset_name": "gqa",
                    "dataset_index": idx,
                    "sample_id": str(idx),
                    "model_token_count": 10,
                    "score_token_count": 6,
                }
                for idx in range(2)
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "must set score_tokens_per_sample to null"):
                _load_selection_manifest(handle.name)


class DualMaskCollectorTest(unittest.TestCase):
    @staticmethod
    def _fused_experts(score_mask):
        experts = nn.Module()
        experts.gate_up_proj = nn.Parameter(torch.ones(1, 1, 4))
        down_input = torch.tensor(
            [[1.0, 2.0], [100.0, 200.0], [300.0, 400.0], [3.0, 4.0]]
        )
        channel_values = {
            "saved_down_input": down_input,
            "saved_down_grad": torch.ones_like(down_input),
            "saved_up_output": down_input + 1.0,
            "saved_up_out_grad": torch.ones_like(down_input),
            "saved_gate_output": down_input + 2.0,
            "saved_gate_grad": torch.ones_like(down_input),
        }
        hidden_values = {
            "saved_down_output": torch.ones(4, 1),
            "saved_down_out_grad": torch.ones(4, 1),
            "saved_up_input": torch.ones(4, 1),
            "saved_up_in_grad": torch.ones(4, 1),
            "saved_gate_input": torch.ones(4, 1),
            "saved_gate_in_grad": torch.ones(4, 1),
        }
        for name, value in {**channel_values, **hidden_values}.items():
            setattr(experts, name, [value])
        experts.saved_text_mask = [torch.tensor([True, True, False, False])]
        experts.saved_visual_mask = [torch.tensor([False, False, True, True])]
        experts.saved_score_mask = [torch.as_tensor(score_mask, dtype=torch.bool)]
        experts.saved_router_weights = [torch.ones(4)]
        return experts, down_input

    def test_scores_use_fixed_mask_but_affinity_counts_use_full_route(self):
        experts, down_input = self._fused_experts([True, False, False, True])
        fused_metrics = {}
        score_mask = torch.tensor([True, False, False, True])
        expected = compute_activation_I_masked(
            down_input, down_input + 1.0, down_input + 2.0, score_mask
        )

        records, _, _, _ = loop_1_score_collector(
            range(1),
            experts=experts,
            is_fused=True,
            down_proj_t=torch.ones(1, 1, 2),
            up_proj=torch.ones(1, 2, 1),
            gate_proj=torch.ones(1, 2, 1),
            down_grad_t=None,
            up_grad_w=None,
            gate_grad_w=None,
            fused_metric_stacks=fused_metrics,
            ema=0.9,
            _kwargs={"attn_mask": score_mask.view(1, -1)},
        )

        torch.testing.assert_close(fused_metrics["3proj_act"][0], expected)
        self.assertEqual(float(fused_metrics["token_count_text"][0]), 2.0)
        self.assertEqual(float(fused_metrics["token_count_visual"][0]), 2.0)
        self.assertTrue(records[0]["has_activation"])

    def test_expert_routed_only_outside_score_mask_is_not_hessian_active(self):
        experts, _ = self._fused_experts([False, False, False, False])
        fused_metrics = {}

        records, _, _, _ = loop_1_score_collector(
            range(1),
            experts=experts,
            is_fused=True,
            down_proj_t=torch.ones(1, 1, 2),
            up_proj=torch.ones(1, 2, 1),
            gate_proj=torch.ones(1, 2, 1),
            down_grad_t=None,
            up_grad_w=None,
            gate_grad_w=None,
            fused_metric_stacks=fused_metrics,
            ema=0.9,
            _kwargs={"attn_mask": torch.zeros(1, 4, dtype=torch.bool)},
        )

        self.assertFalse(records[0]["has_activation"])
        self.assertEqual(set(fused_metrics), {"token_count_text", "token_count_visual"})


if __name__ == "__main__":
    unittest.main()
