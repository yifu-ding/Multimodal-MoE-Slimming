import unittest

import torch

from scripts.merge_scores import LoadedScores, _extract_layers, _merge_payloads, _validate_strict_metadata
from scripts.discover_moe_layers_from_config import discover_moe_layers


def _payload(layer: int, *, manifest_sha: str = "abc") -> dict:
    return {
        "channel_scores": {"activation": {layer: {0: torch.tensor([float(layer + 1)])}}},
        "expert_scores": {"second_exact_attr": {layer: {0: float(layer + 1)}}},
        "ema_matrix": {layer: {0: 0.1}},
        "ema_matrix_prior_corrected": {layer: {0: 0.1}},
        "layerwise_loss": {layer: float(layer + 1)},
        "layerwise_second_order_sum": {layer: float((layer + 1) * 10)},
        "metadata": {
            "model_name_or_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "resolved_model_name_or_path": "/models/qwen",
            "loss_fn": "rel_l2",
            "dataset": "mixed",
            "selection_manifest_sha256": manifest_sha,
            "selected_num_samples": 512,
            "batch_size": 16,
            "score_tokens_per_sample": 2048,
            "score_token_budget": 1048576,
            "score_token_counts_variable": False,
            "score_token_sampling": "uniform",
            "score_aggregation": "mean",
            "fill_zero_for_unrouted": False,
            "layerwise_beta": 0.95,
            "layers": list(range(4)),
            "layer_to_num_experts": {idx: 128 for idx in range(4)},
            "layer_to_num_channels": {idx: 768 for idx in range(4)},
        },
    }


class MergeLayerShardsTest(unittest.TestCase):
    def test_qwen_moe_layers_are_discovered_from_config(self):
        class _TextConfig:
            num_hidden_layers = 8
            num_experts = 128
            decoder_sparse_step = 2
            mlp_only_layers = [3]

        class _Config:
            text_config = _TextConfig()

        self.assertEqual(discover_moe_layers(_Config()), [1, 5, 7])

    def test_observed_layers_override_full_model_metadata(self):
        self.assertEqual(_extract_layers(_payload(2)), [2])

    def test_second_order_layer_totals_are_preserved(self):
        loaded = [
            LoadedScores(f"shard{layer}.pt", _payload(layer), [layer], float(layer))
            for layer in (0, 1)
        ]
        _validate_strict_metadata(loaded)
        merged, warnings = _merge_payloads(loaded)

        self.assertFalse(warnings)
        self.assertEqual(merged["metadata"]["layers"], [0, 1])
        self.assertEqual(merged["layerwise_second_order_sum"], {0: 10.0, 1: 20.0})

    def test_zero_scalar_diagnostics_are_preserved(self):
        first = _payload(0)
        second = _payload(1)
        first["layerwise_loss"][0] = 0.0
        first["layerwise_second_order_sum"][0] = 0.0
        second["layerwise_loss"][1] = 0.0
        second["layerwise_second_order_sum"][1] = 0.0

        merged, warnings = _merge_payloads(
            [
                LoadedScores("shard0.pt", first, [0], 0.0),
                LoadedScores("shard1.pt", second, [1], 1.0),
            ]
        )

        self.assertFalse(warnings)
        self.assertEqual(merged["layerwise_loss"], {0: 0.0, 1: 0.0})
        self.assertEqual(merged["layerwise_second_order_sum"], {0: 0.0, 1: 0.0})

    def test_strict_merge_rejects_mixed_manifests(self):
        loaded = [
            LoadedScores("a.pt", _payload(0), [0], 0.0),
            LoadedScores("b.pt", _payload(1, manifest_sha="different"), [1], 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "selection_manifest_sha256"):
            _validate_strict_metadata(loaded)

    def test_strict_merge_rejects_duplicate_layers(self):
        loaded = [
            LoadedScores("a.pt", _payload(0), [0], 0.0),
            LoadedScores("b.pt", _payload(0), [0], 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "appears in both"):
            _validate_strict_metadata(loaded)


if __name__ == "__main__":
    unittest.main()
