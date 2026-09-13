import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.merge_method_validation_b import merge
from src.calibration.collect_method_validation_b import (
    fused_energy_and_ablation,
    select_quantile_experts,
)


class MethodValidationBTest(unittest.TestCase):
    def test_energy_and_ablation_use_routed_outputs(self):
        experts = SimpleNamespace(
            saved_down_output=[
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                None,
            ],
            saved_router_weights=[torch.tensor([0.5, 0.25]), None],
            saved_score_mask=[torch.tensor([True, True]), None],
            saved_token_indices=[torch.tensor([2, 2]), None],
        )
        energy, ablation, active = fused_energy_and_ablation(experts, num_experts=2)

        weighted = torch.tensor([[0.5, 1.0], [0.75, 1.0]])
        expected_energy = weighted.square().mean(dim=-1).sum()
        expected_ablation = weighted.sum(dim=0).square().mean()
        self.assertAlmostEqual(float(energy[0]), float(expected_energy))
        self.assertAlmostEqual(float(ablation[0]), float(expected_ablation))
        self.assertEqual(active.tolist(), [True, False])

    def test_quantile_selection_is_ordered(self):
        score = torch.arange(1, 11, dtype=torch.float64)
        selected = select_quantile_experts(score, torch.ones(10, dtype=torch.int64))
        self.assertEqual([item["label"] for item in selected], ["low", "medium", "high"])
        self.assertEqual([item["expert_idx"] for item in selected], [1, 4, 8])

    def test_merge_rejects_overlap_and_keeps_sweep(self):
        metadata = {
            "model_name_or_path": "model",
            "selection_manifest": "manifest.json",
            "selection_manifest_sha256": "abc",
            "batch_size": 2,
            "num_samples": 4,
            "loss_fn": "l2",
            "normalization": "test",
        }
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for rank, layer in enumerate((0, 1)):
                path = Path(root) / f"shard{rank}.pt"
                torch.save(
                    {
                        "schema_version": 1,
                        "kind": "method_validation_b_shard",
                        "metadata": {
                            **metadata,
                            "sweep_layer": 0 if layer == 0 else None,
                        },
                        "layers": {
                            layer: {"beta_sweep": {"curves": []} if layer == 0 else None}
                        },
                    },
                    path,
                )
                paths.append(path)
            payload = merge(paths)
        self.assertEqual(list(payload["layers"]), [0, 1])
        self.assertEqual(payload["metadata"]["sweep_layer"], 0)


if __name__ == "__main__":
    unittest.main()
