import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.calibration.collect_hessian_beta_sweep import (
    select_quantile_experts,
    validate_probe,
    verify_configuration,
)


def _probe(manifest_path: Path, manifest_sha256: str) -> dict:
    diagonal = torch.arange(1, 11, dtype=torch.float32) * 2.0
    return {
        "schema_version": 1,
        "layer_idx": 0,
        "loss_fn": "rel_l2",
        "num_batches": 2,
        "num_score_tokens": 40,
        "hessian_per_token": torch.diag(diagonal),
        "gradient_per_token": torch.zeros(10),
        "active_batch_counts": torch.full((10,), 2, dtype=torch.int64),
        "metadata": {
            "model_name_or_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": manifest_sha256,
            "batch_size": 2,
            "score_tokens_per_sample": 10,
            "selected_num_samples": 4,
            "attn_implementation": "flash_attention_2",
        },
    }


def _variable_probe(manifest_path: Path, manifest_sha256: str) -> dict:
    payload = _probe(manifest_path, manifest_sha256)
    payload["num_score_tokens"] = 26
    payload["metadata"].update(
        {
            "score_tokens_per_sample": None,
            "score_token_budget": 26,
            "score_token_counts_variable": True,
        }
    )
    return payload


class HessianBetaSweepConfigurationTest(unittest.TestCase):
    def _manifest(self, root: Path) -> tuple[Path, str]:
        path = root / "mixed.json"
        payload = {
            "schema_version": 1,
            "score_tokens_per_sample": 10,
            "samples": [
                {
                    "dataset_name": "gqa",
                    "dataset_index": idx,
                    "sample_id": f"sample-{idx}",
                    "selection_rank": idx,
                }
                for idx in range(4)
            ],
        }
        raw = json.dumps(payload).encode("utf-8")
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def _variable_manifest(self, root: Path) -> tuple[Path, str]:
        path = root / "mixed-variable.json"
        quotas = [4, 6, 7, 9]
        payload = {
            "schema_version": 2,
            "score_tokens_per_sample": None,
            "score_token_budget": sum(quotas),
            "score_token_counts_variable": True,
            "samples": [
                {
                    "dataset_name": "gqa",
                    "dataset_index": idx,
                    "sample_id": f"sample-{idx}",
                    "selection_rank": idx,
                    "model_token_count": quota + 5,
                    "score_token_count": quota,
                }
                for idx, quota in enumerate(quotas)
            ],
        }
        raw = json.dumps(payload).encode("utf-8")
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_quantile_selection_uses_hessian_diagonal_over_two(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path, digest = self._manifest(Path(temporary_dir))
            payload = _probe(manifest_path, digest)
            validate_probe(payload)
            selected = select_quantile_experts(payload)

        self.assertEqual([item["label"] for item in selected], ["low", "medium", "high"])
        self.assertEqual([item["expert_idx"] for item in selected], [1, 4, 8])
        self.assertEqual(
            [item["hessian_score_per_token"] for item in selected],
            [2.0, 5.0, 9.0],
        )

    def test_manifest_and_probe_configuration_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path, digest = self._manifest(root)
            payload = _probe(manifest_path, digest)
            args = argparse.Namespace(
                model_name_or_path=None,
                selection_manifest=None,
                layer=None,
                loss_fn=None,
                batch_size=None,
                score_tokens_per_sample=None,
                score_token_budget=None,
                attn_implementation=None,
            )
            expected, resolved, manifest, actual_digest = verify_configuration(
                args, payload, root / "hessian_probe_L0.pt"
            )

        self.assertEqual(expected["layer_idx"], 0)
        self.assertEqual(resolved, manifest_path.resolve())
        self.assertEqual(len(manifest["samples"]), 4)
        self.assertEqual(actual_digest, digest)

    def test_variable_quota_manifest_and_probe_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path, digest = self._variable_manifest(root)
            payload = _variable_probe(manifest_path, digest)
            args = argparse.Namespace(
                model_name_or_path=None,
                selection_manifest=None,
                layer=None,
                loss_fn=None,
                batch_size=None,
                score_tokens_per_sample=None,
                score_token_budget=26,
                attn_implementation=None,
            )
            expected, resolved, manifest, actual_digest = verify_configuration(
                args, payload, root / "hessian_probe_L0.pt"
            )

        self.assertTrue(expected["score_token_counts_variable"])
        self.assertIsNone(expected["score_tokens_per_sample"])
        self.assertEqual(expected["score_token_budget"], 26)
        self.assertEqual(
            [sample["score_token_count"] for sample in manifest["samples"]],
            [4, 6, 7, 9],
        )
        self.assertEqual(resolved, manifest_path.resolve())
        self.assertEqual(actual_digest, digest)

    def test_inconsistent_loss_and_manifest_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path, digest = self._manifest(root)
            payload = _probe(manifest_path, digest)
            wrong_loss = argparse.Namespace(
                model_name_or_path=None,
                selection_manifest=None,
                layer=None,
                loss_fn="l2",
                batch_size=None,
                score_tokens_per_sample=None,
                score_token_budget=None,
                attn_implementation=None,
            )
            with self.assertRaisesRegex(ValueError, "does not match probe"):
                verify_configuration(wrong_loss, payload, root / "probe.pt")

            payload["metadata"]["selection_manifest_sha256"] = "0" * 64
            no_overrides = argparse.Namespace(
                model_name_or_path=None,
                selection_manifest=None,
                layer=None,
                loss_fn=None,
                batch_size=None,
                score_tokens_per_sample=None,
                score_token_budget=None,
                attn_implementation=None,
            )
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_configuration(no_overrides, payload, root / "probe.pt")


if __name__ == "__main__":
    unittest.main()
