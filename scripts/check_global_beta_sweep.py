"""Recompute exported synchronous gradients and statistics from batch measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from src.calibration.collect_global_beta_sweep import BETAS, REPEAT_BETAS, summarize


def check(directory):
    raw = torch.load(directory / "raw.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata == raw["metadata"]
    assert metadata["complete"] and metadata["fixed_router"]
    assert metadata["num_experts"] == 128 and metadata["num_samples"] == 32
    assert metadata["parameterization"] == "independent_alpha_vector_at_equal_values"
    assert metadata["sweep_mode"] == "all_experts_synchronous"
    assert [sample for batch in raw["batches"] for sample in batch["sample_ids"]] == metadata["sample_ids"]
    for batch in raw["batches"]:
        assert [p["beta_global"] for p in batch["measurements"]] == list(BETAS)
        assert [p["beta_global"] for p in batch["repeats"]] == list(REPEAT_BETAS)
        assert batch["route_comparisons"] == 21
        for p in batch["measurements"] + batch["repeats"]:
            assert p["gradient_sum"].shape == (128,)
            assert p["gradient_sum"].dtype == torch.float64
            assert torch.isfinite(p["gradient_sum"]).all()
    for source, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
    rows, losses, residuals, validation, linearity = summarize(
        raw["batches"], raw["reference_gradient_at_beta095"], loss_fn=metadata["loss_fn"])
    for name, expected in (("gradients", rows), ("losses", losses), ("linearity_residuals", residuals)):
        with (directory / f"{name}.csv").open() as handle:
            actual = list(csv.DictReader(handle))
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected):
            assert actual_row == {key: str(value) for key, value in expected_row.items()}
    assert len(rows) == 768
    assert sum(row["beta_global"] != 1 for row in rows) == 640
    assert validation["passed"]
    assert validation == json.loads((directory / "validation.json").read_text())
    assert linearity == json.loads((directory / "linearity.json").read_text())
    print(f"PASS {directory}: 128 experts x 6 beta points; raw tensors, CSVs, statistics, numerical checks, and provenance agree.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    check(parser.parse_args().data_dir)
