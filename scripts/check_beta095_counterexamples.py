"""Validate real beta095 collection, normalization, finite differences and curve endpoints."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_beta095_counterexamples import analyze


def check(directory):
    raw = torch.load(directory / "raw.pt", map_location="cpu", weights_only=True)
    meta = json.loads((directory / "metadata.json").read_text())
    summary = json.loads((directory / "statistics.json").read_text())
    with (directory / "scores.csv").open() as handle:
        rows = [{k: (v if k == "loss_fn" else float(v)) for k, v in row.items()} for row in csv.DictReader(handle)]
    with (directory / "beta_curves.csv").open() as handle:
        curves = list(csv.DictReader(handle))
    if not meta.get("complete") or not meta["route_consistency_verified"]:
        raise ValueError("Collection/sweep incomplete or routers not verified.")
    tokens = sum(batch["num_score_tokens"] for batch in raw["batches"])
    assert tokens == meta["num_score_tokens"]
    assert meta["gradient_background"] == .95
    assert meta["ground_truth_background"] == meta["hessian_background"] == meta["teacher_background"] == 1
    by_id = {int(row["expert"]): row for row in rows}
    for field, source in (("gradient_at_work", "gradient"), ("gradient_at_identity", "identity_gradient"),
                          ("true_removal_at_identity", "removal"), ("true_removal_delta_at_work", "removal_work")):
        measured = sum(b[source] for b in raw["batches"]) / tokens
        np.testing.assert_allclose([row[field] for row in rows], [float(measured[int(row["expert"])]) for row in rows], rtol=1e-12, atol=0)
    for row in rows:
        assert row["first_order_signed"] == -.95 * row["gradient_at_work"]
        assert row["first_order_abs"] == abs(.95 * row["gradient_at_work"])
    loss_error = sum(b["repeat_loss_error"] for b in raw["batches"]) / tokens
    grad_error = .95 * sum(b["repeat_gradient_error"] for b in raw["batches"]) / tokens
    recalculated, pairs = analyze(rows, repeat_loss_error=loss_error, repeat_gradient_error=grad_error,
                                  hessian_rtol=summary["hessian_match_rtol"])
    assert recalculated == summary
    with (directory / "counterexamples.csv").open() as handle:
        saved_pairs = list(csv.DictReader(handle))
    assert len(saved_pairs) == len(pairs)
    for actual, expected in zip(saved_pairs, pairs):
        for key, value in expected.items():
            assert (actual[key] == value if isinstance(value, str) else float(actual[key]) == value)
    finite_difference_errors = []
    for batch in raw["batches"]:
        for item in batch["finite_differences"]:
            error = abs(item["central_difference_sum"] - item["gradient_sum"]) / max(abs(item["gradient_sum"]), 1e-30)
            finite_difference_errors.append(error)
    assert max(finite_difference_errors) < .01, "Measured gradient fails central difference check."
    identity_max = max(abs(row["gradient_at_identity"]) for row in rows)
    work_max = max(abs(row["gradient_at_work"]) for row in rows)
    assert work_max > 0
    assert identity_max < work_max * 1e-5
    endpoint_errors, gradient_errors = [], []
    for curve in curves:
        row = by_id[int(curve["expert"])]
        beta, background = float(curve["beta"]), float(curve["background_beta"])
        delta = float(curve["measured_delta_loss"])
        if beta == 0:
            expected = row["true_removal_at_identity"] if background == 1 else row["true_removal_delta_at_work"]
            endpoint_errors.append(abs(delta - expected))
        if beta == background:
            endpoint_errors.append(abs(delta))
            expected = row["gradient_at_identity"] if background == 1 else row["gradient_at_work"]
            gradient_errors.append(abs(float(curve["local_gradient_at_beta"]) - expected))
    assert max(endpoint_errors) <= summary["truth_tolerance"]
    assert max(gradient_errors) <= max(work_max * 1e-5, grad_error)
    if meta["loss_fn"] == "l2":
        assert summary["metrics"]["hessian_half_at_identity"]["max_relative_error"] < 1e-4
    report = {"passed": True, "num_experts": len(rows), "num_score_tokens": tokens,
              "num_curve_points": len(curves), "num_counterexample_rows": len(pairs),
              "max_identity_gradient": identity_max, "max_work_gradient": work_max,
              "repeat_loss_error_per_token": loss_error, "repeat_score_error_per_token": grad_error,
              "max_gradient_finite_difference_relative_error": max(finite_difference_errors),
              "max_curve_endpoint_error": max(endpoint_errors), "max_curve_baseline_gradient_error": max(gradient_errors)}
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    check(parser.parse_args().data_dir)
