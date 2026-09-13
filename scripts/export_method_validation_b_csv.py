"""Export the compact, publication-facing data used by method-validation B."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if payload.get("kind") != "method_validation_b":
        raise ValueError(f"Not a merged method-validation B payload: {args.input}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "method_validation_B_scores.csv"
    beta_path = output_dir / "method_validation_B_beta_curves.csv"
    metadata_path = output_dir / "method_validation_B_metadata.json"

    score_fields = (
        "layer",
        "expert",
        "hvp_hessian_half",
        "expert_output_energy",
        "single_expert_ablation",
        "identity_gradient",
        "active_batch_count",
    )
    num_points = 0
    max_base_loss = 0.0
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields, lineterminator="\n")
        writer.writeheader()
        for layer_idx, result in sorted(payload["layers"].items()):
            max_base_loss = max(max_base_loss, abs(float(result["base_loss_per_token"])))
            tensors = {
                "hvp_hessian_half": result["hessian_score_per_token"],
                "expert_output_energy": result["energy_per_token"],
                "single_expert_ablation": result["ablation_per_token"],
                "identity_gradient": result["gradient_per_token"],
                "active_batch_count": result["active_batch_counts"],
            }
            for expert_idx in range(int(tensors["hvp_hessian_half"].numel())):
                if int(tensors["active_batch_count"][expert_idx]) <= 0:
                    continue
                writer.writerow(
                    {
                        "layer": int(layer_idx),
                        "expert": expert_idx,
                        **{
                            name: int(values[expert_idx])
                            if name == "active_batch_count"
                            else f"{float(values[expert_idx]):.17g}"
                            for name, values in tensors.items()
                        },
                    }
                )
                num_points += 1

    sweep_layer = int(payload["metadata"]["sweep_layer"])
    sweep = payload["layers"][sweep_layer]["beta_sweep"]
    if not bool(sweep["route_consistency_verified"]):
        raise ValueError("Refusing to export a beta sweep without fixed-router verification.")
    beta_fields = (
        "sensitivity",
        "quantile",
        "layer",
        "expert",
        "beta",
        "measured_delta_mse",
        "first_order_prediction",
        "second_order_prediction",
        "identity_gradient",
        "hvp_hessian_half",
        "local_gradient_at_beta",
    )
    betas = sweep["beta_values"].double()
    layer_result = payload["layers"][sweep_layer]
    gradient = layer_result["gradient_per_token"].double()
    hessian = layer_result["hessian_score_per_token"].double()
    with beta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=beta_fields, lineterminator="\n")
        writer.writeheader()
        for curve in sweep["curves"]:
            expert_idx = int(curve["expert_idx"])
            g_e = float(gradient[expert_idx])
            h_e = float(hessian[expert_idx])
            local_gradients = curve["local_gradient_at_beta_per_token"].double()
            for beta, measured, local_gradient in zip(
                betas, curve["measured_delta_per_token"].double(), local_gradients
            ):
                delta = float(beta) - 1.0
                writer.writerow(
                    {
                        "sensitivity": curve["label"],
                        "quantile": f"{float(curve['quantile']):.17g}",
                        "layer": sweep_layer,
                        "expert": expert_idx,
                        "beta": f"{float(beta):.17g}",
                        "measured_delta_mse": f"{float(measured):.17g}",
                        "first_order_prediction": f"{g_e * delta:.17g}",
                        "second_order_prediction": f"{g_e * delta + h_e * delta**2:.17g}",
                        "identity_gradient": f"{g_e:.17g}",
                        "hvp_hessian_half": f"{h_e:.17g}",
                        "local_gradient_at_beta": f"{float(local_gradient):.17g}",
                    }
                )

    metadata = {
        "schema_version": 1,
        "figure": "method_validation_B",
        "model_name_or_path": payload["metadata"]["model_name_or_path"],
        "selection_manifest_sha256": payload["metadata"]["selection_manifest_sha256"],
        "sample_selection": (
            f"first {int(payload['metadata']['num_samples'])} samples of the frozen calibration manifest"
        ),
        "num_samples": int(payload["metadata"]["num_samples"]),
        "batch_size": int(payload["metadata"]["batch_size"]),
        "layers": [int(layer) for layer in sorted(payload["layers"])],
        "num_layer_expert_points": num_points,
        "normalization": payload["metadata"]["normalization"],
        "validation_dtype": payload["metadata"]["validation_dtype"],
        "sweep_layer": sweep_layer,
        "route_consistency_verified": True,
        "route_comparisons": int(sweep["route_comparisons"]),
        "max_identity_base_loss_per_token": max_base_loss,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Exported {num_points} expert points and {len(betas) * len(sweep['curves'])} beta points")


if __name__ == "__main__":
    main()
