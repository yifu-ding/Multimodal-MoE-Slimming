"""Check numerical acceptance criteria for a validation-B shard or merged payload."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--median-relative-error", type=float, default=1e-4)
    parser.add_argument("--max-relative-error", type=float, default=1e-2)
    parser.add_argument("--max-base-loss", type=float, default=1e-8)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if payload.get("kind") not in {"method_validation_b_shard", "method_validation_b"}:
        raise ValueError(f"Unsupported validation payload: {args.input}")

    failures = []
    for layer_idx, result in sorted(payload["layers"].items()):
        base_loss = abs(float(result["base_loss_per_token"]))
        if base_loss > args.max_base_loss:
            failures.append(f"L{layer_idx} base_loss={base_loss:.3e}")
        reference = result["hessian_score_per_token"].double()
        active = result["active_batch_counts"] > 0
        for name in ("energy_per_token", "ablation_per_token"):
            values = result[name].double()
            valid = active & torch.isfinite(reference) & torch.isfinite(values) & (reference > 0)
            relative = (values[valid] - reference[valid]).abs() / reference[valid].abs()
            if relative.numel() == 0:
                failures.append(f"L{layer_idx} {name}: no valid points")
                continue
            median = float(relative.median())
            maximum = float(relative.max())
            print(
                f"L{int(layer_idx)} {name}: points={relative.numel()} "
                f"median_rel={median:.3e} max_rel={maximum:.3e} base={base_loss:.3e}"
            )
            if median >= args.median_relative_error:
                failures.append(f"L{layer_idx} {name} median_rel={median:.3e}")
            if maximum >= args.max_relative_error:
                failures.append(f"L{layer_idx} {name} max_rel={maximum:.3e}")
    if failures:
        raise RuntimeError("Validation-B acceptance failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
