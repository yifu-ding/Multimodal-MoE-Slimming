"""Convert a dense teacher cache `.pt` into shard-manifest format.

The output format matches `build_teacher_hidden_cache.py` so downstream
distillation can use shard lazy sampling without loading the whole cache
into memory at startup.
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from tqdm import trange

from src.calibration.representation_distill.build_teacher_hidden_cache import (
    _build_shard_dir,
    _write_cache_shard,
)
from src.calibration.representation_distill.common import ensure_dir, utc_now_iso


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shard a dense teacher cache `.pt` into manifest + shard `.pt` files."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=16,
        help="How many samples to store per shard file.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dir(os.path.dirname(args.output_path))

    payload = torch.load(args.input_path, map_location="cpu")
    if "shards" in payload and "teacher_cache" not in payload:
        raise ValueError(
            f"Input {args.input_path} already looks like a sharded manifest; nothing to convert."
        )
    if "teacher_cache" not in payload:
        raise KeyError(f"Input {args.input_path} does not contain `teacher_cache`.")

    teacher_cache = payload["teacher_cache"].float()
    modality_labels = payload.get("modality_labels", None)
    position_ids = payload.get("position_ids", None)
    next_block_cache = payload.get("next_block_cache", None)

    if modality_labels is None:
        raise KeyError("Dense teacher cache is missing `modality_labels`.")
    if position_ids is None:
        raise KeyError("Dense teacher cache is missing `position_ids`.")

    shard_dir = _build_shard_dir(args.output_path)
    shard_manifest = []
    total_samples = int(teacher_cache.shape[0])

    for shard_idx, start in enumerate(trange(0, total_samples, args.samples_per_shard, desc="Writing cache shards")):
        end = min(start + args.samples_per_shard, total_samples)
        shard_manifest.append(
            _write_cache_shard(
                shard_dir=shard_dir,
                shard_idx=shard_idx,
                teacher_cache=teacher_cache[start:end].clone(),
                modality_labels=modality_labels[start:end].clone(),
                position_ids=position_ids[start:end].clone(),
                next_block_cache=next_block_cache[start:end].clone() if next_block_cache is not None else None,
            )
        )

    metadata = dict(payload.get("metadata", {}))
    metadata["shard_dir"] = os.path.basename(shard_dir)
    metadata["total_samples"] = total_samples
    metadata["converted_from_dense_pt"] = os.path.abspath(args.input_path)
    metadata["converted_at"] = utc_now_iso()

    manifest_payload = {
        "shards": shard_manifest,
        "dataset_ids": payload.get("dataset_ids"),
        "sample_manifest": payload.get("sample_manifest", []),
        "metadata": metadata,
    }
    torch.save(manifest_payload, args.output_path)
    print(f"[representation_distill] Saved sharded manifest to {args.output_path}")


if __name__ == "__main__":
    main()
