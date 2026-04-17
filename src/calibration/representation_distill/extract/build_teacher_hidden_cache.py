import argparse
import os
import sys
from typing import List

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.calibration.representation_distill.common import (
    build_sample_manifest,
    compress_hidden_states,
    ensure_dir,
    extract_block_output,
    get_hidden_size,
    get_num_decoder_layers,
    move_inputs_to_model_device,
    prepare_raw_batch_inputs,
    seed_everything,
    utc_now_iso,
)
from src.calibration.representation_distill.datasets.teacher_pool import build_teacher_pool


def _identity_collate(batch: List[dict]) -> List[dict]:
    return batch


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a multimodal teacher hidden cache for representation-level calibration distillation."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--teacher_layer", type=int, default=0)
    parser.add_argument("--compressed_length", type=int, default=8)
    parser.add_argument("--samples_per_dataset", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_seed", type=int, default=1234)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--video_max_long_side", type=int, default=480)
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    from observations.common import load_model_bundle, resolve_model_name_or_path

    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"

    bundle = load_model_bundle(
        args.model_name_or_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )
    num_layers = get_num_decoder_layers(bundle)
    if args.teacher_layer < 0 or args.teacher_layer >= num_layers:
        raise ValueError(
            f"Invalid teacher_layer={args.teacher_layer}; model has {num_layers} decoder layers."
        )

    teacher_pool_dir = os.path.join(args.output_dir, "teacher_pool")
    samples = build_teacher_pool(
        output_dir=teacher_pool_dir,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        num_video_frames=args.num_video_frames,
        video_max_long_side=args.video_max_long_side,
    )
    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_identity_collate,
    )

    save_dtype = _parse_dtype(args.save_dtype)
    cache_chunks = []
    dataset_ids = []
    sample_manifest = []
    teacher_dtype = None

    for batch in tqdm(loader, desc="Extracting teacher cache", leave=False):
        inputs = prepare_raw_batch_inputs(bundle, batch)
        inputs = move_inputs_to_model_device(bundle.model, inputs)
        hidden = extract_block_output(bundle, inputs, layer_idx=args.teacher_layer)
        teacher_dtype = teacher_dtype or str(hidden.dtype).replace("torch.", "")
        pooled = compress_hidden_states(
            hidden_states=hidden,
            attention_mask=inputs["attention_mask"],
            target_length=args.compressed_length,
        ).to(dtype=save_dtype)
        cache_chunks.append(pooled.cpu())
        dataset_ids.extend(int(sample["dataset_id"]) for sample in batch)
        sample_manifest.extend(build_sample_manifest(batch))

    teacher_cache = torch.cat(cache_chunks, dim=0)
    payload = {
        "teacher_cache": teacher_cache,
        "dataset_ids": torch.tensor(dataset_ids, dtype=torch.long),
        "sample_manifest": sample_manifest,
        "metadata": {
            "method": "multimodal_representation_level_calibration_distillation",
            "model_name_or_path": args.model_name_or_path,
            "resolved_model_name_or_path": resolve_model_name_or_path(args.model_name_or_path),
            "processor_name": bundle.processor.__class__.__name__,
            "model_family": bundle.family,
            "teacher_layer": args.teacher_layer,
            "teacher_layer_type": "full_block_output",
            "compressed_length": args.compressed_length,
            "hidden_size": get_hidden_size(bundle),
            "teacher_cache_dtype": str(save_dtype).replace("torch.", ""),
            "teacher_model_output_dtype": teacher_dtype,
            "samples_per_dataset": args.samples_per_dataset,
            "total_samples": int(teacher_cache.shape[0]),
            "dataset_id_to_name": {
                0: "gqa",
                1: "coco",
                2: "m4_instruct",
                3: "video_mmmu",
            },
            "num_video_frames": args.num_video_frames,
            "video_max_long_side": args.video_max_long_side,
            "shuffle_seed": args.shuffle_seed,
            "created_at": utc_now_iso(),
        },
    }
    out_path = os.path.join(args.output_dir, "teacher_hidden_cache.pt")
    torch.save(payload, out_path)
    print(f"[representation_distill] Saved teacher cache to {out_path}")


if __name__ == "__main__":
    main()
