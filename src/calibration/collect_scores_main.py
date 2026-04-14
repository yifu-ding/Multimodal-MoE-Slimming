import argparse
import copy
import json
import os
import random
import sys
from typing import Dict

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.utils.data import DataLoader, Subset

from observations.common import (
    build_dataset,
    custom_collate_fn,
    discover_layer_structure,
    ensure_dir,
    load_model_bundle,
    normalize_dataset_name,
)
from src.calibration.helpers.helpers import teacher_block
from src.calibration.block_forward import block_forward

from src.calibration.score_accumulator import ScoreAccumulator


def save_score_artifacts(output_dir: str, accumulator: ScoreAccumulator, args) -> None:
    snapshot = copy.deepcopy(accumulator)
    # snapshot.finalize()
    scores_path = os.path.join(output_dir, "scores.pt")
    torch.save(snapshot.build_scores_payload(args), scores_path)
    print(f"[calibration] Saved scores: {scores_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect channel scores for Kimi-VL or Qwen3-VL on multimodal calibration data."
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="gqa")
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--ema", type=float, default=0.9)
    p.add_argument(
        "--loss_fn",
        type=str,
        default="rel_l2",
        choices=["l2", "rel_l2", "cosine"],
        help="Block reconstruction loss used during score collection (saved in scores.pt metadata).",
    )
    p.add_argument("--modality_aware", action="store_true")
    p.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for model loading. Defaults to `cuda:0` when CUDA is available, else `auto`.",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific MoE layer indices to calibrate, e.g. `--layers 20 21 22`.",
    )
    p.add_argument("--force", "-f", action="store_true")
    return p


def run_collection(args) -> None:
    args.dataset = normalize_dataset_name(args.dataset)
    supported_datasets = {"gqa", "coco", "video_mmmu"}
    if args.dataset not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}. "
            f"Supported datasets: {sorted(supported_datasets)}"
        )

    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, "scores.pt")
    if os.path.exists(out_path) and not args.force:
        print(
            f"[calibration] Found existing scores at {out_path}. "
            "Pass --force or -f to force overwrite."
        )
        return

    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"

    bundle = load_model_bundle(
        args.model_name_or_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )

    layer_to_num_experts, layer_to_num_channels = discover_layer_structure(bundle)
    accumulator = ScoreAccumulator(
        layer_to_num_experts,
        layer_to_num_channels,
    )

    print(
        f"[calibration] Discovered {len(layer_to_num_experts)} MoE layers, "
        f"{sum(layer_to_num_experts.values())} experts total."
    )

    dataset = build_dataset(args.dataset, bundle.family)
    pool = list(range(args.start_idx, len(dataset)))
    if args.subset_seed is not None and args.subset_seed >= 0:
        rng = random.Random(args.subset_seed)
        indices = rng.sample(pool, min(args.num_samples, len(pool)))
    else:
        indices = pool[: args.num_samples]

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    print("[calibration] Collecting block-reconstruction scores with attn_mlp collector...")
    target_layers = accumulator.layers
    if args.layers is not None:
        available = set(accumulator.layers)
        requested = []
        seen = set()
        for layer_idx in args.layers:
            if layer_idx in seen:
                continue
            seen.add(layer_idx)
            if layer_idx not in available:
                raise ValueError(
                    f"Requested layer {layer_idx} is not a MoE layer. "
                    f"Available MoE layers: {accumulator.layers}"
                )
            requested.append(layer_idx)
        if not requested:
            raise ValueError("`--layers` provided but no valid layer indices remained.")
        target_layers = requested
        print(
            f"[calibration] Restricting block calibration to {len(target_layers)} "
            f"specified layer(s): {target_layers}"
        )
    for layer_idx in target_layers:
        current_teacher_block = teacher_block(bundle, layer_idx)
        copied_block = copy.deepcopy(current_teacher_block)
        block_dtype = next(current_teacher_block.parameters()).dtype
        layer_loss = block_forward(
            bundle=bundle,
            cnt_block=copied_block,
            layer_idx=layer_idx,
            dataloader=loader,
            dataset_name=args.dataset,
            saliency_ema=args.ema,
            loss_fn=args.loss_fn,
            second_order_mode="exact",  # default second-order mode is exact
            dtype=block_dtype,
            verbose=True,
        )
        accumulator.layerwise_loss[layer_idx] = float(layer_loss)
        accumulator.absorb_layer_scores(layer_idx, copied_block)
        save_score_artifacts(args.output_dir, accumulator, args)
        print(f"[calibration] Layer {layer_idx}: layer loss={layer_loss:.6f}. "
              f"Have saved to {args.output_dir}/scores.pt")

    save_score_artifacts(args.output_dir, accumulator, args)


def main() -> None:
    # import sys as _sys
    # if len(_sys.argv) > 1 and _sys.argv[1] == "threshold":
    #     _sys.argv.pop(1)
    #     args = _threshold_arg_parser().parse_args()
    #     _run_threshold_calibration(args)
    # else:
    args = build_arg_parser().parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
