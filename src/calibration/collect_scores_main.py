import argparse
import copy
import hashlib
import json
import os
import random
import sys
from typing import Dict, List

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from observations.common import (
    build_dataset,
    custom_collate_fn,
    discover_layer_structure,
    ensure_dir,
    load_model_bundle,
    normalize_dataset_name,
    prepare_inputs,
)
from src.calibration.helpers.helpers import teacher_block
from src.calibration.block_forward import block_forward

from src.calibration.score_accumulator import ScoreAccumulator
from src.calibration.representation_distill.runtime.dump_original_data import (
    ManifestRawDataset,
)


def save_score_artifacts(output_dir: str, accumulator: ScoreAccumulator, args) -> None:
    snapshot = copy.deepcopy(accumulator)
    # snapshot.finalize()
    scores_path = os.path.join(output_dir, "scores.pt")
    torch.save(snapshot.build_scores_payload(args), scores_path)
    print(f"[calibration] Saved scores: {scores_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect channel scores for Kimi-VL, Qwen3-VL, or InternVL on multimodal calibration data."
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="gqa")
    p.add_argument(
        "--selection_manifest",
        type=str,
        default=None,
        help="Frozen mixed-calibration manifest. When set, --dataset/start_idx/subset_seed selection is bypassed.",
    )
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--token_per_sample", type=int, default=2048)
    p.add_argument(
        "--max_filter_multiplier",
        type=float,
        default=4.0,
        help="Cap token-filter scanning pool to at most num_samples * multiplier (<=0 disables the cap).",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--ema", type=float, default=0.9)
    p.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "ema"],
        help="Aggregate per-batch scores with an order-independent arithmetic mean or legacy EMA.",
    )
    p.add_argument(
        "--fill_zero_for_unrouted",
        action="store_true",
        help="Update unrouted experts with zero-filled loop_1 activations instead of leaving them untouched.",
    )
    p.add_argument(
        "--loss_fn",
        type=str,
        default="rel_l2",
        choices=["l2", "rel_l2", "cosine", "kl_div"],
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


def _identity_collate(batch):
    return batch


def _load_selection_manifest(path: str) -> tuple[dict, str]:
    with open(path, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("Selection manifest must be a JSON object containing a `samples` list.")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(
            f"Unsupported selection manifest schema_version={payload.get('schema_version')!r}; expected 1."
        )
    samples = payload["samples"]
    if not samples:
        raise ValueError("Selection manifest contains no samples.")
    score_tokens_per_sample = int(payload.get("score_tokens_per_sample", 0))
    if score_tokens_per_sample <= 0:
        raise ValueError(
            "Selection manifest must define a positive score_tokens_per_sample."
        )
    required_sample_fields = {"dataset_name", "dataset_index", "sample_id"}
    for sample_idx, sample in enumerate(samples):
        missing = sorted(required_sample_fields - set(sample))
        if missing:
            raise ValueError(
                f"Manifest sample {sample_idx} is missing required fields: {missing}."
            )
    ranks = [int(item.get("selection_rank", idx)) for idx, item in enumerate(samples)]
    if sorted(ranks) != list(range(len(samples))):
        raise ValueError("Manifest selection_rank values must be exactly 0..num_samples-1.")
    payload["samples"] = [item for _, item in sorted(zip(ranks, samples))]
    return payload, hashlib.sha256(raw).hexdigest()


def _count_sample_tokens(bundle, dataset_name: str, sample: Dict) -> int:
    batch = custom_collate_fn([sample])
    inputs = prepare_inputs(bundle, batch, dataset_name)
    attention_mask = None
    if hasattr(inputs, "get"):
        attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is None and hasattr(inputs, "__getitem__"):
        try:
            attention_mask = inputs["attention_mask"]
        except Exception:
            attention_mask = None
    if attention_mask is None:
        raise KeyError("prepare_inputs did not return `attention_mask`.")
    return int(attention_mask[0].sum().item())


def _select_calibration_indices(dataset, bundle, args) -> List[int]:
    pool_end = len(dataset)
    if args.max_filter_multiplier is not None and args.max_filter_multiplier > 0:
        max_candidates = max(args.num_samples, int(args.num_samples * args.max_filter_multiplier))
        pool_end = min(pool_end, args.start_idx + max_candidates)
    pool = list(range(args.start_idx, pool_end))
    if not pool:
        print(
            f"[calibration] Warning: empty candidate pool for start_idx={args.start_idx}. "
            "Continuing with zero samples."
        )
        return []
    if pool_end < len(dataset):
        print(
            f"[calibration] Candidate pool capped to {len(pool)} samples "
            f"(start_idx={args.start_idx}, max_filter_multiplier={args.max_filter_multiplier})."
        )

    scan_indices = pool[:]
    rng = None
    if args.subset_seed is not None and args.subset_seed >= 0:
        rng = random.Random(args.subset_seed)
        rng.shuffle(scan_indices)

    qualified_indices: List[int] = []
    fallback_indices: List[int] = []
    progress = tqdm(scan_indices, desc="Filtering samples", leave=False)
    for sample_idx in progress:
        token_count = _count_sample_tokens(bundle, args.dataset, dataset[sample_idx])
        if token_count >= args.token_per_sample:
            qualified_indices.append(sample_idx)
        else:
            fallback_indices.append(sample_idx)
        progress.set_postfix(
            qualified=len(qualified_indices),
            short=len(fallback_indices),
            refresh=False,
        )
    progress.close()

    selected = qualified_indices[: args.num_samples]
    if len(selected) < args.num_samples and fallback_indices:
        deficit = args.num_samples - len(selected)
        if rng is None:
            rng = random.Random()
        supplement = rng.sample(fallback_indices, min(deficit, len(fallback_indices)))
        selected.extend(supplement)
        print(
            f"[calibration] Qualified samples are insufficient for token_per_sample={args.token_per_sample}. "
            f"Supplemented {len(supplement)} sample(s) from shorter candidates."
        )

    print(
        f"[calibration] Calibration subset prepared from {len(pool)} candidates: "
        f"qualified={len(qualified_indices)}, short={len(fallback_indices)}, "
        f"selected={len(selected)}/{args.num_samples}."
    )
    if len(selected) < args.num_samples:
        print(
            f"[calibration] Warning: requested {args.num_samples} samples, but only "
            f"{len(selected)} are available after fallback. Continuing collection."
        )
    return selected


def _build_calibration_dataset(bundle, args):
    dataset_kwargs = {}
    if args.dataset == "m4_instruct":
        # M4 rows are pre-materialized in a Python list before token filtering.
        # Keep a larger candidate pool than `num_samples` so token-length filtering
        # can still backfill from remaining rows.
        dataset_kwargs["max_rows"] = max(args.start_idx + args.num_samples * 4, 4096)
    return build_dataset(args.dataset, bundle.family, **dataset_kwargs)


def _assert_cuda_runtime_compat(device_map: str) -> None:
    if not isinstance(device_map, str):
        return
    lowered = device_map.lower()
    if lowered != "auto" and not lowered.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return
    try:
        device_idx = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_idx)
        runtime_arch = f"sm_{major}{minor}"
        built_arches = set(torch.cuda.get_arch_list())
    except Exception:
        return
    if runtime_arch in built_arches:
        return
    raise RuntimeError(
        "Current PyTorch CUDA build does not support this GPU architecture: "
        f"runtime device requires `{runtime_arch}`, but torch was built for {sorted(built_arches)}. "
        "This causes `no kernel image is available for execution on the device`. "
        "Please install a torch build that supports your GPU (for H20/sm_90, use a modern CUDA12 torch), "
        "or run with `--device_map cpu` as a fallback."
    )


def run_collection(args) -> None:
    selection_manifest = None
    if args.selection_manifest:
        selection_manifest, manifest_sha256 = _load_selection_manifest(
            args.selection_manifest
        )
        args.dataset = "mixed"
        args.selection_manifest_sha256 = manifest_sha256
        args.selection_source_summary = selection_manifest.get("source_summary", {})
        args.score_tokens_per_sample = int(
            selection_manifest["score_tokens_per_sample"]
        )
        args.score_token_sampling = selection_manifest.get(
            "score_token_sampling",
            "per_sample_proportional_modality_uniform_positions",
        )
    else:
        args.dataset = normalize_dataset_name(args.dataset)
        args.score_tokens_per_sample = None
        args.score_token_sampling = None
    supported_datasets = {"gqa", "coco", "video_mmmu", "m4_instruct", "star"}
    if selection_manifest is None and args.dataset not in supported_datasets:
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
    _assert_cuda_runtime_compat(device_map)

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

    if selection_manifest is not None:
        manifest_samples = selection_manifest["samples"]
        dataset = ManifestRawDataset(
            manifest_samples,
            num_video_frames=int(selection_manifest.get("num_video_frames", 8)),
            video_max_long_side=int(
                selection_manifest.get("video_max_long_side", 480)
            ),
        )
        args.num_samples = len(manifest_samples)
        args.selected_num_samples = len(manifest_samples)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_identity_collate,
        )
        print(
            f"[calibration] Loaded frozen mixed selection with {len(dataset)} samples "
            f"from {args.selection_manifest} (sha256={args.selection_manifest_sha256[:12]}...)."
        )
    else:
        dataset = _build_calibration_dataset(bundle, args)
        indices = _select_calibration_indices(dataset, bundle, args)
        args.selected_num_samples = len(indices)
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
        layer_loss, layer_second_order_sum = block_forward(
            bundle=bundle,
            cnt_block=copied_block,
            layer_idx=layer_idx,
            dataloader=loader,
            dataset_name=args.dataset,
            saliency_ema=args.ema,
            score_aggregation=args.aggregation,
            fill_zero_for_unrouted=args.fill_zero_for_unrouted,
            loss_fn=args.loss_fn,
            dtype=block_dtype,
            verbose=True,
            raw_samples=selection_manifest is not None,
            score_tokens_per_sample=args.score_tokens_per_sample,
        )
        accumulator.layerwise_loss[layer_idx] = float(layer_loss)
        accumulator.layerwise_second_order_sum[layer_idx] = float(layer_second_order_sum)
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
