"""用多模态教师模型前向, 抽取指定 decoder 层 hidden, 压缩到固定长度后落盘为 teacher cache.

供表征级校准蒸馏 (distill_synthetic_hidden) 使用, 输出含 teacher_cache, modality_labels, 元数据等.
"""
import argparse
import os
import sys
import random
from typing import Dict, List

# 将仓库根与父目录加入 sys.path, 以便以包形式导入 src.* 与 observations.*
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
    BlockExtractionResult,
    CompressedResult,
    build_compression_token_masks,
    build_sample_manifest,
    compress_hidden_states,
    ensure_dir,
    extract_block_output,
    get_decoder_layer,
    get_hidden_size,
    get_num_decoder_layers,
    move_inputs_to_model_device,
    prepare_raw_batch_inputs,
    seed_everything,
    utc_now_iso,
)
from src.calibration.representation_distill.runtime.forward_from_hidden import forward_from_hidden
from src.calibration.representation_distill.runtime.dump_original_data import (
    SUPPORTED_DATASETS,
    dump_original_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_collate(batch: List[dict]) -> List[dict]:
    # 不做 stack, 保持 batch 为样本 dict 列表, 供多模态变长字段后续逐条处理
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


def _count_sample_tokens(bundle, sample: Dict) -> int:
    inputs = prepare_raw_batch_inputs(bundle, [sample])
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
        raise KeyError("prepare_raw_batch_inputs did not return `attention_mask`.")
    return int(attention_mask[0].sum().item())


def _select_teacher_samples(samples: List[Dict], bundle, args) -> List[Dict]:
    if not samples:
        print("[representation_distill] Warning: empty teacher sample pool. Continuing with zero samples.")
        return []

    scan_indices = list(range(len(samples)))
    rng = None
    if args.subset_seed is not None and args.subset_seed >= 0:
        rng = random.Random(args.subset_seed)
        rng.shuffle(scan_indices)

    qualified_indices: List[int] = []
    fallback_indices: List[int] = []
    for sample_idx in scan_indices:
        token_count = _count_sample_tokens(bundle, samples[sample_idx])
        if token_count >= args.token_per_sample:
            qualified_indices.append(sample_idx)
        else:
            fallback_indices.append(sample_idx)

    target_size = len(samples)
    selected_indices = qualified_indices[:target_size]
    if len(selected_indices) < target_size and fallback_indices:
        deficit = target_size - len(selected_indices)
        if rng is None:
            rng = random.Random()
        supplement = rng.sample(fallback_indices, min(deficit, len(fallback_indices)))
        selected_indices.extend(supplement)
        print(
            f"[representation_distill] Qualified samples are insufficient for token_per_sample={args.token_per_sample}. "
            f"Supplemented {len(supplement)} sample(s) from shorter candidates."
        )

    print(
        f"[representation_distill] Teacher sample subset prepared from {len(samples)} candidates: "
        f"qualified={len(qualified_indices)}, short={len(fallback_indices)}, "
        f"selected={len(selected_indices)}/{target_size}."
    )
    if len(selected_indices) < target_size:
        print(
            f"[representation_distill] Warning: requested {target_size} samples, but only "
            f"{len(selected_indices)} are available after fallback. Continuing extraction."
        )
    return [samples[idx] for idx in selected_indices]


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


def _build_shard_dir(output_path: str) -> str:
    return os.path.join(
        os.path.dirname(output_path),
        f"{os.path.splitext(os.path.basename(output_path))[0]}_shards",
    )


def _write_cache_shard(
    *,
    shard_dir: str,
    shard_idx: int,
    teacher_cache: torch.Tensor,
    modality_labels: torch.Tensor,
    position_ids: torch.Tensor,
    next_block_cache: torch.Tensor | None,
) -> dict:
    ensure_dir(shard_dir)
    shard_name = f"shard_{shard_idx:06d}.pt"
    shard_path = os.path.join(shard_dir, shard_name)
    shard_payload = {
        "teacher_cache": teacher_cache,
        "modality_labels": modality_labels,
        "position_ids": position_ids,
        "num_samples": int(teacher_cache.shape[0]),
    }
    if next_block_cache is not None:
        shard_payload["next_block_cache"] = next_block_cache
    torch.save(shard_payload, shard_path)
    return {
        "path": os.path.join(os.path.basename(shard_dir), shard_name),
        "num_samples": int(teacher_cache.shape[0]),
    }


# ---------------------------------------------------------------------------
# CLI (命令行参数)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a multimodal teacher hidden cache for representation-level calibration distillation."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # 取第几层 decoder block 的输出作为教师表征
    parser.add_argument("--teacher_layer", type=int, default=0)
    # 沿序列维压缩后的 token 数, 控制缓存体积与下游蒸馏成本
    parser.add_argument("--compressed_length", type=int, default=64)
    parser.add_argument(
        "--compression_mode",
        type=str,
        default="sample",
        choices=["pool", "sample", "attention_weighted"],
        help=(
            "'sample' keeps real tokens via uniform random sampling; "
            "'attention_weighted' uses attention importance to prefer informative tokens; "
            "'pool' uses legacy mean-pooling."
        ),
    )
    parser.add_argument(
        "--attn_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for attention_weighted sampling. "
             "Larger → more uniform; smaller → more greedy. Only used when compression_mode=attention_weighted.",
    )
    parser.add_argument(
        "--modality_aware_compression",
        action="store_true",
        help="Pool/sample text/image/video tokens separately before concatenating them into the teacher cache.",
    )
    # 每个子数据集各采多少条, 总样本数约为 len(teacher_datasets) * samples_per_dataset
    parser.add_argument("--samples_per_dataset", type=int, default=1024)
    parser.add_argument(
        "--teacher_datasets",
        type=str,
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=list(SUPPORTED_DATASETS),
        help="Teacher pool datasets to include. Default uses the original 4-dataset mixture.",
    )
    # 前向时的 DataLoader batch, 与蒸馏里的 teacher_batch_size 含义不同
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--token_per_sample", type=int, default=2048)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_seed", type=int, default=1234)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--video_max_long_side", type=int, default=480)
    # 写入磁盘的 teacher_cache 元素类型, 可与模型前向 dtype 不同以省空间
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--cache_next_block_targets",
        action="store_true",
        help=(
            "Also cache the next decoder block output for the compressed teacher hidden states. "
            "This lets distillation use block supervision without loading the teacher model online."
        ),
    )
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
    # 预留: 按模态固定长度压缩时可传入, 当前为 None 表示走通用路径
    modality_lengths = None

    from observations.common import load_model_bundle, resolve_model_name_or_path

    # 1) 加载教师模型与 processor
    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    _assert_cuda_runtime_compat(device_map)

    bundle = load_model_bundle(
        args.model_name_or_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        max_decoder_layer=args.teacher_layer + (1 if args.cache_next_block_targets else 0),
    )
    num_layers = get_num_decoder_layers(bundle)
    if args.teacher_layer < 0 or args.teacher_layer >= num_layers:
        raise ValueError(
            f"Invalid teacher_layer={args.teacher_layer}; model has {num_layers} decoder layers."
        )
    if args.cache_next_block_targets and args.teacher_layer >= num_layers - 1:
        raise ValueError(
            "Cannot cache next-block targets for the final decoder layer because there is no subsequent block."
        )

    # 2) 从选定数据混合中导出原始样本列表 (路径, 文本等), 顺序已按脚本内逻辑固定
    selected_datasets = list(dict.fromkeys(args.teacher_datasets))
    samples = dump_original_data(
        output_dir=os.path.dirname(args.output_path),
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        num_video_frames=args.num_video_frames,
        video_max_long_side=args.video_max_long_side,
        selected_datasets=selected_datasets,
    )
    samples = _select_teacher_samples(samples, bundle, args)
    # shuffle=False: 样本顺序由 dump_original_data 决定, 可复现
    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_identity_collate,
    )

    save_dtype = _parse_dtype(args.save_dtype)
    shard_dir = _build_shard_dir(args.output_path)
    shard_manifest = []
    dataset_ids = []
    sample_manifest = []
    teacher_dtype = None
    sample_generator = torch.Generator()
    sample_generator.manual_seed(args.seed)

    # attention_weighted 压缩需要从 forward 里拿到 token 重要性
    need_attn = args.compression_mode == "attention_weighted"

    total_samples = 0
    for shard_idx, batch in enumerate(tqdm(loader, desc="Extracting teacher cache", leave=False)):
        inputs = prepare_raw_batch_inputs(bundle, batch)
        inputs = move_inputs_to_model_device(bundle.model, inputs)
        compression_masks = None
        # 为 True 时按模态分别 pool/sample, 再拼成压缩序列
        if args.modality_aware_compression:
            compression_masks = build_compression_token_masks(bundle, inputs)
        extraction: BlockExtractionResult = extract_block_output(
            bundle, inputs, layer_idx=args.teacher_layer,
            capture_attn_importance=need_attn,
        )
        hidden = extraction.hidden_states
        # 记录模型前向实际 dtype, 与 save_dtype 区分
        teacher_dtype = teacher_dtype or str(hidden.dtype).replace("torch.", "")
        # 沿序列维压到 compressed_length, 并得到逐 token 模态标签
        result: CompressedResult = compress_hidden_states(
            hidden_states=hidden,
            attention_mask=inputs["attention_mask"],
            target_length=args.compressed_length,
            compression_masks=compression_masks,
            modality_lengths=modality_lengths,
            mode=args.compression_mode,
            generator=sample_generator,
            attn_importance=extraction.attn_importance,
            attn_temperature=args.attn_temperature,
        )
        shard_teacher_cache = result.hidden_states.to(dtype=save_dtype).cpu()
        shard_next_block_cache = None
        if args.cache_next_block_targets:
            compressed_attention_mask = torch.ones(
                result.hidden_states.shape[:2],
                dtype=inputs["attention_mask"].dtype,
                device=result.hidden_states.device,
            )
            next_block_layer = get_decoder_layer(bundle, args.teacher_layer + 1)
            next_block_dtype = next(next_block_layer.parameters()).dtype
            next_block_hidden = forward_from_hidden(
                bundle=bundle,
                hidden_states=result.hidden_states.to(dtype=next_block_dtype),
                attention_mask=compressed_attention_mask,
                start_layer=args.teacher_layer + 1,
                end_layer=args.teacher_layer + 1,
                position_ids=result.position_ids,
                apply_final_norm=False,
            )
            shard_next_block_cache = next_block_hidden.to(dtype=save_dtype).cpu()
        shard_modality_labels = result.modality_labels.cpu()
        shard_position_ids = result.position_ids.cpu()
        shard_manifest.append(
            _write_cache_shard(
                shard_dir=shard_dir,
                shard_idx=shard_idx,
                teacher_cache=shard_teacher_cache,
                modality_labels=shard_modality_labels,
                position_ids=shard_position_ids,
                next_block_cache=shard_next_block_cache,
            )
        )
        total_samples += int(shard_teacher_cache.shape[0])
        dataset_ids.extend(int(sample["dataset_id"]) for sample in batch)
        sample_manifest.extend(build_sample_manifest(batch))
        del inputs, extraction, result, shard_teacher_cache, shard_modality_labels, shard_position_ids
        if shard_next_block_cache is not None:
            del shard_next_block_cache, next_block_hidden
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) 写入 manifest payload；真实 cache 已经按 shard 落盘
    _mode_suffix = "modality_aware_" if args.modality_aware_compression else "uniform_"
    compression_mode_label = _mode_suffix + args.compression_mode

    payload = {
        "shards": shard_manifest,
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
            "cached_next_block_layer": args.teacher_layer + 1 if args.cache_next_block_targets else None,
            "compressed_length": args.compressed_length,
            "compression_mode": compression_mode_label,
            "modality_lengths": modality_lengths,
            "hidden_size": get_hidden_size(bundle),
            "teacher_cache_dtype": str(save_dtype).replace("torch.", ""),
            "next_block_cache_dtype": str(save_dtype).replace("torch.", "") if args.cache_next_block_targets else None,
            "teacher_model_output_dtype": teacher_dtype,
            "samples_per_dataset": args.samples_per_dataset,
            "total_samples": total_samples,
            "token_per_sample": args.token_per_sample,
            "subset_seed": args.subset_seed,
            "dataset_id_to_name": {
                idx: name
                for idx, name in enumerate(SUPPORTED_DATASETS)
                if name in selected_datasets
            },
            "teacher_datasets": selected_datasets,
            "num_video_frames": args.num_video_frames,
            "video_max_long_side": args.video_max_long_side,
            "shuffle_seed": args.shuffle_seed,
            "attn_temperature": args.attn_temperature if need_attn else None,
            "cache_next_block_targets": bool(args.cache_next_block_targets),
            "shard_dir": os.path.basename(shard_dir),
            "created_at": utc_now_iso(),
        },
    }
    # 4) 落盘, 供 distill_synthetic_hidden 等读取
    out_path = args.output_path
    torch.save(payload, out_path)
    print(f"[representation_distill] Saved teacher cache to {out_path}")


if __name__ == "__main__":
    main()
