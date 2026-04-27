"""在线版 synthetic hidden distillation, 不依赖预先生成的 teacher cache 文件.

整体分两阶段, 但都在一次进程里顺序完成:

1) 初始化阶段 (filling synthetic pool)
   - 从多数据集 round-robin 流中反复取 raw batch, 前向教师模型, 在指定 decoder 层取 block 输出,
     再按 ``compression_mode`` 压到 ``compressed_length``, 得到与离线 cache 同构的
     ``(B, L, D)`` teacher hidden, 以及 ``modality_labels`` / ``position_ids``.
   - 连续累积直到凑满 ``synthetic_size`` 条序列, 作为合成集在参数空间里的初值模板.
   - 若 ``lambda_block > 0``, 初始化时还会为每条合成 anchor 缓存下一层 block 的输出,
     供后续 block 监督使用 (在线版不再每步前向下一层).

2) 蒸馏阶段 (optimization loop)
   - 仍从同一数据流在线抽 ``teacher_batch_size`` 条 teacher 序列, 与当前可学习的 synthetic
     表示算分布损失; 教师侧每步都是新样本, 不再读固定 ``.pt`` shard.
   - 可学习参数是 ``_build_synthetic_banks`` 返回的 ``ParameterDict`` (按模态分 bank),
     通过 ``_assemble_synthetic_hidden`` 拼出与模板同形状的 ``synthetic_hidden``;
     ``position_ids``, ``modality_labels`` 在在线脚本里冻结为初始化时的值.
   - 优化器为 Adam, 对 ``bank_params`` 做 ``backward`` + ``step``.

与 ``distill_synthetic_hidden.py`` 的关系: 损失定义, 加权, 消融, diversity warmup 等逻辑复用该模块;
本文件只负责数据管线 (流式 teacher) 与在线特有的 ``anchor_next_block_targets`` 持久化.

Loss 项 (详见 ``_compute_losses``): 按模态分组的 MMD / cov / mean / var, 以及全局 synthetic token
上的 diversity (div); 可选 ``block_rel_l2`` 将 synthetic 与初始化时缓存的 next-block teacher 对齐.
"""
import argparse
import os
import random
import sys
from typing import Dict, List, Sequence

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from tqdm import tqdm

from src.calibration.representation_distill.build_teacher_hidden_cache import (
    _assert_cuda_runtime_compat,
    _count_sample_tokens,
    _identity_collate,
    _parse_dtype,
    _select_teacher_samples,
)
from src.calibration.representation_distill.common import (
    BlockExtractionResult,
    CompressedResult,
    build_compression_token_masks,
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
from src.calibration.representation_distill.distill_synthetic_hidden import (
    _apply_ablation_presets,
    _assemble_synthetic_hidden,
    _build_synthetic_banks,
    _compute_cached_next_block_rel_l2,
    _compute_diagnostics,
    _compute_losses,
    _compute_weighted_total_loss,
    _log_stage,
    _sample_synthetic_batch_indices,
)
from src.calibration.representation_distill.runtime.dump_original_data import (
    SUPPORTED_DATASETS,
    dump_original_data,
)
from src.calibration.representation_distill.runtime.forward_from_hidden import (
    forward_from_hidden,
)


# ---------------------------------------------------------------------------
# 数据流: 多数据集 round-robin, 为在线前向提供无限样本序列 (实际受每数据集样本数限制)
# ---------------------------------------------------------------------------


class _RoundRobinDatasetStream:
    """在多个 teacher 数据集之间轮流取 batch, 每个数据集内部指针循环, 用尽则 reshuffle.

    ``next_batch`` 每次返回 ``(dataset_name, raw_batch)``, 供 ``prepare_raw_batch_inputs`` 使用.
    """

    def __init__(
        self,
        samples_by_dataset: Dict[str, List[Dict]],
        *,
        batch_size: int,
        seed: int,
        dataset_order: Sequence[str],
    ):
        self.samples_by_dataset = {
            name: list(samples)
            for name, samples in samples_by_dataset.items()
            if samples
        }
        self.batch_size = int(batch_size)
        self.dataset_order = [name for name in dataset_order if name in self.samples_by_dataset]
        if not self.dataset_order:
            raise ValueError("Round-robin stream has no non-empty datasets.")
        self._rng = random.Random(seed)
        self._pointers = {name: 0 for name in self.dataset_order}
        for name in self.dataset_order:
            self._rng.shuffle(self.samples_by_dataset[name])
        self._dataset_cursor = 0

    def next_batch(self) -> tuple[str, List[Dict]]:
        # 轮询数据集游标, 从当前数据集顺序取 ``batch_size`` 条样本 (可能跨 epoch reshuffle)
        dataset_name = self.dataset_order[self._dataset_cursor]
        self._dataset_cursor = (self._dataset_cursor + 1) % len(self.dataset_order)
        dataset_samples = self.samples_by_dataset[dataset_name]
        batch = []
        while len(batch) < self.batch_size:
            pointer = self._pointers[dataset_name]
            if pointer >= len(dataset_samples):
                self._rng.shuffle(dataset_samples)
                pointer = 0
            batch.append(dataset_samples[pointer])
            pointer += 1
            self._pointers[dataset_name] = pointer
        return dataset_name, batch


def _filter_and_group_samples(samples: List[Dict], bundle, args) -> Dict[str, List[Dict]]:
    # 按 ``teacher_datasets`` 分组, 并对每个数据集调用 ``_select_teacher_samples`` (与离线脚本一致的筛选/重排)
    grouped = {name: [] for name in args.teacher_datasets}
    for sample in samples:
        grouped[sample["dataset_name"]].append(sample)

    filtered: Dict[str, List[Dict]] = {}
    for dataset_name in args.teacher_datasets:
        dataset_samples = grouped.get(dataset_name, [])
        if not dataset_samples:
            filtered[dataset_name] = []
            continue
        reordered = _select_teacher_samples(dataset_samples, bundle, args)
        filtered[dataset_name] = reordered
        _log_stage(
            f"Prepared dataset stream `{dataset_name}` with {len(reordered)} samples."
        )
    return filtered


# ---------------------------------------------------------------------------
# 落盘: 与离线蒸馏 payload 结构对齐, 额外可带 ``anchor_next_block_targets`` (在线 block 监督锚点)
# ---------------------------------------------------------------------------


def _build_output_payload(
    *,
    synthetic_hidden: torch.Tensor,
    synth_position_ids: torch.Tensor,
    synth_labels: torch.Tensor | None,
    anchor_next_block_targets: torch.Tensor | None,
    args,
    teacher_meta: dict,
    final_losses: dict[str, float] | None,
    history: list[dict],
) -> dict:
    # ``synthetic_hidden`` 已是 float32 CPU; ``metadata`` 记录超参与训练末期的标量 loss 历史
    payload = {
        "synthetic_hidden": synthetic_hidden.detach().cpu().float(),
        "attention_mask": torch.ones(
            synthetic_hidden.shape[0],
            synthetic_hidden.shape[1],
            dtype=torch.long,
        ),
        "position_ids": synth_position_ids.detach().cpu().long(),
        "metadata": {
            "method": "multimodal_representation_level_calibration_distillation_online",
            "teacher_metadata": teacher_meta,
            "synthetic_size": args.synthetic_size,
            "synthetic_batch_size": args.synthetic_batch_size,
            "teacher_batch_size": args.teacher_batch_size,
            "train_steps": args.train_steps,
            "teacher_datasets": list(args.teacher_datasets),
            "samples_per_dataset": args.samples_per_dataset,
            "compressed_length": int(synthetic_hidden.shape[1]),
            "hidden_size": int(synthetic_hidden.shape[2]),
            "dtype": "float32",
            "lr": args.lr,
            "seed": args.seed,
            "train_dtype": args.train_dtype,
            "position_ids_strategy": "frozen_from_stream_init_samples",
            "wandb_every_n_steps": args.wandb_every_n_steps,
            "loss_weights": {
                "mmd": args.lambda_mmd,
                "cov": args.lambda_cov,
                "div": args.lambda_div,
                "mean": args.lambda_mean,
                "var": args.lambda_var,
                "block_rel_l2": args.lambda_block,
            },
            "diversity_ablation": args.diversity_ablation,
            "distribution_ablation": args.distribution_ablation,
            "div_warmup_steps": args.div_warmup_steps,
            "mmd_subsample": args.mmd_subsample,
            "final_losses": final_losses,
            "history": history,
            "created_at": utc_now_iso(),
        },
    }
    if synth_labels is not None:
        payload["modality_labels"] = synth_labels.detach().cpu()
    if anchor_next_block_targets is not None:
        payload["anchor_next_block_targets"] = anchor_next_block_targets.detach().cpu().float()
    return payload


def _save_payload(
    *,
    output_path: str,
    synthetic_hidden: torch.Tensor,
    synth_position_ids: torch.Tensor,
    synth_labels: torch.Tensor | None,
    anchor_next_block_targets: torch.Tensor | None,
    args,
    teacher_meta: dict,
    final_losses: dict[str, float] | None,
    history: list[dict],
) -> dict:
    payload = _build_output_payload(
        synthetic_hidden=synthetic_hidden,
        synth_position_ids=synth_position_ids,
        synth_labels=synth_labels,
        anchor_next_block_targets=anchor_next_block_targets,
        args=args,
        teacher_meta=teacher_meta,
        final_losses=final_losses,
        history=history,
    )
    torch.save(payload, output_path)
    return payload


def _prune_saved_pt_paths(saved_paths: list[str], *, max_keep: int) -> list[str]:
    if max_keep <= 0:
        max_keep = 1
    retained: list[str] = []
    for path in saved_paths:
        if path not in retained:
            retained.append(path)
    while len(retained) > max_keep:
        stale_path = retained.pop(0)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            _log_stage(f"Removed old checkpoint: {stale_path}")
    return retained


# ---------------------------------------------------------------------------
# 单次教师前向: raw batch -> 压缩 hidden (+ 可选 next block 目标)
# ---------------------------------------------------------------------------


def _extract_teacher_chunk(
    *,
    bundle,
    raw_batch: Sequence[Dict],
    teacher_layer: int,
    compressed_length: int,
    compression_mode: str,
    attn_temperature: float,
    modality_aware_compression: bool,
    train_dtype: torch.dtype,
    sample_generator: torch.Generator,
    include_next_block: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # 与 ``build_teacher_hidden_cache`` 同一条链路: 取 layer hidden, 再 ``compress_hidden_states`` 到固定 token 数
    inputs = prepare_raw_batch_inputs(bundle, raw_batch)
    inputs = move_inputs_to_model_device(bundle.model, inputs)
    compression_masks = None
    if modality_aware_compression:
        compression_masks = build_compression_token_masks(bundle, inputs)
    extraction: BlockExtractionResult = extract_block_output(
        bundle,
        inputs,
        layer_idx=teacher_layer,
        capture_attn_importance=(compression_mode == "attention_weighted"),
    )
    result: CompressedResult = compress_hidden_states(
        hidden_states=extraction.hidden_states,
        attention_mask=inputs["attention_mask"],
        target_length=compressed_length,
        compression_masks=compression_masks,
        modality_lengths=None,
        mode=compression_mode,
        generator=sample_generator,
        attn_importance=extraction.attn_importance,
        attn_temperature=attn_temperature,
    )

    # 从压缩后的表征继续跑一层 decoder, 得到 ``teacher_layer+1`` 的 block 输出, 仅 CPU 侧缓存给 block loss
    next_block_target = None
    if include_next_block:
        compressed_attention_mask = torch.ones(
            result.hidden_states.shape[:2],
            dtype=inputs["attention_mask"].dtype,
            device=result.hidden_states.device,
        )
        next_block_layer = get_decoder_layer(bundle, teacher_layer + 1)
        next_block_dtype = next(next_block_layer.parameters()).dtype
        next_block_hidden = forward_from_hidden(
            bundle=bundle,
            hidden_states=result.hidden_states.to(dtype=next_block_dtype),
            attention_mask=compressed_attention_mask,
            start_layer=teacher_layer + 1,
            end_layer=teacher_layer + 1,
            position_ids=result.position_ids,
            apply_final_norm=False,
        )
        next_block_target = next_block_hidden.to(dtype=train_dtype).cpu()
        del next_block_hidden

    hidden = result.hidden_states.to(dtype=train_dtype).cpu()
    labels = result.modality_labels.cpu()
    position_ids = result.position_ids.cpu()
    del inputs, extraction, result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hidden, labels, position_ids, next_block_target


# ---------------------------------------------------------------------------
# 累积 ``target_count`` 条压缩 teacher 样本: 反复 ``next_batch`` + ``_extract_teacher_chunk``, ``torch.cat`` 拼接
# ---------------------------------------------------------------------------


def _collect_stream_examples(
    *,
    stream: _RoundRobinDatasetStream,
    target_count: int,
    bundle,
    args,
    train_dtype: torch.dtype,
    sample_generator: torch.Generator,
    include_next_block: bool,
    desc: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    hidden_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    position_parts: List[torch.Tensor] = []
    next_parts: List[torch.Tensor] = []
    collected = 0
    progress = None
    if desc is not None:
        progress = tqdm(total=target_count, desc=desc, dynamic_ncols=True, leave=False)
    # 若单次前向 batch 大于剩余条数, 只截取前 ``take`` 行, 避免池子大小被 batch 边界撑爆
    while collected < target_count:
        dataset_name, raw_batch = stream.next_batch()
        hidden, labels, position_ids, next_block_target = _extract_teacher_chunk(
            bundle=bundle,
            raw_batch=raw_batch,
            teacher_layer=args.teacher_layer,
            compressed_length=args.compressed_length,
            compression_mode=args.compression_mode,
            attn_temperature=args.attn_temperature,
            modality_aware_compression=args.modality_aware_compression,
            train_dtype=train_dtype,
            sample_generator=sample_generator,
            include_next_block=include_next_block,
        )
        remaining = target_count - collected
        take = min(remaining, hidden.shape[0])
        hidden_parts.append(hidden[:take])
        label_parts.append(labels[:take])
        position_parts.append(position_ids[:take])
        if include_next_block and next_block_target is not None:
            next_parts.append(next_block_target[:take])
        collected += take
        if progress is not None:
            progress.update(take)
            progress.set_postfix(dataset=dataset_name)
    if progress is not None:
        progress.close()
    out_next = torch.cat(next_parts, dim=0) if next_parts else None
    return (
        torch.cat(hidden_parts, dim=0),
        torch.cat(label_parts, dim=0),
        torch.cat(position_parts, dim=0),
        out_next,
    )


# ---------------------------------------------------------------------------
# CLI: 在线蒸馏独有参数 (数据流, 压缩, 数据集列表); 损失权重与离线脚本含义一致
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online synthetic hidden distillation without prebuilt teacher cache."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--teacher_layer", type=int, default=0)
    parser.add_argument("--compressed_length", type=int, default=64)
    parser.add_argument(
        "--compression_mode",
        type=str,
        default="sample",
        choices=["pool", "sample", "attention_weighted"],
    )
    parser.add_argument("--attn_temperature", type=float, default=1.0)
    parser.add_argument("--modality_aware_compression", action="store_true")
    parser.add_argument("--samples_per_dataset", type=int, default=1024)
    parser.add_argument(
        "--teacher_datasets",
        type=str,
        nargs="+",
        default=["gqa", "coco", "m4_instruct"],
        choices=list(SUPPORTED_DATASETS),
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--teacher_batch_size", type=int, default=256)
    parser.add_argument("--synthetic_size", type=int, default=256)
    parser.add_argument("--synthetic_batch_size", type=int, default=0)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--shuffle_seed", type=int, default=1234)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--video_max_long_side", type=int, default=480)
    parser.add_argument("--token_per_sample", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--train_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--init_std", type=float, default=0.0)
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_cov", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    parser.add_argument("--lambda_mean", type=float, default=0.5)
    parser.add_argument("--lambda_var", type=float, default=0.5)
    parser.add_argument("--lambda_block", type=float, default=0.0)
    parser.add_argument(
        "--diversity_ablation",
        type=str,
        default="full",
        choices=["full", "no_div"],
    )
    parser.add_argument(
        "--distribution_ablation",
        type=str,
        default="full",
        choices=["full", "moment_only", "mmd_only"],
    )
    parser.add_argument("--div_warmup_steps", type=int, default=200)
    parser.add_argument("--mmd_subsample", type=int, default=2048)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_every_n_steps", type=int, default=1)
    parser.add_argument("--use_ema_normalized_losses", action="store_true")
    parser.add_argument("--loss_ema_decay", type=float, default=0.99)
    parser.add_argument("--checkpoint_interval", type=int, default=0)
    return parser


def main() -> None:
    # ----- 通用准备: 随机种子, 输出目录, 设备, 教师模型 bundle, 数据元信息 -----
    args = build_arg_parser().parse_args()
    ablation_notes = _apply_ablation_presets(args)
    train_dtype = _parse_dtype(args.train_dtype)
    seed_everything(args.seed)
    ensure_dir(os.path.dirname(args.output_path))

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    _assert_cuda_runtime_compat(device_map)

    _log_stage(
        "Starting online synthetic hidden distillation "
        f"(train_steps={args.train_steps}, synthetic_size={args.synthetic_size}, "
        f"synthetic_batch_size={args.synthetic_batch_size}, teacher_batch_size={args.teacher_batch_size}, "
        f"teacher_datasets={list(args.teacher_datasets)})."
    )

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "model_name_or_path": args.model_name_or_path,
                "output_path": args.output_path,
                "teacher_layer": args.teacher_layer,
                "compressed_length": args.compressed_length,
                "compression_mode": args.compression_mode,
                "modality_aware_compression": args.modality_aware_compression,
                "teacher_datasets": list(args.teacher_datasets),
                "samples_per_dataset": args.samples_per_dataset,
                "teacher_batch_size": args.teacher_batch_size,
                "synthetic_size": args.synthetic_size,
                "synthetic_batch_size": args.synthetic_batch_size,
                "train_steps": args.train_steps,
                "lr": args.lr,
                "seed": args.seed,
                "train_dtype": args.train_dtype,
                "lambda_mmd": args.lambda_mmd,
                "lambda_cov": args.lambda_cov,
                "lambda_div": args.lambda_div,
                "lambda_mean": args.lambda_mean,
                "lambda_var": args.lambda_var,
                "lambda_block": args.lambda_block,
                "diversity_ablation": args.diversity_ablation,
                "distribution_ablation": args.distribution_ablation,
                "div_warmup_steps": args.div_warmup_steps,
                "mmd_subsample": args.mmd_subsample,
                "wandb_every_n_steps": args.wandb_every_n_steps,
                "ablation_notes": ablation_notes,
            },
        )
        wandb_run.define_metric("step")
        wandb_run.define_metric("loss/*", step_metric="step")
        wandb_run.define_metric("diag/*", step_metric="step")
        wandb_run.define_metric("schedule/*", step_metric="step")

    from observations.common import load_model_bundle, resolve_model_name_or_path

    max_decoder_layer = args.teacher_layer + (1 if args.lambda_block > 0 else 0)
    bundle = load_model_bundle(
        args.model_name_or_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        max_decoder_layer=max_decoder_layer,
    )
    num_layers = get_num_decoder_layers(bundle)
    if args.teacher_layer < 0 or args.teacher_layer >= num_layers:
        raise ValueError(
            f"Invalid teacher_layer={args.teacher_layer}; model has {num_layers} decoder layers."
        )
    if args.lambda_block > 0 and args.teacher_layer >= num_layers - 1:
        raise ValueError("lambda_block > 0 requires a valid next decoder block.")

    # ----- 构造 round-robin 流: 磁盘上 dump 样本列表, 再按数据集分组并筛选 -----
    all_samples = dump_original_data(
        output_dir=os.path.dirname(args.output_path),
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
        shuffle_seed=None,
        num_video_frames=args.num_video_frames,
        video_max_long_side=args.video_max_long_side,
        selected_datasets=list(args.teacher_datasets),
    )
    grouped_samples = _filter_and_group_samples(all_samples, bundle, args)
    stream = _RoundRobinDatasetStream(
        grouped_samples,
        batch_size=args.batch_size,
        seed=args.shuffle_seed,
        dataset_order=list(args.teacher_datasets),
    )
    sample_generator = torch.Generator()
    sample_generator.manual_seed(args.seed)

    # 写入最终 ``.pt`` 的 teacher 侧元信息 (不含大 tensor, 仅描述配置)
    teacher_meta = {
        "method": "multimodal_representation_level_calibration_distillation_online",
        "model_name_or_path": args.model_name_or_path,
        "resolved_model_name_or_path": resolve_model_name_or_path(args.model_name_or_path),
        "processor_name": bundle.processor.__class__.__name__,
        "model_family": bundle.family,
        "teacher_layer": args.teacher_layer,
        "teacher_layer_type": "full_block_output",
        "cached_next_block_layer": args.teacher_layer + 1 if args.lambda_block > 0 else None,
        "compressed_length": args.compressed_length,
        "compression_mode": args.compression_mode,
        "hidden_size": get_hidden_size(bundle),
        "teacher_datasets": list(args.teacher_datasets),
        "samples_per_dataset": args.samples_per_dataset,
        "token_per_sample": args.token_per_sample,
        "num_video_frames": args.num_video_frames,
        "video_max_long_side": args.video_max_long_side,
        "created_at": utc_now_iso(),
    }

    # ----- 阶段一: 用流式 teacher 填满 ``synthetic_size`` 的初始化池 (CPU 缓存再搬到训练 device) -----
    # 此处每条样本对应之后 synthetic 的一条序列; 若开 block loss, 同时缓存与之一一对应的 next-block 目标
    _log_stage("Filling synthetic pool from streamed teacher hidden states.")
    init_hidden_cpu, synth_labels_cpu, init_position_ids_cpu, anchor_next_block_targets_cpu = _collect_stream_examples(
        stream=stream,
        target_count=args.synthetic_size,
        bundle=bundle,
        args=args,
        train_dtype=train_dtype,
        sample_generator=sample_generator,
        include_next_block=(args.lambda_block > 0),
        desc="Initializing synthetic pool",
    )
    init_hidden = init_hidden_cpu.to(device=device, dtype=train_dtype)
    synth_labels = synth_labels_cpu.to(device=device)
    init_position_ids = init_position_ids_cpu.to(device=device, dtype=torch.long)
    anchor_next_block_targets = (
        anchor_next_block_targets_cpu.to(device=device, dtype=train_dtype)
        if anchor_next_block_targets_cpu is not None
        else None
    )

    # ----- 将初始化 hidden 拆成按模态的 ``nn.Parameter`` bank, 训练时只更新这些参数 -----
    # ``synth_template_labels`` / ``synth_template_bank_indices`` 把每条序列每个 token 映射到对应 bank 行,
    # ``_assemble_synthetic_hidden`` 按索引 gather 出当前步用的 ``synthetic_hidden`` 子 batch
    bank_params, synth_template_labels_flat, synth_template_bank_indices_flat = _build_synthetic_banks(
        init_hidden=init_hidden,
        init_labels=synth_labels,
        init_std=args.init_std,
    )
    synth_template_labels = synth_labels
    synth_template_bank_indices = synth_template_bank_indices_flat.view(
        init_hidden.shape[0], init_hidden.shape[1]
    )
    synth_position_ids = init_position_ids.clone()
    synth_attention_mask = torch.ones(init_hidden.shape[:2], dtype=torch.long, device=device)
    effective_synth_batch_size = (
        args.synthetic_size if args.synthetic_batch_size <= 0
        else min(args.synthetic_batch_size, args.synthetic_size)
    )

    # Adam 只作用于 ``bank_params``; ``position_ids``, 模板 label, ``anchor_next_block_targets`` 均不参与梯度
    optimizer = torch.optim.Adam(list(bank_params.parameters()), lr=args.lr)
    loss_ema_state: dict[str, torch.Tensor] | None = {} if args.use_ema_normalized_losses else None
    history: list[dict] = []
    saved_pt_paths: list[str] = []
    final_losses = None

    # ----- 阶段二: 每步从数据流再抽一批 teacher, 与 synthetic 算 loss, 反传更新 bank -----
    _log_stage(f"Entering optimization loop at step=0, target_step={args.train_steps}.")
    progress = tqdm(
        range(args.train_steps),
        desc="Distilling compact hidden (online)",
        dynamic_ncols=True,
        leave=True,
    )
    for step in progress:
        # 在线 teacher batch: 与初始化独立, 每步都是新前向, 分布上覆盖流式数据混合
        teacher_batch_cpu, teacher_batch_labels_cpu, _, _ = _collect_stream_examples(
            stream=stream,
            target_count=args.teacher_batch_size,
            bundle=bundle,
            args=args,
            train_dtype=train_dtype,
            sample_generator=sample_generator,
            include_next_block=False,
        )
        teacher_batch = teacher_batch_cpu.to(device=device, dtype=train_dtype)
        teacher_batch_labels = teacher_batch_labels_cpu.to(device=device)

        # 随机子采样 synthetic 序列索引 (可 ``synthetic_batch_size < synthetic_size`` 以降低每步显存)
        synth_batch_indices = _sample_synthetic_batch_indices(
            synthetic_size=args.synthetic_size,
            synthetic_batch_size=effective_synth_batch_size,
            device=synth_template_labels.device,
        )
        synth_batch_labels = synth_template_labels.index_select(0, synth_batch_indices)
        synth_batch_bank_indices = synth_template_bank_indices.index_select(0, synth_batch_indices)
        synth_batch_attention_mask = synth_attention_mask.index_select(0, synth_batch_indices)
        synthetic_hidden = _assemble_synthetic_hidden(
            bank_params=bank_params,
            template_labels=synth_batch_labels,
            template_bank_indices=synth_batch_bank_indices,
            synthetic_size=synth_batch_indices.shape[0],
            compressed_length=init_hidden.shape[1],
            hidden_size=init_hidden.shape[2],
        )

        # ``_compute_losses`` 返回: ``mmd``, ``cov``, ``mean``, ``var``, ``div``, 以及可选的 ``mmd/mod*`` 等诊断键
        losses = _compute_losses(
            teacher_batch,
            synthetic_hidden,
            teacher_batch_labels,
            synth_batch_labels,
            mmd_subsample=args.mmd_subsample,
        )

        # ``div`` 项乘以 ``div_scale``, 从 0 线性爬升到 1, 减轻训练早期 diversity 项主导
        div_scale = 1.0
        if args.div_warmup_steps > 0:
            div_scale = min(1.0, (step + 1) / float(args.div_warmup_steps))

        # block 监督: 用初始化阶段为同索引 synthetic 缓存的 teacher next-block 表征, 与当前 synthetic 算相对 L2
        if args.lambda_block > 0:
            cached_teacher_target = anchor_next_block_targets.index_select(0, synth_batch_indices)
            losses["block_rel_l2"] = _compute_cached_next_block_rel_l2(
                cached_teacher_target=cached_teacher_target,
                synthetic_hidden=synthetic_hidden,
                synthetic_attention_mask=synth_batch_attention_mask,
            )

        total_loss, weighted_terms, normalized_terms = _compute_weighted_total_loss(
            args=args,
            losses=losses,
            div_scale=div_scale,
            ema_state=loss_ema_state,
        )

        # 标准一阶更新: ``total_loss`` 对 ``bank_params`` 反传, Adam 步进
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_losses = {
            "total": float(total_loss.detach().cpu().item()),
            "mmd": float(losses["mmd"].detach().cpu().item()),
            "cov": float(losses["cov"].detach().cpu().item()),
            "div": float(losses["div"].detach().cpu().item()),
            "mean": float(losses["mean"].detach().cpu().item()),
            "var": float(losses["var"].detach().cpu().item()),
            "div_scale": div_scale,
        }
        if "block_rel_l2" in losses:
            final_losses["block_rel_l2"] = float(losses["block_rel_l2"].detach().cpu().item())
        for k, v in weighted_terms.items():
            final_losses[f"weighted/{k}"] = float(v.detach().cpu().item())
        for k, v in normalized_terms.items():
            final_losses[f"normalized/{k}"] = float(v.detach().cpu().item())
        for k, v in losses.items():
            if "/" in k:
                final_losses[k] = float(v.detach().cpu().item())

        # 诊断: token norm, centroid 距离, diversity 余弦分位数等, 仅周期性写入 ``final_losses`` 与 history
        should_log_diagnostics = (
            step % args.log_interval == 0
            or step == args.train_steps - 1
        )
        if should_log_diagnostics:
            diagnostics = _compute_diagnostics(
                teacher_batch=teacher_batch,
                synthetic_hidden=synthetic_hidden,
                teacher_labels=teacher_batch_labels,
                synth_labels=synth_batch_labels,
                mmd_subsample=args.mmd_subsample,
            )
            for k, v in diagnostics.items():
                final_losses[k] = float(v.detach().cpu().item())

        # 注意: 在线版未实现 teacher-teacher baseline ratio (离线 ``distill_synthetic_hidden`` 中有)
        should_log_wandb = (
            wandb_run is not None
            and (
                step == args.train_steps - 1
                or (
                    args.wandb_every_n_steps > 0
                    and step % args.wandb_every_n_steps == 0
                )
            )
        )
        if should_log_wandb:
            wandb_payload = {
                "step": step,
                "loss/total": final_losses["total"],
                "loss/raw/mmd": final_losses["mmd"],
                "loss/raw/cov": final_losses["cov"],
                "loss/raw/mean": final_losses["mean"],
                "loss/raw/var": final_losses["var"],
                "loss/raw/div": final_losses["div"],
                "loss/weighted/mmd": final_losses["weighted/mmd"],
                "loss/weighted/cov": final_losses["weighted/cov"],
                "loss/weighted/mean": final_losses["weighted/mean"],
                "loss/weighted/var": final_losses["weighted/var"],
                "loss/weighted/div": final_losses["weighted/div"],
                "schedule/div_scale": div_scale,
                "meta/lr": args.lr,
            }
            if "block_rel_l2" in final_losses:
                wandb_payload["loss/raw/block_rel_l2"] = final_losses["block_rel_l2"]
                wandb_payload["loss/weighted/block_rel_l2"] = final_losses["weighted/block_rel_l2"]
            for k, v in final_losses.items():
                if "/" in k:
                    if k.startswith("diag/"):
                        wandb_payload[k] = v
                    else:
                        wandb_payload[f"loss/{k}"] = v
            wandb_run.log(wandb_payload, step=step)

        progress.set_postfix(
            loss=f"{final_losses['total']:.4f}",
            mmd=f"{final_losses['mmd']:.4f}",
            div=f"{final_losses['div']:.4f}",
        )
        if should_log_diagnostics:
            history.append({"step": step, **final_losses})

        if (
            args.checkpoint_interval > 0
            and (step + 1) % args.checkpoint_interval == 0
            and step != args.train_steps - 1
        ):
            checkpoint_hidden = _assemble_synthetic_hidden(
                bank_params=bank_params,
                template_labels=synth_template_labels,
                template_bank_indices=synth_template_bank_indices,
                synthetic_size=args.synthetic_size,
                compressed_length=init_hidden.shape[1],
                hidden_size=init_hidden.shape[2],
            )
            checkpoint_path = os.path.join(
                os.path.dirname(args.output_path),
                f"{os.path.splitext(os.path.basename(args.output_path))[0]}-step{step + 1}.pt",
            )
            _save_payload(
                output_path=checkpoint_path,
                synthetic_hidden=checkpoint_hidden,
                synth_position_ids=synth_position_ids,
                synth_labels=synth_labels,
                anchor_next_block_targets=anchor_next_block_targets,
                args=args,
                teacher_meta=teacher_meta,
                final_losses=final_losses,
                history=history,
            )
            saved_pt_paths.append(checkpoint_path)
            saved_pt_paths = _prune_saved_pt_paths(saved_pt_paths, max_keep=2)

        # 释放本步 teacher GPU tensor, 降低峰值显存 (下一迭代会重新从 CPU 拉 batch)
        del teacher_batch_cpu, teacher_batch_labels_cpu, teacher_batch, teacher_batch_labels

    # 训练结束: 用最新 ``bank_params`` 拼出完整 ``synthetic_size`` 张量并保存主 checkpoint
    final_hidden = _assemble_synthetic_hidden(
        bank_params=bank_params,
        template_labels=synth_template_labels,
        template_bank_indices=synth_template_bank_indices,
        synthetic_size=args.synthetic_size,
        compressed_length=init_hidden.shape[1],
        hidden_size=init_hidden.shape[2],
    )
    _save_payload(
        output_path=args.output_path,
        synthetic_hidden=final_hidden,
        synth_position_ids=synth_position_ids,
        synth_labels=synth_labels,
        anchor_next_block_targets=anchor_next_block_targets,
        args=args,
        teacher_meta=teacher_meta,
        final_losses=final_losses,
        history=history,
    )
    saved_pt_paths.append(args.output_path)
    saved_pt_paths = _prune_saved_pt_paths(saved_pt_paths, max_keep=2)
    _log_stage(f"Saved online distilled hidden to {args.output_path}")


if __name__ == "__main__":
    main()
