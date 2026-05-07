"""Prune a VL-MoE model in memory and evaluate on a given task without saving a ckpt."""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from tqdm.auto import tqdm

from observations.common import infer_model_family
from src.base.load_dataset import load_eval_task
from src.base.models import auto_load_model
from src.generate_mask import generate_masks as build_masks_pipeline
from src.prune import apply_structural_pruning
from tasks.video_mmmu import process_media as process_video_media


def _normalize_answer(s) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def move_to_device(inputs: dict, device, dtype=None) -> dict:
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if key in {"attention_mask", "images_seq_mask"}:
            moved[key] = value.to(device=device, dtype=torch.bool)
        elif (
            key in {"images", "pixel_values"}
            and dtype is not None
            and torch.is_floating_point(value)
        ):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device)
    return moved


def _resolve_text_config(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError(f"Model {type(model)} has no config")
    return getattr(cfg, "text_config", getattr(cfg, "llm_config", cfg))


def _get_batch_rows(pool, start: int, batch_size: int):
    end = min(start + batch_size, len(pool))
    return [pool[idx] for idx in range(start, end)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prune a VL-MoE model in memory and evaluate on a task without saving a checkpoint."
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--scores_path", type=str, required=True)
    p.add_argument("--task", type=str, default="gqa",
                   help="Evaluation task: gqa, coco, video_mmmu")
    p.add_argument("--thresholds_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--prune_ratio", type=float, default=0.30)
    p.add_argument("--inter_method", type=str, default="uniform")
    p.add_argument("--intra_method", type=str, default="uniform")
    p.add_argument("--intra_expert_metric", type=str, default="activation")
    p.add_argument(
        "--layerwise_loss_key",
        type=str,
        default="layerwise_second_order_sum",
        choices=["layerwise_loss", "layerwise_second_order_sum"],
        help="Which layerwise score key to use for loss-based inter-layer masking.",
    )
    p.add_argument("--align_inter", type=int, default=0)
    p.add_argument("--min_per_expert", type=int, default=0)
    p.add_argument("--modality_aware", action="store_true")
    p.add_argument("--shared_protect", action="store_true")
    p.add_argument("--text_only", action="store_true")
    p.add_argument("--visual_only", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--expertwise_budget_normalize", action="store_true")
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--smooth_fn", type=str, default="sqrt")
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--tau_skip_path", type=str, default=None)
    p.add_argument("--layer_importance_path", type=str, default=None)
    p.add_argument("--expert_importance_path", type=str, default=None)
    p.add_argument(
        "--ema_source_key",
        type=str,
        default="ema_matrix",
        help="Which tensor key in scores payload to use as EMA source (passed to prepare_scores).",
    )
    return p


def _build_messages_batch(questions, media_type, frame_counts=None):
    """Build chat messages for a batch of questions."""
    messages = []
    for idx, q in enumerate(questions):
        if media_type == "video":
            frame_count = 1 if frame_counts is None else max(1, int(frame_counts[idx]))
            content = [{"type": "image", "image": "placeholder"} for _ in range(frame_count)]
            content.append({"type": "text", "text": q})
        else:
            content = [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": q},
            ]
        messages.append([{"role": "user", "content": content}])
    return messages


<<<<<<< HEAD
def _build_internvl_video_messages_batch(questions, frame_counts):
    """Build InternVL chat messages for video tasks using frames as interleaved images."""
    messages = []
    for q, frame_count in zip(questions, frame_counts):
        content = [{"type": "image", "image": "placeholder"} for _ in range(frame_count)]
        content.append({"type": "text", "text": q})
        messages.append([{"role": "user", "content": content}])
    return messages


def _resize_pil_frame(frame, max_long_side: int = 480):
    width, height = frame.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return frame.convert("RGB")
    scale = max_long_side / long_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return frame.convert("RGB").resize(new_size)


def _sample_frame_directory(frame_dir: str, max_frames: int = 8, max_long_side: int = 480):
    from PIL import Image

    frame_paths = sorted(
        os.path.join(frame_dir, name)
        for name in os.listdir(frame_dir)
        if os.path.isfile(os.path.join(frame_dir, name))
        and os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not frame_paths:
        raise RuntimeError(f"No image frames found in directory: {frame_dir}")

    if len(frame_paths) > max_frames:
        indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int).tolist()
        frame_paths = [frame_paths[idx] for idx in indices]

    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as frame:
            frames.append(_resize_pil_frame(frame, max_long_side=max_long_side))
    return frames


def _sample_video_frames_fallback(video_path: str, max_frames: int = 8, max_long_side: int = 480):
    from PIL import Image
    from tasks.video_mmmu import process_media

    if os.path.isdir(video_path):
        return _sample_frame_directory(video_path, max_frames=max_frames, max_long_side=max_long_side)

    try:
        frames, _ = process_media(video_path, max_frames=max_frames, max_long_side=max_long_side)
        if frames:
            return frames
    except Exception:
        pass

    decoded_frames = list(iio.imiter(video_path, plugin="pyav"))
    if not decoded_frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    if len(decoded_frames) <= max_frames:
        picked = decoded_frames
    else:
        indices = np.linspace(0, len(decoded_frames) - 1, max_frames, dtype=int).tolist()
        picked = [decoded_frames[idx] for idx in indices]

    return [_resize_pil_frame(Image.fromarray(frame_np), max_long_side=max_long_side) for frame_np in picked]


def _coerce_internvl_video_visual(visual):
    if isinstance(visual, tuple):
        visual = visual[0]
    if isinstance(visual, list):
        if visual and isinstance(visual[0], str):
            return _sample_video_frames_fallback(visual[0])
        return list(visual)
    if isinstance(visual, str):
        return _sample_video_frames_fallback(visual)
    return [visual]
=======
def _resolve_text_config(model):
    config = getattr(model, "config", None)
    for candidate in (
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        getattr(getattr(model, "language", None), "config", None),
        config,
    ):
        if candidate is not None:
            return candidate
    raise AttributeError(f"Cannot resolve text/language config for model type {type(model).__name__}.")


def _build_deepseek_conversations(questions, media_type, frame_counts=None):
    conversations = []
    for idx, q in enumerate(questions):
        if media_type == "video":
            frame_count = 1 if frame_counts is None else max(1, int(frame_counts[idx]))
            media_prefix = "<image>\n" * frame_count
        else:
            media_prefix = "<image>\n"
        conversations.append(
            [
                {"role": "user", "content": f"{media_prefix}{q}".strip()},
                {"role": "assistant", "content": ""},
            ]
        )
    return conversations


def _prepare_batch_inputs(processor, *, questions, visuals, media_type, frame_counts=None):
    if hasattr(processor, "apply_chat_template"):
        messages_batch = _build_messages_batch(questions, media_type, frame_counts)
        texts = processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if media_type == "video":
            return processor(
                text=texts,
                images=visuals,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=True,
            )
        return processor(
            images=visuals,
            text=texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
        )

    if hasattr(processor, "process_one") and hasattr(processor, "batchify"):
        conversations_batch = _build_deepseek_conversations(questions, media_type, frame_counts)
        sample_prepares = []
        if media_type == "video":
            cursor = 0
            for conversation, frame_count in zip(conversations_batch, frame_counts):
                sample_images = visuals[cursor: cursor + frame_count]
                cursor += frame_count
                sample_prepares.append(
                    processor(
                        conversations=conversation,
                        images=sample_images or None,
                        force_batchify=False,
                    )
                )
        else:
            for conversation, image in zip(conversations_batch, visuals):
                sample_prepares.append(
                    processor(
                        conversations=conversation,
                        images=[image] if image is not None else None,
                        force_batchify=False,
                    )
                )
        batched = processor.batchify(sample_prepares)
        return {
            "input_ids": batched.input_ids,
            "attention_mask": batched.attention_mask,
            "images": batched.images,
            "images_seq_mask": batched.images_seq_mask,
            "images_spatial_crop": batched.images_spatial_crop,
        }

    raise TypeError(f"Unsupported processor type: {type(processor).__name__}")


def _prepare_video_visual(visual):
    """Match eval/kimi.py behavior: flatten videos into sampled frames."""
    if isinstance(visual, tuple):
        frames, num_frames = visual
        return list(frames), int(num_frames)

    if isinstance(visual, list):
        if len(visual) == 1 and isinstance(visual[0], (str, Path)):
            frames, num_frames = process_video_media(visual[0])
            return list(frames), int(num_frames)
        return list(visual), len(visual)

    if isinstance(visual, (str, Path)):
        frames, num_frames = process_video_media(visual)
        return list(frames), int(num_frames)

    return [visual], 1
>>>>>>> 6a1207d008c8040cb4be17079257016f94d05f59


def _build_generation_kwargs(args):
    generation_kwargs = {}
    enable_tau_skip = bool(args.tau_skip_path)
    if enable_tau_skip:
        import pickle

        with open(args.tau_skip_path, "rb") as f:
            tau_skip_dict = pickle.load(f)
        generation_kwargs["enable_tau_skip"] = True
        generation_kwargs["tau"] = tau_skip_dict["tau"]
    if args.layer_importance_path:
        generation_kwargs["enable_load_layer_importance"] = True
        generation_kwargs["layer_importance_path"] = args.layer_importance_path
    if args.expert_importance_path:
        generation_kwargs["enable_load_expert_importance"] = True
        generation_kwargs["expert_importance_path"] = args.expert_importance_path
    return generation_kwargs


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    model_family = infer_model_family(args.model_path)

    # ── Mask generation ──
    print(f"[Run] Loading scores from: {args.scores_path}")
    masks = None
    if args.prune_ratio is not None and args.prune_ratio > 0:
        mask_result = build_masks_pipeline(
            scores_dir=args.scores_path,
            prune_kwargs={
                "prune_ratio": args.prune_ratio,
                "thresholds_path": args.thresholds_path,
                "mask_method_kwargs": {
                    "inter_layer_method": args.inter_method,
                    "intra_layer_method": args.intra_method,
                    "intra_expert_metric": args.intra_expert_metric,
                    "layerwise_loss_key": args.layerwise_loss_key,
                },
                "adjust_masks_kwargs": {
                    "align_inter": args.align_inter,
                    "min_per_expert": args.min_per_expert,
                },
                "modality_aware": args.modality_aware,
                "shared_protect": args.shared_protect,
                "text_only": args.text_only,
                "visual_only": args.visual_only,
                "use_ema": bool(args.use_ema),
                "normalize": args.normalize,
                "expertwise_budget_normalize": args.expertwise_budget_normalize,
                "ema_source_key": args.ema_source_key,
                "prune_hidden": False,
                "prune_gqa": False,
                "smooth_fn": args.smooth_fn,
            },
            device="cpu",
            verbose=True,
        )
        mask_tensor = mask_result["intermediate_masks"]
        layers = [int(layer) for layer in mask_result.get("layers", list(range(mask_tensor.shape[0])))]
        masks = {
            layer_idx: mask_tensor[pos].detach().cpu().bool()
            for pos, layer_idx in enumerate(layers)
        }
        k_e = mask_result["K_E_inter"].detach().cpu()
        i_orig = int(mask_tensor.shape[-1])
        print(
            f"[Run] Generated masks for {len(layers)} layers. "
            f"I_orig={i_orig}, I_prime min={int(k_e.min().item())} "
            f"max={int(k_e.max().item())} mean={float(k_e.float().mean().item()):.1f}"
        )

    # ── Model loading (auto-dispatch) ──
    print(f"[Run] Loading model from: {args.model_path}")
    model, processor = auto_load_model(args.model_path)
    model.eval()
    text_config = _resolve_text_config(model)
    apply_structural_pruning(model, masks, text_config)
    print("[Run] Applied structural pruning in memory; no checkpoint will be saved.")

    device = next(model.parameters()).device
    print(f"[Run] Model ready. Primary device: {device}")

    # ── Dataset loading (auto-dispatch) ──
    print(f"[Run] Loading task: {args.task}")
    pool, task = load_eval_task(
        args.task,
        start_idx=args.start_idx,
        num_samples=args.num_samples,
        subset_seed=args.subset_seed,
    )
    total = len(pool)
    effective_batch_size = args.batch_size
    if model_family == "internvl" and task.media_type == "video":
        effective_batch_size = 1
        if args.batch_size != 1:
            print(
                "[Run] InternVL video evaluation uses frame-as-image inputs; "
                "forcing batch_size=1 for processor compatibility."
            )
    print(f"[Run] Evaluating {total} samples (batch_size={effective_batch_size}).")
    generation_extra_kwargs = _build_generation_kwargs(args)

    # ── Eval loop ──
    predictions = []
    skipped_missing_video = 0
    with torch.no_grad():
        for i in tqdm(range(0, total, effective_batch_size), desc=f"Prune+Eval {args.task}", unit="batch"):
            batch_rows = _get_batch_rows(pool, i, effective_batch_size)
            visuals = []
            frame_counts = []
            questions = []
            gt_answers = []
            kept_rows = []
            for row in batch_rows:
<<<<<<< HEAD
                try:
                    visual = task.doc_to_visual(row)
                except FileNotFoundError as exc:
                    if args.task == "videomme":
                        skipped_missing_video += 1
                        video_id = row.get("videoID", "<unknown>")
                        print(f"[Run] Skip videomme sample with missing video: {video_id} ({exc})")
                        continue
                    raise
                if model_family == "internvl" and task.media_type == "video":
                    visuals.append(_coerce_internvl_video_visual(visual))
                elif isinstance(visual, tuple):
                    # video_mmmu returns (frames_list, num_frames)
                    visuals.append(visual[0] if task.media_type == "video" else visual[0])
=======
                visual = task.doc_to_visual(row)
                if task.media_type == "video":
                    frames, num_frames = _prepare_video_visual(visual)
                    visuals.extend(frames)
                    frame_counts.append(num_frames)
>>>>>>> 6a1207d008c8040cb4be17079257016f94d05f59
                elif isinstance(visual, list):
                    visuals.append(visual[0])
                else:
                    visuals.append(visual)
                questions.append(task.doc_to_text(row))
                gt_answers.append(task.doc_to_answer(row))
                kept_rows.append(row)

<<<<<<< HEAD
            if not kept_rows:
                continue

            if model_family == "internvl" and task.media_type == "video":
                messages_batch = _build_internvl_video_messages_batch(
                    questions,
                    [len(frames) for frames in visuals],
                )
                texts = [
                    processor.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for message in messages_batch
                ]
                flat_images = [frame for frames in visuals for frame in frames]
                inputs = processor(
                    images=flat_images,
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    truncation=True,
                )
            else:
                messages_batch = _build_messages_batch(questions, task.media_type)
                texts = [
                    processor.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for message in messages_batch
                ]
                if task.media_type == "video":
                    inputs = processor(
                        videos=visuals,
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        truncation=True,
                    )
                else:
                    inputs = processor(
                        images=visuals,
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        truncation=True,
                    )

            inputs = move_to_device(inputs, device)
=======
            inputs = _prepare_batch_inputs(
                processor,
                questions=questions,
                visuals=visuals,
                media_type=task.media_type,
                frame_counts=frame_counts,
            )

            inputs = move_to_device(inputs, device, dtype=next(model.parameters()).dtype)
>>>>>>> 6a1207d008c8040cb4be17079257016f94d05f59
            input_len = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                **generation_extra_kwargs,
            )

            for j, (gt, out_ids) in enumerate(zip(gt_answers, outputs)):
                pred_ids = out_ids[input_len:]
                pred = processor.decode(pred_ids, skip_special_tokens=True)
                record = {
                    "gt": gt,
                    "pred": pred,
                    "correct": _normalize_answer(pred) == _normalize_answer(gt),
                }
                # Preserve question field if available
                if "question" in kept_rows[j]:
                    record["question"] = kept_rows[j]["question"]
                # Extra fields for task-specific eval (e.g. question_type for VideoMMMU)
                for field in task.extra_fields:
                    if field in kept_rows[j]:
                        record[field] = kept_rows[j][field]
                predictions.append(record)

    # ── Evaluate with task-specific metric ──
    eval_result = task.evaluate(predictions)
    metric_name = eval_result["metric_name"]
    metric_value = eval_result["metric_value"]
    detail = eval_result.get("detail", "")
    print(f"\n[Run] {metric_name}: {metric_value:.4f}  ({detail})")
    if skipped_missing_video:
        print(f"[Run] Skipped {skipped_missing_video} videomme sample(s) with missing video files.")

    summary = {
        "model": args.model_path,
        "task": args.task,
        "dataset": args.task,
        "num_samples": total,
        "num_evaluated": len(predictions),
        "num_skipped_missing_video": skipped_missing_video,
        metric_name.lower(): round(metric_value, 6),
        "scores_path": args.scores_path,
        "prune_ratio": args.prune_ratio,
        "tau_skip_path": args.tau_skip_path,
        "layer_importance_path": args.layer_importance_path,
        "expert_importance_path": args.expert_importance_path,
        "inter_method": args.inter_method,
        "intra_method": args.intra_method,
        "intra_expert_metric": args.intra_expert_metric,
        "modality_aware": args.modality_aware,
        "shared_protect": args.shared_protect,
        "text_only": args.text_only,
        "visual_only": args.visual_only,
        "use_ema": bool(args.use_ema),
        "saved_pruned_checkpoint": False,
    }
    # Include extra eval fields (correct, total, etc.)
    for k, v in eval_result.items():
        if k not in ("metric_name", "metric_value", "detail"):
            summary[k] = v

    if args.output_dir:
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Run] Summary saved: {summary_path}")

        preds_path = os.path.join(args.output_dir, "predictions.json")
        with open(preds_path, "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"[Run] Predictions saved: {preds_path}")


if __name__ == "__main__":
    main()
