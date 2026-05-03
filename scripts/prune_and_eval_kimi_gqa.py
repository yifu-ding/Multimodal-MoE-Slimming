"""Prune a VL-MoE model in memory and evaluate on a given task without saving a ckpt."""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from tqdm.auto import tqdm

from src.base.load_dataset import load_eval_task
from src.base.models import auto_load_model
from src.generate_mask import generate_masks as build_masks_pipeline
from src.prune import apply_structural_pruning


def _normalize_answer(s) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def move_to_device(inputs: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


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
    p.add_argument("--align_inter", type=int, default=0)
    p.add_argument("--min_per_expert", type=int, default=0)
    p.add_argument("--modality_aware", action="store_true")
    p.add_argument("--shared_protect", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--expertwise_budget_normalize", action="store_true")
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--smooth_fn", type=str, default="sqrt")
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument(
        "--ema_source_key",
        type=str,
        default="ema_matrix",
        help="Which tensor key in scores payload to use as EMA source (passed to prepare_scores).",
    )
    return p


def _build_messages_batch(questions, media_type):
    """Build chat messages for a batch of questions."""
    messages = []
    for q in questions:
        if media_type == "video":
            content = [
                {"type": "video", "video": "placeholder"},
                {"type": "text", "text": q},
            ]
        else:
            content = [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": q},
            ]
        messages.append([{"role": "user", "content": content}])
    return messages


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

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
                },
                "adjust_masks_kwargs": {
                    "align_inter": args.align_inter,
                    "min_per_expert": args.min_per_expert,
                },
                "modality_aware": args.modality_aware,
                "shared_protect": args.shared_protect,
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
    text_config = model.config.text_config
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
    print(f"[Run] Evaluating {total} samples (batch_size={args.batch_size}).")

    # ── Eval loop ──
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, total, args.batch_size), desc=f"Prune+Eval {args.task}", unit="batch"):
            batch_rows = _get_batch_rows(pool, i, args.batch_size)
            visuals = []
            questions = []
            gt_answers = []
            for row in batch_rows:
                visual = task.doc_to_visual(row)
                if isinstance(visual, tuple):
                    # video_mmmu returns (frames_list, num_frames)
                    visuals.append(visual[0] if task.media_type == "video" else visual[0])
                elif isinstance(visual, list):
                    visuals.append(visual[0])
                else:
                    visuals.append(visual)
                questions.append(task.doc_to_text(row))
                gt_answers.append(task.doc_to_answer(row))

            messages_batch = _build_messages_batch(questions, task.media_type)
            texts = processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                return_tensors="pt",
            )

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
            input_len = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
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
                if "question" in batch_rows[j]:
                    record["question"] = batch_rows[j]["question"]
                # Extra fields for task-specific eval (e.g. question_type for VideoMMMU)
                for field in task.extra_fields:
                    if field in batch_rows[j]:
                        record[field] = batch_rows[j][field]
                predictions.append(record)

    # ── Evaluate with task-specific metric ──
    eval_result = task.evaluate(predictions)
    metric_name = eval_result["metric_name"]
    metric_value = eval_result["metric_value"]
    detail = eval_result.get("detail", "")
    print(f"\n[Run] {metric_name}: {metric_value:.4f}  ({detail})")

    summary = {
        "model": args.model_path,
        "task": args.task,
        "dataset": args.task,
        "num_samples": total,
        metric_name.lower(): round(metric_value, 6),
        "scores_path": args.scores_path,
        "prune_ratio": args.prune_ratio,
        "inter_method": args.inter_method,
        "intra_method": args.intra_method,
        "intra_expert_metric": args.intra_expert_metric,
        "modality_aware": args.modality_aware,
        "shared_protect": args.shared_protect,
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
