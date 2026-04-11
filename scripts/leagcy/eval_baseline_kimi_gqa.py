"""Baseline evaluation of Kimi-VL on GQA testdev_balanced.

No pruning, no token skip — pure model accuracy.

Usage:
    python scripts/eval_baseline_kimi_gqa.py \
        --model_name_or_path moonshotai/Kimi-VL-A3B-Instruct \
        --output_dir results/baseline_kimi_gqa \
        --num_samples 500
"""
import argparse
import json
import os
import random
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import transformers
from tqdm.auto import tqdm

from models.kimi import load_model as load_model_patched
from models.kimi import load_model_plain
from observations.common import resolve_model_name_or_path
from src.modality_router import attach_modality_aware_router, load_affinity
from tasks.gqa import (
    gqa_doc_to_answer,
    gqa_doc_to_text,
    gqa_doc_to_visual,
    load_gqa_instruction_rows,
)


def normalize_answer(s: str) -> str:
    return s.strip().lower()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline eval: Kimi-VL on GQA.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=0,
                   help="Number of samples to evaluate (0 = full testdev_balanced).")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=None,
                   help="If set, randomly sample num_samples rows from [start_idx, end) instead of sequential slice.")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--affinity_path",
        type=str,
        default=None,
        help="Path to affinity.pt saved by collect_scores.py --modality_aware. "
             "If provided, enables modality-aware routing at inference time.",
    )
    p.add_argument(
        "--affinity_threshold",
        type=float,
        default=0.9,
        help="Affinity magnitude cutoff for modality-aware routing. Default: 0.9.",
    )
    return p


def move_to_device(inputs: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Eval] Resolving model: {args.model_name_or_path}")
    print(f"[Eval] transformers: {transformers.__version__} @ {transformers.__file__}")
    model_path = resolve_model_name_or_path(args.model_name_or_path)
    print(f"[Eval] Loading model from: {model_path}")
    threshold_noop = (
        args.affinity_path is not None and args.affinity_threshold >= 1.0
    )

    if args.affinity_path is not None and not threshold_noop:
        # load_model_patched binds gate_forward and model_forward which propagate
        # moe_text_index / moe_media_index to every gate — required for affinity routing.
        print("[Eval] Affinity path provided: loading model with full MoDES patches.")
        model, processor = load_model_patched(
            model_path,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    else:
        model, processor = load_model_plain(
            model_path,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

    model.eval()
    device = next(model.parameters()).device
    print(f"[Eval] Model loaded. Primary device: {device}")

    if args.affinity_path is not None and not threshold_noop:
        print(f"[Eval] Loading affinity from: {args.affinity_path}")
        affinity = load_affinity(args.affinity_path)
        attach_modality_aware_router(model, affinity, threshold=args.affinity_threshold)
    elif threshold_noop:
        print(
            "[Eval] affinity_threshold>=1.0 makes thresholded affinity routing a "
            "strict no-op; skipping affinity attachment so results match the "
            "no-affinity path."
        )
    else:
        print("[Eval] No affinity path provided; using standard routing.")

    print("[Eval] Loading GQA testdev_balanced...")
    rows = load_gqa_instruction_rows()
    pool = rows[args.start_idx:]

    if args.num_samples > 0:
        if args.subset_seed is not None:
            rng = random.Random(args.subset_seed)
            pool = rng.sample(pool, min(args.num_samples, len(pool)))
        else:
            pool = pool[: args.num_samples]

    total = len(pool)
    print(f"[Eval] Evaluating {total} samples (batch_size={args.batch_size}).")

    correct = 0
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, total, args.batch_size), desc="Eval GQA", unit="batch"):
            batch_rows = pool[i : i + args.batch_size]

            images = []
            questions = []
            gt_answers = []
            for row in batch_rows:
                images.append(gqa_doc_to_visual(row)[0])
                questions.append(gqa_doc_to_text(row))
                gt_answers.append(gqa_doc_to_answer(row))

            messages_batch = [
                [{"role": "user", "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": q},
                ]}]
                for q in questions
            ]
            texts = processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            inputs = processor(
                images=images,
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
                is_correct = normalize_answer(pred) == normalize_answer(gt)
                correct += int(is_correct)
                predictions.append({
                    "question": batch_rows[j]["question"],
                    "gt": gt,
                    "pred": pred,
                    "correct": is_correct,
                })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n[Eval] Accuracy: {accuracy:.4f}  ({correct}/{total})")

    summary = {
        "model": args.model_name_or_path,
        "dataset": "gqa_testdev_balanced",
        "num_samples": total,
        "accuracy": round(accuracy, 6),
        "correct": correct,
        "affinity_path": args.affinity_path,
        "affinity_threshold": args.affinity_threshold if args.affinity_path else None,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Eval] Summary saved: {summary_path}")

    preds_path = os.path.join(args.output_dir, "predictions.json")
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Predictions saved: {preds_path}")


if __name__ == "__main__":
    main()
