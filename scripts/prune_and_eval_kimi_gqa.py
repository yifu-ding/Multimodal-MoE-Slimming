"""Prune Kimi-VL in memory and evaluate directly on GQA without saving a ckpt."""

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
from tqdm.auto import tqdm

from models.kimi import load_model
from observations.common import resolve_model_name_or_path
from src.generate_mask import generate_masks as build_masks_pipeline
from src.prune import apply_structural_pruning
from tasks.gqa import (
    gqa_doc_to_answer,
    gqa_doc_to_text,
    gqa_doc_to_visual,
    load_gqa_instruction_rows,
)


def normalize_answer(s: str) -> str:
    return s.strip().lower()


def move_to_device(inputs: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prune Kimi-VL in memory and evaluate on GQA without saving a checkpoint."
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--scores_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--prune_ratio", type=float, default=0.30)
    p.add_argument("--inter_method", type=str, default="uniform")
    p.add_argument("--intra_method", type=str, default="uniform")
    p.add_argument("--intra_expert_metric", type=str, default="activation")
    p.add_argument("--align_inter", type=int, default=0)
    p.add_argument("--min_per_expert", type=int, default=0)
    p.add_argument("--modality_aware", action="store_true")
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Run] Loading scores from: {args.scores_path}")
    mask_result = build_masks_pipeline(
        scores_dir=args.scores_path,
        prune_kwargs={
            "prune_ratio": args.prune_ratio,
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
            "prune_hidden": False,
            "prune_gqa": False,
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

    resolved_model_path = resolve_model_name_or_path(args.model_path)
    print(f"[Run] Loading model from: {resolved_model_path}")
    model, processor = load_model(
        resolved_model_path,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    text_config = model.config.text_config
    apply_structural_pruning(model, masks, text_config)
    print("[Run] Applied structural pruning in memory; no checkpoint will be saved.")

    device = next(model.parameters()).device
    print(f"[Run] Model ready. Primary device: {device}")
    print("[Run] Loading GQA testdev_balanced...")
    rows = load_gqa_instruction_rows()
    pool = rows[args.start_idx :]
    if args.num_samples > 0:
        if args.subset_seed is not None:
            rng = random.Random(args.subset_seed)
            pool = rng.sample(pool, min(args.num_samples, len(pool)))
        else:
            pool = pool[: args.num_samples]

    total = len(pool)
    print(f"[Run] Evaluating {total} samples (batch_size={args.batch_size}).")

    correct = 0
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, total, args.batch_size), desc="Prune+Eval GQA", unit="batch"):
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
                predictions.append(
                    {
                        "question": batch_rows[j]["question"],
                        "gt": gt,
                        "pred": pred,
                        "correct": is_correct,
                    }
                )

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n[Run] Accuracy: {accuracy:.4f}  ({correct}/{total})")
    summary = {
        "model": args.model_path,
        "dataset": "gqa_testdev_balanced",
        "num_samples": total,
        "accuracy": round(accuracy, 6),
        "correct": correct,
        "scores_path": args.scores_path,
        "prune_ratio": args.prune_ratio,
        "inter_method": args.inter_method,
        "intra_method": args.intra_method,
        "intra_expert_metric": args.intra_expert_metric,
        "modality_aware": args.modality_aware,
        "saved_pruned_checkpoint": False,
    }
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
