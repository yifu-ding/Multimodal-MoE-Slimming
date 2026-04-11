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
import transformers
from tqdm.auto import tqdm

from models.kimi import load_model
from observations.common import resolve_model_name_or_path
from src.modality_router import attach_modality_aware_router, load_affinity
from src.planners import generate_masks
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
    p.add_argument("--intra_method", type=str, default="expertwise")
    p.add_argument(
        "--layerwise_weight_source",
        type=str,
        default=None,
        choices=["repr_change", "block_loss", None],
    )
    p.add_argument(
        "--expertwise_weight_source",
        type=str,
        default=None,
        choices=["expert_out_contrib", "expert_usage", None],
    )
    p.add_argument("--evict_min_channels", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--affinity_path", type=str, default=None)
    p.add_argument("--affinity_threshold", type=float, default=0.9)
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    threshold_noop = (
        args.affinity_path is not None and args.affinity_threshold >= 1.0
    )

    print(f"[Run] transformers: {transformers.__version__} @ {transformers.__file__}")
    print(f"[Run] Loading scores from: {args.scores_path}")
    score_payload = torch.load(args.scores_path, weights_only=False)
    scores = score_payload["scores"]
    modality_channel_scores = (
        score_payload.get("modality_channel_scores", None)
        if score_payload.get("modality_aware", False)
        else None
    )
    used_modality_channel_budgeting = (
        score_payload.get("modality_aware", False)
        and modality_channel_scores is not None
    )
    layer_to_num_experts = score_payload["layer_to_num_experts"]
    layer_to_num_channels = score_payload["layer_to_num_channels"]

    layerwise_weights = None
    if args.inter_method == "coverage":
        if args.layerwise_weight_source is None:
            raise ValueError("layerwise_weight_source is required for coverage inter-layer")
        sorted_layers = sorted(scores.keys())
        if args.layerwise_weight_source == "repr_change":
            src = score_payload.get("layerwise_repr_change", {})
        else:
            src = score_payload.get("layerwise_loss", {})
        vals = [src.get(l, None) for l in sorted_layers]
        if not all(v is not None for v in vals):
            raise ValueError(
                f"Missing layerwise weights for source={args.layerwise_weight_source}"
            )
        layerwise_weights = torch.tensor(vals, dtype=torch.float32)

    expertwise_weights = None
    if args.intra_method == "coverage" and args.expertwise_weight_source is not None:
        sorted_layers = sorted(scores.keys())
        payload_key = {
            "expert_out_contrib": "attr_coverage",
            "expert_usage": "usage_coverage",
        }[args.expertwise_weight_source]

        precomputed = score_payload.get("expertwise_weights", {})
        src = precomputed.get(payload_key, None)
        if src is not None:
            print(f"[Run] Loading precomputed expertwise_weights[{payload_key}] from scores payload.")
        else:
            if args.expertwise_weight_source == "expert_out_contrib":
                src = score_payload.get(
                    "expert_out_token_contrib",
                    score_payload.get("expert_out_contrib", {}),
                )
            else:
                src = score_payload.get("expert_usage", {})
            print(
                f"[Run] WARNING: expertwise_weights[{payload_key}] not found; "
                "falling back to raw per-expert scores."
            )

        rows = []
        missing = False
        for layer_idx in sorted_layers:
            E = layer_to_num_experts[layer_idx]
            row = []
            layer_src = src.get(layer_idx, {})
            for eid in range(E):
                value = layer_src.get(eid, None)
                if value is None:
                    missing = True
                    value = 0.0
                row.append(float(value))
            rows.append(row)
        expertwise_weights = torch.tensor(rows, dtype=torch.float32)
        if missing:
            print(
                f"[Run] WARNING: {args.expertwise_weight_source} not fully available; "
                "missing experts were filled with 0."
            )

    print(
        f"[Run] Generating masks: prune_ratio={args.prune_ratio}, "
        f"inter={args.inter_method}, intra={args.intra_method}"
    )
    masks = generate_masks(
        scores,
        args.prune_ratio,
        layer_to_num_experts,
        layer_to_num_channels,
        inter_method=args.inter_method,
        intra_method=args.intra_method,
        layerwise_weights=layerwise_weights,
        expertwise_weights=expertwise_weights,
        modality_channel_scores=modality_channel_scores,
        evict_min_channels=args.evict_min_channels,
    )
    first_layer = sorted(masks.keys())[0]
    i_orig = layer_to_num_channels[first_layer]
    all_k = [
        int(masks[l][e].sum())
        for l in sorted(masks.keys())
        for e in range(masks[l].shape[0])
    ]
    print(
        f"[Run] I_orig={i_orig}, I_prime: min={min(all_k)} max={max(all_k)} "
        f"mean={sum(all_k)/len(all_k):.1f}"
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

    if args.affinity_path is not None and not threshold_noop:
        print(f"[Run] Loading affinity from: {args.affinity_path}")
        affinity = load_affinity(args.affinity_path)
        attach_modality_aware_router(model, affinity, threshold=args.affinity_threshold)
    elif threshold_noop:
        print(
            "[Run] affinity_threshold>=1.0 makes thresholded affinity routing a "
            "strict no-op; skipping affinity attachment so results match the "
            "no-affinity path."
        )
    else:
        print("[Run] No affinity path provided; using standard routing.")

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
        "layerwise_weight_source": args.layerwise_weight_source,
        "expertwise_weight_source": args.expertwise_weight_source,
        "used_modality_channel_budgeting": used_modality_channel_budgeting,
        "evict_min_channels": args.evict_min_channels,
        "affinity_path": args.affinity_path,
        "affinity_threshold": args.affinity_threshold if args.affinity_path else None,
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
