"""Unified dataset loading and evaluation for prune-and-eval tasks."""

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TaskHelpers:
    """Task-specific helpers for the eval loop."""
    task_name: str
    doc_to_visual: Callable[[dict], Any]
    doc_to_text: Callable[[dict], str]
    doc_to_answer: Callable[[dict], str]
    evaluate: Callable[[List[dict]], dict]
    # "image" or "video" — controls how visuals are fed to the processor
    media_type: str = "image"
    # extra fields needed per-row (e.g. "question_type" for VideoMMMU)
    extra_fields: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GQA
# ---------------------------------------------------------------------------

def _build_gqa_helpers() -> TaskHelpers:
    from tasks.gqa import (
        gqa_doc_to_answer,
        gqa_doc_to_text,
        gqa_doc_to_visual,
    )

    def evaluate(predictions: List[dict]) -> dict:
        correct = sum(1 for p in predictions if p["correct"])
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        return {
            "metric_name": "Accuracy",
            "metric_value": accuracy,
            "correct": correct,
            "total": total,
            "detail": f"{correct}/{total}",
        }

    return TaskHelpers(
        task_name="gqa",
        doc_to_visual=gqa_doc_to_visual,
        doc_to_text=gqa_doc_to_text,
        doc_to_answer=gqa_doc_to_answer,
        evaluate=evaluate,
        media_type="image",
    )


def _load_gqa_rows() -> List[dict]:
    from tasks.gqa import load_gqa_instruction_rows
    return load_gqa_instruction_rows()


# ---------------------------------------------------------------------------
# COCO Captioning
# ---------------------------------------------------------------------------

def _build_coco_helpers() -> TaskHelpers:
    from tasks.coco import coco_doc_to_answer, coco_doc_to_text, coco_doc_to_visual

    def evaluate(predictions: List[dict]) -> dict:
        try:
            return _coco_cider_eval(predictions)
        except Exception as e:
            print(f"[Eval] CIDEr eval failed ({e}), falling back to saving predictions only.")
            return {
                "metric_name": "CIDEr",
                "metric_value": 0.0,
                "detail": "eval_failed",
            }

    return TaskHelpers(
        task_name="coco",
        doc_to_visual=coco_doc_to_visual,
        doc_to_text=coco_doc_to_text,
        doc_to_answer=coco_doc_to_answer,
        evaluate=evaluate,
        media_type="image",
    )


def _coco_cider_eval(predictions: List[dict]) -> dict:
    from pycocoevalcap.cider.cider import Cider

    gts = {}
    res = {}
    for i, p in enumerate(predictions):
        gt = p["gt"]
        gts[i] = [gt] if isinstance(gt, str) else gt
        res[i] = [p["pred"]]

    cider = Cider()
    score, _ = cider.compute_score(gts, res)
    return {
        "metric_name": "CIDEr",
        "metric_value": round(score, 6),
        "detail": f"CIDEr={score:.4f}",
    }


def _load_coco_rows() -> List[dict]:
    from datasets import load_dataset
    from tasks.dataset_paths import require_dataset_dir

    data = load_dataset(
        require_dataset_dir("COCO-Caption2017", "data"), token=True
    )["validation"]
    rows = []
    for row in data:
        rows.append({
            "imageId": row["id"],
            "answer": row["answer"],
        })
    return rows


# ---------------------------------------------------------------------------
# VideoMMMU
# ---------------------------------------------------------------------------

def _build_videommmu_helpers() -> TaskHelpers:
    from tasks.video_mmmu import (
        videommmu_doc_to_answer,
        videommmu_doc_to_text_adaptation,
        videommmu_doc_to_text_perception_comprehension,
        videommmu_doc_to_visual,
    )

    def doc_to_text(doc: dict) -> str:
        qt = doc.get("question_type", "")
        if qt.endswith("Adaptation") or qt.endswith("Analysis"):
            return videommmu_doc_to_text_adaptation(doc)
        return videommmu_doc_to_text_perception_comprehension(doc)

    def evaluate(predictions: List[dict]) -> dict:
        correct = 0
        total = len(predictions)
        for p in predictions:
            gt = p["gt"].strip()
            pred_raw = p["pred"].strip()
            qt = p.get("question_type", "")
            if qt == "multiple-choice":
                pred_letter = _extract_mc_answer(pred_raw)
                is_correct = pred_letter == gt
            else:
                is_correct = _normalize_answer(pred_raw) == _normalize_answer(gt)
            p["correct"] = is_correct
            correct += int(is_correct)
        accuracy = correct / total if total > 0 else 0.0
        return {
            "metric_name": "Accuracy",
            "metric_value": accuracy,
            "correct": correct,
            "total": total,
            "detail": f"{correct}/{total}",
        }

    return TaskHelpers(
        task_name="video_mmmu",
        doc_to_visual=videommmu_doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=videommmu_doc_to_answer,
        evaluate=evaluate,
        media_type="video",
        extra_fields=["question_type", "options"],
    )


def _load_videommmu_rows() -> List[dict]:
    from datasets import concatenate_datasets, load_dataset
    from tasks.dataset_paths import require_dataset_dir

    adaptation = load_dataset(
        require_dataset_dir("VideoMMMU", "Adaptation"), token=True
    )["test"]
    comprehension = load_dataset(
        require_dataset_dir("VideoMMMU", "Comprehension"), token=True
    )["test"]
    perception = load_dataset(
        require_dataset_dir("VideoMMMU", "Perception"), token=True
    )["test"]
    combined = concatenate_datasets([adaptation, comprehension, perception])
    rows = []
    for row in combined:
        rows.append({
            "id": row["id"],
            "question": row["question"],
            "question_type": row["question_type"],
            "options": row["options"],
            "answer": row["answer"],
            "image": row.get("image"),
        })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    return s.strip().lower()


def _extract_mc_answer(pred: str) -> str:
    """Extract the first option letter (A-Z) from a model prediction."""
    pred = pred.strip()
    if pred and pred[0].isalpha() and pred[0].isupper():
        return pred[0]
    match = re.search(r'\b([A-Z])\b', pred)
    if match:
        return match.group(1)
    return pred.strip()[:1].upper() if pred else ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY = {
    "gqa": (_load_gqa_rows, _build_gqa_helpers),
    "coco": (_load_coco_rows, _build_coco_helpers),
    "video_mmmu": (_load_videommmu_rows, _build_videommmu_helpers),
}

# Aliases
_TASK_REGISTRY["coco2017cap"] = _TASK_REGISTRY["coco"]
_TASK_REGISTRY["vmmmu"] = _TASK_REGISTRY["video_mmmu"]
_TASK_REGISTRY["videommmu"] = _TASK_REGISTRY["video_mmmu"]


def load_eval_task(
    task_name: str,
    start_idx: int = 0,
    num_samples: int = 0,
    subset_seed: Optional[int] = None,
) -> tuple:
    """Load a task's data and helpers.

    Returns:
        (pool, task_helpers) where pool is a list of row dicts and
        task_helpers is a TaskHelpers instance.
    """
    task_name = task_name.strip().lower()
    if task_name not in _TASK_REGISTRY:
        available = ", ".join(sorted(set(k for k in _TASK_REGISTRY)))
        raise ValueError(f"Unknown task: {task_name!r}. Available: {available}")

    load_fn, build_helpers_fn = _TASK_REGISTRY[task_name]
    rows = load_fn()
    pool = rows[start_idx:]
    if num_samples > 0:
        if subset_seed is not None:
            rng = random.Random(subset_seed)
            pool = rng.sample(pool, min(num_samples, len(pool)))
        else:
            pool = pool[:num_samples]

    return pool, build_helpers_fn()
