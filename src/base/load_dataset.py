"""Unified dataset loading and evaluation for prune-and-eval tasks."""

import random
import re
import statistics
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


# ═══════════════════════════════════════════════════════════════════════════════
# Shared evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def _accuracy_evaluate(predictions: List[dict]) -> dict:
    correct = sum(1 for p in predictions if p.get("correct", False))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    return {
        "metric_name": "Accuracy",
        "metric_value": accuracy,
        "correct": correct,
        "total": total,
        "detail": f"{correct}/{total}",
    }


def _mc_accuracy_evaluate(predictions: List[dict]) -> dict:
    """Re-score with MC letter extraction, then compute accuracy."""
    correct = 0
    total = len(predictions)
    for p in predictions:
        pred_letter = _extract_mc_answer(p["pred"])
        gt = p["gt"].strip()
        is_correct = pred_letter == gt
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


def _load_hf_dataset_rows(dataset_path: str, split: str = "test", columns: list = None) -> List[dict]:
    """Generic HF dataset loader."""
    from datasets import load_dataset
    data = load_dataset(dataset_path, split=split, token=True)
    if columns:
        return [{c: row[c] for c in columns if c in row} for row in data]
    return [dict(row) for row in data]


# ═══════════════════════════════════════════════════════════════════════════════
# GQA
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gqa_helpers() -> TaskHelpers:
    from tasks.gqa import gqa_doc_to_answer, gqa_doc_to_text, gqa_doc_to_visual
    return TaskHelpers(
        task_name="gqa",
        doc_to_visual=gqa_doc_to_visual,
        doc_to_text=gqa_doc_to_text,
        doc_to_answer=gqa_doc_to_answer,
        evaluate=_accuracy_evaluate,
        media_type="image",
    )

def _load_gqa_rows() -> List[dict]:
    from tasks.gqa import load_gqa_instruction_rows
    return load_gqa_instruction_rows()


# ═══════════════════════════════════════════════════════════════════════════════
# COCO Captioning
# ═══════════════════════════════════════════════════════════════════════════════

def _build_coco_helpers() -> TaskHelpers:
    from tasks.coco import coco_doc_to_answer, coco_doc_to_text, coco_doc_to_visual
    def evaluate(predictions: List[dict]) -> dict:
        try:
            return _coco_cider_eval(predictions)
        except Exception as e:
            print(f"[Eval] CIDEr eval failed ({e}), falling back to saving predictions only.")
            return {"metric_name": "CIDEr", "metric_value": 0.0, "detail": "eval_failed"}
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
    gts, res = {}, {}
    for i, p in enumerate(predictions):
        gt = p["gt"]
        gts[i] = [gt] if isinstance(gt, str) else gt
        res[i] = [p["pred"]]
    score, _ = Cider().compute_score(gts, res)
    return {"metric_name": "CIDEr", "metric_value": round(score, 6), "detail": f"CIDEr={score:.4f}"}

def _load_coco_rows() -> List[dict]:
    from datasets import load_dataset
    from tasks.dataset_paths import require_dataset_dir
    data = load_dataset(require_dataset_dir("COCO-Caption2017", "data"), token=True)["validation"]
    return [{"imageId": row["id"], "answer": row["answer"]} for row in data]


# ═══════════════════════════════════════════════════════════════════════════════
# TextVQA
# ═══════════════════════════════════════════════════════════════════════════════

def _build_textvqa_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def doc_to_text(doc):
        q = doc["question"].capitalize()
        return f"{q}\nAnswer the question using a single word or phrase."

    def doc_to_answer(doc):
        answers = doc.get("answers", [])
        if answers:
            return answers[0]
        return doc.get("answer", "")

    def evaluate(predictions: List[dict]) -> dict:
        try:
            from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
            proc = EvalAIAnswerProcessor()
            total_acc = 0.0
            for p in predictions:
                pred = proc(p["pred"])
                gt_answers = p.get("answers") or [p["gt"]]
                gt_answers = [proc(a) for a in gt_answers]
                gt_accs = []
                for i, gt in enumerate(gt_answers):
                    others = [gt_answers[j] for j in range(len(gt_answers)) if j != i]
                    matching = sum(1 for o in others if o == pred)
                    gt_accs.append(min(1.0, matching / 3.0))
                acc = statistics.mean(gt_accs) if gt_accs else 0.0
                p["correct"] = acc > 0
                total_acc += acc
            accuracy = total_acc / len(predictions) if predictions else 0.0
            return {
                "metric_name": "Accuracy",
                "metric_value": round(accuracy, 6),
                "detail": f"vqa_acc={accuracy:.4f}",
            }
        except ImportError:
            return _accuracy_evaluate(predictions)

    return TaskHelpers(
        task_name="textvqa",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=evaluate,
        media_type="image",
        extra_fields=["answers"],
    )

def _load_textvqa_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/textvqa", split="validation")


# ═══════════════════════════════════════════════════════════════════════════════
# ChartQA
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chartqa_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def doc_to_text(doc):
        return f"{doc['question']}\nAnswer the question with a single word."

    def doc_to_answer(doc):
        return str(doc["answer"])

    def evaluate(predictions: List[dict]) -> dict:
        correct = 0
        for p in predictions:
            p["correct"] = _relaxed_correctness(p["pred"], p["gt"])
            correct += int(p["correct"])
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
        task_name="chartqa",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=evaluate,
        media_type="image",
    )

def _relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> bool:
    def _to_float(text):
        try:
            return float(text.rstrip("%")) / 100.0 if text.endswith("%") else float(text)
        except ValueError:
            return None
    pf, tf = _to_float(prediction.strip()), _to_float(target.strip())
    if pf is not None and tf:
        return abs(pf - tf) / abs(tf) <= max_relative_change
    return prediction.strip().lower() == target.strip().lower()

def _load_chartqa_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/ChartQA", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# MMStar
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mmstar_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def doc_to_text(doc):
        return f"{doc['question']}\nAnswer with the option's letter from the given choices directly."

    def doc_to_answer(doc):
        return doc["answer"]

    return TaskHelpers(
        task_name="mmstar",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="image",
    )

def _load_mmstar_rows() -> List[dict]:
    return _load_hf_dataset_rows("Lin-Chen/MMStar", split="val")


# ═══════════════════════════════════════════════════════════════════════════════
# MMBench
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mmbench_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def doc_to_text(doc):
        question = doc["question"]
        options = []
        for key in ["A", "B", "C", "D"]:
            val = doc.get(key, "")
            if val:
                options.append(f"{key}. {val}")
        opts_str = "\n".join(options)
        hint = doc.get("hint", "")
        prefix = f"{hint}\n" if hint else ""
        return f"{prefix}{question}\n{opts_str}\nAnswer with the option's letter from the given choices directly."

    def doc_to_answer(doc):
        return doc["answer"]

    return TaskHelpers(
        task_name="mmbench",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="image",
    )

def _load_mmbench_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/MMBench", split="dev")


# ═══════════════════════════════════════════════════════════════════════════════
# MMVet
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mmvet_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def doc_to_text(doc):
        return doc["question"]

    def doc_to_answer(doc):
        return doc["answer"]

    return TaskHelpers(
        task_name="mmvet",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_accuracy_evaluate,
        media_type="image",
    )

def _load_mmvet_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/MMVet", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# MME
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mme_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def doc_to_text(doc):
        return doc["question"].strip()

    def doc_to_answer(doc):
        return doc["answer"].strip().lower()

    def evaluate(predictions: List[dict]) -> dict:
        correct = 0
        for p in predictions:
            pred = _parse_yes_no(p["pred"])
            gt = p["gt"].strip().lower().replace(".", "")
            p["correct"] = pred == gt
            correct += int(p["correct"])
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
        task_name="mme",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=evaluate,
        media_type="image",
    )

def _parse_yes_no(pred: str) -> str:
    pred = pred.lower().strip().replace(".", "")
    if pred in ("yes", "no"):
        return pred
    if len(pred) == 1:
        return {"y": "yes", "n": "no"}.get(pred, "other")
    prefix = pred[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"

def _load_mme_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/MME", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# RealWorldQA
# ═══════════════════════════════════════════════════════════════════════════════

def _build_realworldqa_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def doc_to_text(doc):
        return f"{doc['question']}\nAnswer with the option's letter from the given choices directly."

    def doc_to_answer(doc):
        return doc["answer"]

    return TaskHelpers(
        task_name="realworldqa",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="image",
    )

def _load_realworldqa_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/RealWorldQA", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# VideoMMMU
# ═══════════════════════════════════════════════════════════════════════════════

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
                is_correct = _extract_mc_answer(pred_raw) == gt
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
    adaptation = load_dataset(require_dataset_dir("VideoMMMU", "Adaptation"), token=True)["test"]
    comprehension = load_dataset(require_dataset_dir("VideoMMMU", "Comprehension"), token=True)["test"]
    perception = load_dataset(require_dataset_dir("VideoMMMU", "Perception"), token=True)["test"]
    combined = concatenate_datasets([adaptation, comprehension, perception])
    return [{
        "id": row["id"], "question": row["question"],
        "question_type": row["question_type"], "options": row["options"],
        "answer": row["answer"], "image": row.get("image"),
    } for row in combined]


# ═══════════════════════════════════════════════════════════════════════════════
# MVBench (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mvbench_helpers() -> TaskHelpers:
    from lmms_eval.tasks.mvbench.utils import mvbench_doc_to_visual, mvbench_doc_to_text

    def doc_to_visual(doc):
        return mvbench_doc_to_visual(doc)

    def doc_to_text(doc):
        return mvbench_doc_to_text(doc, lmms_eval_specific_kwargs={
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        })

    def doc_to_answer(doc):
        return doc["answer"]

    return TaskHelpers(
        task_name="mvbench",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="video",
    )

def _load_mvbench_rows() -> List[dict]:
    return _load_hf_dataset_rows("OpenGVLab/MVBench", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# EgoSchema (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_egoschema_helpers() -> TaskHelpers:
    from lmms_eval.tasks.egoschema.utils import egoschema_doc_to_visual, egoschema_doc_to_text

    def doc_to_visual(doc):
        return egoschema_doc_to_visual(doc)

    def doc_to_text(doc):
        return egoschema_doc_to_text(doc, lmms_eval_specific_kwargs={
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        })

    def doc_to_answer(doc):
        return str(doc["answer"])

    return TaskHelpers(
        task_name="egoschema",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="video",
    )

def _load_egoschema_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/egoschema", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# VideoMME (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_videomme_helpers() -> TaskHelpers:
    from lmms_eval.tasks.videomme.utils import videomme_doc_to_visual

    def doc_to_visual(doc):
        return videomme_doc_to_visual(doc)

    def doc_to_text(doc):
        question = doc["question"]
        options = doc.get("options", [])
        letters = [chr(ord("A") + i) for i in range(len(options))]
        opts = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))
        return f"{question}\n{opts}\nAnswer with the option's letter from the given choices directly."

    def doc_to_answer(doc):
        return doc["answer"]

    return TaskHelpers(
        task_name="videomme",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="video",
    )

def _load_videomme_rows() -> List[dict]:
    return _load_hf_dataset_rows("lmms-lab/Video-MME", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# LongVideoBench (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_longvideobench_helpers() -> TaskHelpers:
    from lmms_eval.tasks.longvideobench.utils import (
        longvideobench_doc_to_visual_v as lvb_doc_to_visual,
        longvideobench_doc_to_text as lvb_doc_to_text,
    )

    def doc_to_visual(doc):
        return lvb_doc_to_visual(doc)

    def doc_to_text(doc):
        return lvb_doc_to_text(doc, lmms_eval_specific_kwargs={
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        })

    def doc_to_answer(doc):
        return doc.get("correct_choice", doc.get("answer", ""))

    return TaskHelpers(
        task_name="longvideobench",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="video",
    )

def _load_longvideobench_rows() -> List[dict]:
    return _load_hf_dataset_rows("longvideobench/LongVideoBench", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

_TASK_REGISTRY = {
    # Image tasks
    "gqa":          (_load_gqa_rows, _build_gqa_helpers),
    "coco":         (_load_coco_rows, _build_coco_helpers),
    "textvqa":      (_load_textvqa_rows, _build_textvqa_helpers),
    "chartqa":      (_load_chartqa_rows, _build_chartqa_helpers),
    "mmstar":       (_load_mmstar_rows, _build_mmstar_helpers),
    "mmbench":      (_load_mmbench_rows, _build_mmbench_helpers),
    "mmvet":        (_load_mmvet_rows, _build_mmvet_helpers),
    "mme":          (_load_mme_rows, _build_mme_helpers),
    "realworldqa":  (_load_realworldqa_rows, _build_realworldqa_helpers),
    # Video tasks
    "video_mmmu":      (_load_videommmu_rows, _build_videommmu_helpers),
    "mvbench":         (_load_mvbench_rows, _build_mvbench_helpers),
    "egoschema":       (_load_egoschema_rows, _build_egoschema_helpers),
    "videomme":        (_load_videomme_rows, _build_videomme_helpers),
    "longvideobench":  (_load_longvideobench_rows, _build_longvideobench_helpers),
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
