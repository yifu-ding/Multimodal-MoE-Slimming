"""Unified dataset loading and evaluation for prune-and-eval tasks."""

import csv
import json
import os
import random
import re
import statistics
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pyarrow.parquet as pq


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


def _load_hf_dataset(dataset_path: str, config_name: Optional[str] = None, split: str = "test"):
    """Generic HF dataset loader. Returns the Dataset object directly (lazy image decoding)."""
    from datasets import load_dataset
    if config_name is None:
        return load_dataset(dataset_path, split=split, token=True)
    return load_dataset(dataset_path, config_name, split=split, token=True)


def _hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _tmp_media_root() -> str:
    root = os.path.join("/tmp", "modes_media_cache")
    os.makedirs(root, exist_ok=True)
    return root


def _mvbench_roots() -> List[str]:
    roots: List[str] = []
    env_root = os.environ.get("MVBENCH_ROOT", "").strip()
    if env_root:
        roots.append(env_root)

    roots.extend(
        [
            "/home/data2/dyf/MVBench",
            os.path.join(_hf_home(), "datasets", "MVBench"),
            os.path.join(
                _hf_home(),
                "hub",
                "datasets--OpenGVLab--MVBench",
                "snapshots",
                "230a2d4fac8900333c61754641c7a13e069ac9c6",
            ),
        ]
    )

    deduped: List[str] = []
    for root in roots:
        if root and root not in deduped:
            deduped.append(root)
    return deduped


def _videomme_roots() -> List[str]:
    roots: List[str] = []
    env_root = os.environ.get("VIDEO_MME_ROOT", "").strip()
    if env_root:
        roots.append(env_root)

    roots.extend(
        [
            "/home/data2/dyf/Video-MME",
            os.path.join(_hf_home(), "datasets", "Video-MME_8frame"),
            os.path.join(_hf_home(), "datasets", "Video-MME"),
        ]
    )

    deduped: List[str] = []
    for root in roots:
        if root and root not in deduped:
            deduped.append(root)
    return deduped


def _videomme_archive_roots() -> List[str]:
    roots = []

    env_root = os.environ.get("VIDEO_MME_ROOT", "").strip()
    if env_root:
        roots.append(env_root)

    roots.append(os.path.join(_hf_home(), "datasets", "Video-MME"))

    snapshot_root = os.path.join(
        _hf_home(),
        "hub",
        "datasets--lmms-lab--Video-MME",
        "snapshots",
    )
    if os.path.isdir(snapshot_root):
        for child in sorted(Path(snapshot_root).iterdir()):
            if child.is_dir():
                roots.append(str(child))

    deduped: List[str] = []
    for root in roots:
        if root and root not in deduped:
            deduped.append(root)
    return deduped


_MVBENCH_DATA_FOLDERS = {
    "object_interaction": ["star/Charades_segment", "star/Charades_v1_480", "data0613/star/Charades_v1_480"],
    "action_sequence": ["star/Charades_segment", "star/Charades_v1_480", "data0613/star/Charades_v1_480"],
    "action_prediction": ["star/Charades_segment", "star/Charades_v1_480", "data0613/star/Charades_v1_480"],
    "action_localization": ["sta/sta_video_segment", "sta/sta_video"],
    "moving_count": ["clevrer/video_validation"],
    "fine_grained_pose": ["nturgbd_convert"],
    "character_order": ["perception/videos"],
    "object_shuffle": ["perception/videos"],
    "egocentric_navigation": ["vlnqa"],
    "moving_direction": ["clevrer/video_validation"],
    "episodic_reasoning": ["tvqa/video_fps3_hq_segment", "tvqa/frames_fps3_hq"],
    "fine_grained_action": ["Moments_in_Time_Raw/videos", "Moments_in_Time_Raw/videos/validation"],
    "scene_transition": ["scene_qa/video"],
    "state_change": ["perception/videos"],
    "moving_attribute": ["clevrer/video_validation"],
    "action_antonym": ["ssv2_video_mp4", "ssv2_video"],
    "unexpected_action": ["FunQA_test/test"],
    "counterfactual_inference": ["clevrer/video_validation"],
    "object_existence": ["clevrer/video_validation"],
    "action_count": ["perception/videos"],
}


def _load_dataset_from_parquet(parquet_path: str):
    from datasets import Dataset

    table = pq.read_table(parquet_path)
    return Dataset(table)


def _load_dataset_from_arrow(arrow_path: str):
    from datasets import Dataset

    return Dataset.from_file(arrow_path)


def _extract_zip_member(archive_paths: List[str], member_name: str, target_root: str) -> Optional[str]:
    target_path = os.path.join(target_root, member_name)
    if os.path.exists(target_path):
        return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    for archive_path in archive_paths:
        if not os.path.exists(archive_path):
            continue
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            if member_name in names:
                zf.extract(member_name, path=target_root)
                return target_path

            basename = os.path.basename(member_name)
            basename_hits = [name for name in names if os.path.basename(name) == basename]
            if len(basename_hits) == 1:
                extracted_name = basename_hits[0]
                zf.extract(extracted_name, path=target_root)
                return os.path.join(target_root, extracted_name)
    return None


def _find_existing_media_path(candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _parse_json_list(raw_value: Any) -> List[Any]:
    if isinstance(raw_value, list):
        return raw_value
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return json.loads(text)


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^\s*[A-Z][\.\):]\s*", "", option).strip()


def _option_letter_from_answer(answer: str, options: List[str]) -> str:
    answer = answer.strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()

    normalized_answer = _normalize_answer(answer)
    for idx, option in enumerate(options):
        if _normalize_answer(option) == normalized_answer:
            return chr(ord("A") + idx)
    return answer


def _load_tsv_rows(tsv_path: str) -> List[dict]:
    with open(tsv_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _resolve_egoschema_video_path(video_idx: str) -> str:
    dataset_root = os.path.join(_hf_home(), "datasets", "egoschema")
    extracted = _find_existing_media_path(
        [
            os.path.join(dataset_root, "videos", f"{video_idx}.mp4"),
            os.path.join(dataset_root, "videos", f"{video_idx}.MP4"),
        ]
    )
    if extracted:
        return extracted

    archive_paths = sorted(str(p) for p in Path(dataset_root).glob("videos_chunked_*.zip"))
    for suffix in ("mp4", "MP4"):
        member = f"videos/{video_idx}.{suffix}"
        extracted = _extract_zip_member(archive_paths, member, os.path.join(_tmp_media_root(), "egoschema"))
        if extracted:
            return extracted
    raise FileNotFoundError(f"EgoSchema video not found for {video_idx}")


def _resolve_videomme_video_path(video_id: str) -> str:
    checked_roots: List[str] = []
    for root in _videomme_roots():
        checked_roots.append(root)
        extracted = _find_existing_media_path(
            [
                os.path.join(root, "video", f"{video_id}.mp4"),
                os.path.join(root, "video", f"{video_id}.MP4"),
                os.path.join(root, "video", f"{video_id}.mkv"),
                os.path.join(root, "data", f"{video_id}.mp4"),
                os.path.join(root, "data", f"{video_id}.MP4"),
                os.path.join(root, "data", f"{video_id}.mkv"),
                os.path.join(root, f"{video_id}.mp4"),
                os.path.join(root, f"{video_id}.MP4"),
                os.path.join(root, f"{video_id}.mkv"),
            ]
        )
        if extracted:
            return extracted

    archive_roots = _videomme_archive_roots()
    archive_paths: List[str] = []
    for archive_root in archive_roots:
        archive_paths.extend(sorted(str(p) for p in Path(archive_root).glob("videos_chunked_*.zip")))

    for suffix in ("mp4", "MP4", "mkv"):
        member = f"data/{video_id}.{suffix}"
        extracted = _extract_zip_member(archive_paths, member, os.path.join(_tmp_media_root(), "videomme"))
        if extracted:
            return extracted
    roots_msg = ", ".join(checked_roots) if checked_roots else "<none>"
    archive_msg = ", ".join(archive_roots) if archive_paths else "<no videos_chunked_*.zip found>"
    raise FileNotFoundError(
        "Video-MME video not found for "
        f"{video_id}. Checked roots: {roots_msg}. "
        f"Checked archives under: {archive_msg}. "
        "If you only prepared Video-MME_8frame.tsv/subtitle without the actual videos, "
        "set VIDEO_MME_ROOT to a directory containing video/*.mp4 (or data/*.mp4), "
        "or place the videos under HF_HOME/datasets/Video-MME_8frame/video."
    )


def _resolve_mvbench_video_path(sub_task: str, video_name: str) -> str:
    dataset_folders = _MVBENCH_DATA_FOLDERS.get(sub_task)
    if dataset_folders is None:
        raise FileNotFoundError(f"Unknown MVBench sub_task={sub_task}")

    for root in _mvbench_roots():
        candidates: List[str] = []
        for dataset_folder in dataset_folders:
            candidates.append(os.path.join(root, dataset_folder, video_name))
            candidates.append(os.path.join(root, "video", dataset_folder, video_name))

        extracted = _find_existing_media_path(candidates)
        if extracted:
            return extracted

    for root in _mvbench_roots():
        archive_paths = sorted(str(p) for p in Path(root, "video").glob("*.zip"))
        for dataset_folder in dataset_folders:
            extracted = _extract_zip_member(
                archive_paths,
                f"{dataset_folder}/{video_name}",
                os.path.join(_tmp_media_root(), "mvbench"),
            )
            if extracted:
                return extracted
    raise FileNotFoundError(f"MVBench video not found for sub_task={sub_task}, video={video_name}")


def _resolve_longvideobench_video_path(video_path: str) -> str:
    candidates = [
        os.path.join("/home/data2/dyf/LongVideoBench", "videos", video_path),
        os.path.join(_hf_home(), "datasets", "longvideobench", "videos", video_path),
        os.path.join(_hf_home(), "datasets", "longvideobench___long_video_bench", "videos", video_path),
        os.path.join(
            _hf_home(),
            "hub",
            "datasets--longvideobench--LongVideoBench",
            "snapshots",
            "60d1c89c1919a198b73be39c2babb213b29d6a5c",
            "videos",
            video_path,
        ),
    ]
    extracted = _find_existing_media_path(candidates)
    if extracted:
        return extracted
    raise FileNotFoundError(
        f"LongVideoBench video not found for {video_path}. "
        "Expected extracted videos under /home/data2/dyf/LongVideoBench or $HF_HOME/datasets/longvideobench or the hub snapshot."
    )


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
    from tasks.coco import coco_doc_to_text

    def coco_doc_to_visual(doc):
        return [doc["image"].convert("RGB")]

    def coco_doc_to_answer(doc):
        ans = doc["answer"]
        return ans[0] if isinstance(ans, list) else ans
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

def _load_coco_rows():
    from datasets import load_dataset
    from tasks.dataset_paths import require_dataset_dir
    return load_dataset(require_dataset_dir("COCO-Caption2017", "data"), token=True)["validation"]


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
    return _load_hf_dataset("lmms-lab/textvqa", split="validation")


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
    return _load_hf_dataset("lmms-lab/ChartQA", split="test")


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
    return _load_hf_dataset("Lin-Chen/MMStar", split="val")


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
    local_parquet = os.path.join(_hf_home(), "datasets", "MMBench", "en", "dev-00000-of-00001.parquet")
    if os.path.exists(local_parquet):
        return _load_dataset_from_parquet(local_parquet)
    return _load_hf_dataset("lmms-lab/MMBench", "en", split="dev")


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
    return _load_hf_dataset("lmms-lab/MMVet", split="test")


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
    return _load_hf_dataset("lmms-lab/MME", split="test")


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
    return _load_hf_dataset("lmms-lab/RealWorldQA", split="test")


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

def _load_videommmu_rows():
    from datasets import concatenate_datasets, load_dataset
    from tasks.dataset_paths import require_dataset_dir
    adaptation = load_dataset(require_dataset_dir("VideoMMMU", "Adaptation"), token=True)["test"]
    comprehension = load_dataset(require_dataset_dir("VideoMMMU", "Comprehension"), token=True)["test"]
    perception = load_dataset(require_dataset_dir("VideoMMMU", "Perception"), token=True)["test"]
    return concatenate_datasets([adaptation, comprehension, perception])


# ═══════════════════════════════════════════════════════════════════════════════
# MVBench (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mvbench_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        if doc.get("local_video_path") and os.path.exists(doc["local_video_path"]):
            return [doc["local_video_path"]]
        return [_resolve_mvbench_video_path(doc["sub_task"], doc["video"])]

    def doc_to_text(doc):
        options = "\n".join(
            f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(doc["candidates"])
        )
        return (
            f"Question:{doc['question']}\n"
            f"Option:\n{options}\n"
            "Answer with the option's letter from the given choices directly."
        )

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
    local_tsv = os.path.join(_hf_home(), "datasets", "MVBench_8frame", "MVBench_8frame.tsv")
    if os.path.exists(local_tsv):
        rows: List[dict] = []
        for row in _load_tsv_rows(local_tsv):
            candidates = [_strip_option_prefix(option) for option in _parse_json_list(row["candidates"])]
            answer = _option_letter_from_answer(row["answer"], candidates)
            rows.append(
                {
                    **row,
                    "candidates": candidates,
                    "answer": answer,
                    "local_video_path": os.path.join(
                        _hf_home(),
                        "datasets",
                        "MVBench_8frame",
                        "video",
                        row["sub_task"],
                        row["video"],
                    ),
                }
            )
        return rows

    mvbench_json_root = os.path.join(_hf_home(), "datasets", "MVBench", "json")
    if os.path.isdir(mvbench_json_root):
        rows: List[dict] = []
        for json_path in sorted(Path(mvbench_json_root).glob("*.json")):
            sub_task = json_path.stem
            with open(json_path, "r") as f:
                task_rows = json.load(f)
            for row in task_rows:
                row["sub_task"] = sub_task
                rows.append(row)
        return rows

    # Fallback: build a merged list from all configs if local json cache is absent.
    from datasets import concatenate_datasets, load_dataset

    config_names = [
        "action_sequence",
        "moving_count",
        "action_prediction",
        "episodic_reasoning",
        "action_antonym",
        "action_count",
        "scene_transition",
        "object_shuffle",
        "object_existence",
        "fine_grained_pose",
        "unexpected_action",
        "moving_direction",
        "state_change",
        "object_interaction",
        "character_order",
        "action_localization",
        "counterfactual_inference",
        "fine_grained_action",
        "moving_attribute",
        "egocentric_navigation",
    ]
    datasets = []
    for config_name in config_names:
        ds = load_dataset("OpenGVLab/MVBench", config_name, split="test", token=True)
        ds = ds.add_column("sub_task", [config_name] * len(ds))
        datasets.append(ds)
    return concatenate_datasets(datasets)


# ═══════════════════════════════════════════════════════════════════════════════
# EgoSchema (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_egoschema_helpers() -> TaskHelpers:
    from lmms_eval.tasks.egoschema.utils import egoschema_doc_to_text

    def doc_to_visual(doc):
        return [_resolve_egoschema_video_path(doc["video_idx"])]

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
    local_parquet = os.path.join(_hf_home(), "datasets", "egoschema", "GENERATION", "test-00000-of-00001.parquet")
    if os.path.exists(local_parquet):
        return _load_dataset_from_parquet(local_parquet)
    return _load_hf_dataset("lmms-lab/egoschema", "GENERATION", split="test")


def _build_egoschema_subset_helpers() -> TaskHelpers:
    from lmms_eval.tasks.egoschema.utils import egoschema_doc_to_text

    def doc_to_visual(doc):
        return [_resolve_egoschema_video_path(doc["video_idx"])]

    def doc_to_text(doc):
        return egoschema_doc_to_text(doc, lmms_eval_specific_kwargs={
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        })

    def doc_to_answer(doc):
        answer = str(doc["answer"]).strip()
        if answer.isdigit():
            idx = int(answer)
            if 0 <= idx < 5:
                return chr(ord("A") + idx)
        return answer

    return TaskHelpers(
        task_name="egoschema_subset",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="video",
    )


def _load_egoschema_subset_rows() -> List[dict]:
    local_parquet = os.path.join(_hf_home(), "datasets", "egoschema", "Subset", "test-00000-of-00001.parquet")
    if os.path.exists(local_parquet):
        return _load_dataset_from_parquet(local_parquet)
    return _load_hf_dataset("lmms-lab/egoschema", "Subset", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# VideoMME (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_videomme_helpers() -> TaskHelpers:
    def doc_to_visual(doc):
        if doc.get("local_video_path") and os.path.exists(doc["local_video_path"]):
            return [doc["local_video_path"]]
        return [_resolve_videomme_video_path(doc["videoID"])]

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
    local_tsv = os.path.join(_hf_home(), "datasets", "Video-MME_8frame", "Video-MME_8frame.tsv")
    if os.path.exists(local_tsv):
        rows: List[dict] = []
        for row in _load_tsv_rows(local_tsv):
            options = [_strip_option_prefix(option) for option in _parse_json_list(row["candidates"])]
            rows.append(
                {
                    "videoID": row["video"],
                    "question": row["question"],
                    "options": options,
                    "answer": row["answer"].strip().upper(),
                    "duration": row.get("duration", ""),
                    "domain": row.get("domain", ""),
                    "sub_category": row.get("sub_category", ""),
                    "task_type": row.get("task_type", ""),
                    "subtitle_path": row.get("subtitle_path", ""),
                    "video_path": row.get("video_path", ""),
                    "local_video_path": os.path.join(
                        _hf_home(),
                        "datasets",
                        "Video-MME_8frame",
                        "video",
                        os.path.basename(row["video_path"]),
                    ),
                }
            )
        return rows

    local_arrow = os.path.join(
        _hf_home(),
        "datasets",
        "lmms-lab___video-mme",
        "videomme",
        "0.0.0",
        "ead1408f75b618502df9a1d8e0950166bf0a2a0b",
        "video-mme-test.arrow",
    )
    if os.path.exists(local_arrow):
        return _load_dataset_from_arrow(local_arrow)
    return _load_hf_dataset("lmms-lab/Video-MME", "videomme", split="test")


# ═══════════════════════════════════════════════════════════════════════════════
# LongVideoBench (video - multiple choice)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_longvideobench_helpers() -> TaskHelpers:
    from lmms_eval.tasks.longvideobench.utils import (
        longvideobench_doc_to_text as lvb_doc_to_text,
    )

    def doc_to_visual(doc):
        return [_resolve_longvideobench_video_path(doc["video_path"])]

    def doc_to_text(doc):
        return lvb_doc_to_text(doc, lmms_eval_specific_kwargs={
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        })

    def doc_to_answer(doc):
        answer = doc.get("correct_choice", doc.get("answer", ""))
        if isinstance(answer, int) and 0 <= answer < 26:
            return chr(ord("A") + answer)
        return str(answer)

    return TaskHelpers(
        task_name="longvideobench",
        doc_to_visual=doc_to_visual,
        doc_to_text=doc_to_text,
        doc_to_answer=doc_to_answer,
        evaluate=_mc_accuracy_evaluate,
        media_type="video",
    )

def _load_longvideobench_rows() -> List[dict]:
    local_arrow = os.path.join(
        _hf_home(),
        "datasets",
        "longvideobench___long_video_bench",
        "default",
        "0.0.0",
        "60d1c89c1919a198b73be39c2babb213b29d6a5c",
        "long_video_bench-validation.arrow",
    )
    if os.path.exists(local_arrow):
        return _load_dataset_from_arrow(local_arrow)
    return _load_hf_dataset("longvideobench/LongVideoBench", split="validation")


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
    "egoschema_subset": (_load_egoschema_subset_rows, _build_egoschema_subset_helpers),
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
    dataset = load_fn()

    # HF Dataset objects need .select() for slicing; plain lists use normal slicing.
    try:
        from datasets import Dataset as HFDataset
        is_hf = isinstance(dataset, HFDataset)
    except ImportError:
        is_hf = False

    if is_hf:
        indices = list(range(start_idx, len(dataset)))
        if num_samples > 0:
            if subset_seed is not None:
                rng = random.Random(subset_seed)
                indices = rng.sample(indices, min(num_samples, len(indices)))
            else:
                indices = indices[:num_samples]
        pool = dataset.select(indices)
    else:
        pool = dataset[start_idx:]
        if num_samples > 0:
            if subset_seed is not None:
                rng = random.Random(subset_seed)
                pool = rng.sample(pool, min(num_samples, len(pool)))
            else:
                pool = pool[:num_samples]

    return pool, build_helpers_fn()
