"""M4-Instruct-Data (lmms-lab/M4-Instruct-Data) loader for calibration.

The dataset consists of:
  - m4_instruct_annotations.json: 1 GB annotation file with conversations + image paths
  - Zip archives containing images referenced by annotations

For calibration we only use single-image samples (the majority of the dataset).
Multi-image samples are skipped to keep the pipeline simple.

Expected local layout (auto-downloaded via HF hub):
  $HF_HOME/datasets/M4-Instruct-Data/m4_instruct_annotations.json
  $HF_HOME/datasets/M4-Instruct-Data/<SubsetName>/...   (extracted images)

If images are not extracted, they can also live under:
  storage/datasets/M4-Instruct-Data/<SubsetName>/...
"""

import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

from tasks.dataset_paths import get_hf_datasets_root, resolve_dataset_dir

_M4_ANNOTATIONS: Optional[List[Dict[str, Any]]] = None
_M4_IMAGE_ROOT: Optional[str] = None


def _resolve_m4_root() -> str:
    """Return the local root directory for M4-Instruct-Data."""
    candidates = [
        os.path.join(get_hf_datasets_root(), "M4-Instruct-Data"),
        os.path.join("storage", "datasets", "M4-Instruct-Data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def _load_annotations() -> List[Dict[str, Any]]:
    """Load and cache the annotation JSON, filtering to single-image samples."""
    global _M4_ANNOTATIONS, _M4_IMAGE_ROOT
    if _M4_ANNOTATIONS is not None:
        return _M4_ANNOTATIONS

    root = _resolve_m4_root()
    _M4_IMAGE_ROOT = root

    ann_path = os.path.join(root, "m4_instruct_annotations.json")
    if not os.path.isfile(ann_path):
        # Try downloading from HF hub
        try:
            from huggingface_hub import hf_hub_download
            ann_path = hf_hub_download(
                repo_id="lmms-lab/M4-Instruct-Data",
                filename="m4_instruct_annotations.json",
                repo_type="dataset",
                local_dir=root,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"M4-Instruct-Data annotations not found at {ann_path}. "
                f"Download failed: {e}"
            ) from e

    print(f"[M4-Instruct] Loading annotations from {ann_path} ...")
    with open(ann_path, "r") as f:
        all_annotations = json.load(f)

    # Filter to single-image samples for calibration
    single_image = [
        ann for ann in all_annotations
        if isinstance(ann.get("image"), list) and len(ann["image"]) == 1
    ]
    print(
        f"[M4-Instruct] Loaded {len(all_annotations)} total annotations, "
        f"{len(single_image)} single-image samples."
    )
    _M4_ANNOTATIONS = single_image
    return _M4_ANNOTATIONS


def _extract_question_and_answer(conversations: List[Dict[str, str]]):
    """Extract question (human turn) and answer (gpt turn) from conversations."""
    question = ""
    answer = ""
    for turn in conversations:
        if turn["from"] == "human":
            # Remove <image> tags and strip
            question = re.sub(r"<image>\s*", "", turn["value"]).strip()
        elif turn["from"] == "gpt":
            answer = turn["value"].strip()
    return question, answer


def _load_image(image_path: str) -> Image.Image:
    """Load an image from the M4 dataset directory."""
    global _M4_IMAGE_ROOT
    if _M4_IMAGE_ROOT is None:
        _M4_IMAGE_ROOT = _resolve_m4_root()

    full_path = os.path.join(_M4_IMAGE_ROOT, image_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(
            f"M4-Instruct image not found: {full_path}. "
            f"Make sure to extract the relevant zip archive under {_M4_IMAGE_ROOT}/."
        )
    return Image.open(full_path).convert("RGB")


def load_m4_instruct_rows() -> List[Dict[str, Any]]:
    """Load M4-Instruct-Data single-image annotations as a list of row dicts.

    Each row has: sample_id, question, answer, image_path, metadata.
    """
    annotations = _load_annotations()
    rows = []
    for ann in annotations:
        question, answer = _extract_question_and_answer(ann["conversations"])
        if not question:
            continue
        rows.append({
            "sample_id": ann["sample_id"],
            "question": question,
            "answer": answer,
            "image_path": ann["image"][0],
            "metadata": ann.get("metadata", {}),
        })
    return rows


def m4_instruct_doc_to_visual(doc):
    image = _load_image(doc["image_path"])
    return [image]


def m4_instruct_doc_to_text(doc):
    return doc["question"]


def m4_instruct_doc_to_org_text(doc):
    return doc["question"]


def m4_instruct_doc_to_answer(doc):
    return doc["answer"]


def m4_instruct_doc_to_full_answer(doc):
    return doc["answer"]


def m4_instruct_transform(batch):
    """Transform an M4-Instruct batch into model input format.

    Returns:
        Dict with model_input_text, model_input_visual, model_input_answer,
        model_input_full_answer, model_input_org_text.
    """
    processed_texts = []
    processed_visuals = []
    processed_answers = []
    processed_full_answers = []
    processed_org_texts = []

    for sample_id, question, answer, image_path in zip(
        batch["sample_id"], batch["question"], batch["answer"], batch["image_path"]
    ):
        doc = {
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "image_path": image_path,
        }
        visuals = m4_instruct_doc_to_visual(doc)
        text = m4_instruct_doc_to_text(doc)
        org_text = m4_instruct_doc_to_org_text(doc)
        processed_visuals.append(visuals[0])
        processed_texts.append(text)
        processed_answers.append(answer)
        processed_full_answers.append(answer)
        processed_org_texts.append(org_text)

    return {
        "model_input_text": processed_texts,
        "model_input_visual": processed_visuals,
        "model_input_answer": processed_answers,
        "model_input_full_answer": processed_full_answers,
        "model_input_org_text": processed_org_texts,
    }
