"""M4-Instruct-Data (lmms-lab/M4-Instruct-Data) loader for calibration.

Lazy-download approach:
  1. Annotation JSON (~1 GB) is downloaded once via hf_hub_download (cached).
  2. Image zip archives are downloaded on-demand — only the zips needed for the
     requested samples are fetched.  Images are extracted from zips on-the-fly
     using Python's zipfile module.

All samples in M4-Instruct are multi-image (2-8 images per sample).
"""

import json
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_M4_CACHE: Optional[Dict[str, Any]] = None  # {"annotations": [...], "zip_handles": {...}}


def _get_cache_dir() -> str:
    """Return a local cache directory for M4-Instruct-Data files."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = os.path.join(hf_home, "datasets", "M4-Instruct-Data")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _download_annotation_json() -> str:
    """Download the annotation JSON via hf_hub_download (cached after first download)."""
    from huggingface_hub import hf_hub_download

    cache_dir = _get_cache_dir()
    local_path = os.path.join(cache_dir, "m4_instruct_annotations.json")
    if os.path.isfile(local_path):
        return local_path

    print("[M4-Instruct] Downloading annotation JSON (~1 GB, one-time) ...")
    path = hf_hub_download(
        repo_id="lmms-lab/M4-Instruct-Data",
        filename="m4_instruct_annotations.json",
        repo_type="dataset",
        local_dir=cache_dir,
    )
    return path


def _get_zip_name(image_path: str) -> str:
    """Derive the zip archive name from an image path, e.g. 'HQ-Edit/images/1.jpg' -> 'HQ-Edit.zip'."""
    top_dir = image_path.split("/")[0]
    return f"{top_dir}.zip"


def _download_zip(zip_name: str) -> str:
    """Download a single zip archive via hf_hub_download (cached)."""
    from huggingface_hub import hf_hub_download

    cache_dir = _get_cache_dir()
    local_path = os.path.join(cache_dir, zip_name)
    if os.path.isfile(local_path):
        return local_path

    print(f"[M4-Instruct] Downloading {zip_name} ...")
    path = hf_hub_download(
        repo_id="lmms-lab/M4-Instruct-Data",
        filename=zip_name,
        repo_type="dataset",
        local_dir=cache_dir,
    )
    return path


def _init_cache() -> Dict[str, Any]:
    """Load annotations and initialize the cache dict."""
    global _M4_CACHE
    if _M4_CACHE is not None:
        return _M4_CACHE

    ann_path = _download_annotation_json()
    print(f"[M4-Instruct] Parsing annotations from {ann_path} ...")
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    print(f"[M4-Instruct] Loaded {len(annotations)} annotations.")

    _M4_CACHE = {
        "annotations": annotations,
        "zip_handles": {},  # zip_name -> zipfile.ZipFile (opened lazily)
    }
    return _M4_CACHE


def _open_zip(zip_name: str) -> zipfile.ZipFile:
    """Open (and cache) a zip file handle. Downloads the zip if needed."""
    cache = _init_cache()
    if zip_name not in cache["zip_handles"]:
        zip_path = _download_zip(zip_name)
        cache["zip_handles"][zip_name] = zipfile.ZipFile(zip_path, "r")
    return cache["zip_handles"][zip_name]


def _load_image_from_zip(image_path: str) -> Image.Image:
    """Load a single image from the appropriate zip archive."""
    zip_name = _get_zip_name(image_path)
    zf = _open_zip(zip_name)
    with zf.open(image_path) as img_file:
        return Image.open(img_file).convert("RGB")


def _extract_question_and_answer(
    conversations: List[Dict[str, str]],
) -> Tuple[str, str]:
    """Extract question (human turn) and answer (gpt turn) from conversations."""
    question = ""
    answer = ""
    for turn in conversations:
        if turn["from"] == "human":
            question = turn["value"].strip()
        elif turn["from"] == "gpt":
            answer = turn["value"].strip()
    return question, answer


def _strip_image_tags(text: str) -> str:
    """Remove <image> tags from text."""
    return re.sub(r"<image>\s*", "", text).strip()


# Zip archives sorted by size (ascending) so we download the smallest first.
_PREFERRED_ZIP_ORDER = [
    "MIT-States_PropertyCoherence.zip",  # 74 MB
    "MIT-States_StateCoherence.zip",     # 100 MB
    "OCR-VQA.zip",                       # 218 MB
    "IEdit.zip",                         # 271 MB
    "DocVQA.zip",                        # 2.4 GB
    "CLEVR-Change.zip",                  # 1.5 GB
    "VizWiz.zip",                        # 4.9 GB
]


def load_m4_instruct_rows(max_rows: int = 1024) -> List[Dict[str, Any]]:
    """Load up to *max_rows* M4-Instruct annotations.

    Prioritises samples from smaller zip archives to minimise downloads.
    Each row dict has: sample_id, question, answer, image_paths, num_images,
                       org_text, metadata.
    """
    cache = _init_cache()
    annotations = cache["annotations"]

    # Group annotations by source zip
    from collections import defaultdict
    by_zip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        images = ann.get("image")
        if not isinstance(images, list) or not images:
            continue
        zn = _get_zip_name(images[0])
        by_zip[zn].append(ann)

    # Pick from preferred (small) zips first, then the rest
    seen_zips = set()
    ordered_zips = []
    for zn in _PREFERRED_ZIP_ORDER:
        if zn in by_zip:
            ordered_zips.append(zn)
            seen_zips.add(zn)
    for zn in by_zip:
        if zn not in seen_zips:
            ordered_zips.append(zn)

    rows: List[Dict[str, Any]] = []
    for zn in ordered_zips:
        if len(rows) >= max_rows:
            break
        for ann in by_zip[zn]:
            if len(rows) >= max_rows:
                break
            question_raw, answer = _extract_question_and_answer(ann["conversations"])
            if not question_raw:
                continue
            rows.append({
                "sample_id": ann["sample_id"],
                "question": _strip_image_tags(question_raw),
                "answer": answer,
                "org_text": _strip_image_tags(question_raw),
                "image_paths": ann["image"],
                "num_images": len(ann["image"]),
                "metadata": ann.get("metadata", {}),
            })

    zips_used = {_get_zip_name(r["image_paths"][0]) for r in rows}
    print(
        f"[M4-Instruct] Prepared {len(rows)} rows (max_rows={max_rows}) "
        f"from {len(zips_used)} zip(s): {sorted(zips_used)}"
    )
    return rows


def m4_instruct_transform(batch):
    """Transform an M4-Instruct batch into model input format.

    Multi-image: model_input_visual is a flat list of all images across the
    batch; model_input_frames records how many images each sample has (so
    prepare_inputs can duplicate media placeholders, same as video_mmmu).
    """
    processed_texts = []
    processed_visuals = []
    processed_answers = []
    processed_full_answers = []
    processed_org_texts = []
    processed_frames = []

    for question, answer, image_paths, num_images, org_text in zip(
        batch["question"],
        batch["answer"],
        batch["image_paths"],
        batch["num_images"],
        batch["org_text"],
    ):
        images = [_load_image_from_zip(p) for p in image_paths]
        processed_visuals.extend(images)
        processed_frames.append(num_images)
        processed_texts.append(question)
        processed_answers.append(answer)
        processed_full_answers.append(answer)
        processed_org_texts.append(org_text)

    return {
        "model_input_text": processed_texts,
        "model_input_visual": processed_visuals,
        "model_input_answer": processed_answers,
        "model_input_full_answer": processed_full_answers,
        "model_input_org_text": processed_org_texts,
        "model_input_frames": processed_frames,
    }
