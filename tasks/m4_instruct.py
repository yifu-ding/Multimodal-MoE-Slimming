"""M4-Instruct-Data (lmms-lab/M4-Instruct-Data) loader for calibration.

Lazy-download + streaming approach:
  1. Annotation JSON (~1 GB) is downloaded once via hf_hub_download (cached).
  2. Annotation rows are streamed via HF datasets, and we stop as soon as enough
     valid samples are collected (no full JSON load into memory).
  3. Image zip archives are downloaded on-demand — only the zips needed for the
     selected samples are fetched.

All samples in M4-Instruct are multi-image (2-8 images per sample).
"""

import os
import re
import zipfile
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PIL import Image
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

_M4_CACHE: Optional[Dict[str, Any]] = None  # {"zip_handles": {...}}

# Zip archives sorted by size (ascending) so we prefer smaller/local ones first.
_PREFERRED_ZIP_ORDER = [
    "MIT-States_PropertyCoherence.zip",
    "MIT-States_StateCoherence.zip",
    "OCR-VQA.zip",
    "IEdit.zip",
    "CLEVR-Change.zip",
    "DocVQA.zip",
    "VizWiz.zip",
]


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


def _find_local_repo_file(filename: str) -> Optional[str]:
    """Find a dataset file in known local HF cache locations."""
    cache_dir = _get_cache_dir()
    direct_path = os.path.join(cache_dir, filename)
    if os.path.isfile(direct_path):
        return direct_path

    snapshot_root = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
        "datasets--lmms-lab--M4-Instruct-Data",
        "snapshots",
    )
    if os.path.isdir(snapshot_root):
        snapshots = sorted(
            [
                os.path.join(snapshot_root, entry)
                for entry in os.listdir(snapshot_root)
                if os.path.isdir(os.path.join(snapshot_root, entry))
            ]
        )
        for snapshot_dir in reversed(snapshots):
            candidate = os.path.join(snapshot_dir, filename)
            if os.path.isfile(candidate):
                return candidate
    return None


def _offline_mode_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "on", "yes", "true"}


def _list_available_zips() -> set[str]:
    available = set()
    cache_dir = _get_cache_dir()
    if os.path.isdir(cache_dir):
        available.update(name for name in os.listdir(cache_dir) if name.endswith(".zip"))

    snapshot_root = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
        "datasets--lmms-lab--M4-Instruct-Data",
        "snapshots",
    )
    if os.path.isdir(snapshot_root):
        for entry in os.listdir(snapshot_root):
            snapshot_dir = os.path.join(snapshot_root, entry)
            if not os.path.isdir(snapshot_dir):
                continue
            available.update(name for name in os.listdir(snapshot_dir) if name.endswith(".zip"))
    return available


def _get_zip_name(image_path: str) -> str:
    """Derive the zip archive name from an image path, e.g. 'HQ-Edit/images/1.jpg' -> 'HQ-Edit.zip'."""
    top_dir = image_path.split("/")[0]
    return f"{top_dir}.zip"


def _download_zip(zip_name: str) -> str:
    """Download a single zip archive via hf_hub_download (cached)."""
    from huggingface_hub import hf_hub_download

    local_path = _find_local_repo_file(zip_name)
    if local_path is not None:
        return local_path

    if _offline_mode_enabled():
        raise FileNotFoundError(
            f"M4-Instruct asset {zip_name} is not available in local cache, and HF offline mode is enabled. "
            f"Place it under {_get_cache_dir()} or disable HF_HUB_OFFLINE to allow download."
        )

    print(f"[M4-Instruct] Downloading {zip_name} ...")
    try:
        return hf_hub_download(
            repo_id="lmms-lab/M4-Instruct-Data",
            filename=zip_name,
            repo_type="dataset",
            local_dir=_get_cache_dir(),
        )
    except (OfflineModeIsEnabled, LocalEntryNotFoundError) as exc:
        raise FileNotFoundError(
            f"Failed to obtain M4-Instruct asset {zip_name}. "
            f"Checked local cache first and then download failed. "
            f"Expected a cached file under {_get_cache_dir()}."
        ) from exc


def _init_cache() -> Dict[str, Any]:
    """Initialize cache dict."""
    global _M4_CACHE
    if _M4_CACHE is not None:
        return _M4_CACHE

    _M4_CACHE = {
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


def _skip_json_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    return start


def _iter_json_array(path: str, chunk_size: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    """Yield objects from a top-level JSON array without loading the full file."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with open(path, "r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            if chunk:
                buffer += chunk

            cursor = 0
            if not started:
                cursor = _skip_json_whitespace(buffer, cursor)
                if cursor >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON content in {path}")
                    continue
                if buffer[cursor] != "[":
                    raise ValueError(f"Expected top-level JSON array in {path}")
                started = True
                cursor += 1

            while True:
                cursor = _skip_json_whitespace(buffer, cursor)
                if cursor >= len(buffer):
                    break
                if buffer[cursor] == "]":
                    return
                try:
                    item, cursor = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Malformed JSON array in {path}")
                    break
                if not isinstance(item, dict):
                    raise ValueError(f"Expected object entries in {path}, got {type(item).__name__}")
                yield item
                cursor = _skip_json_whitespace(buffer, cursor)
                if cursor >= len(buffer):
                    break
                if buffer[cursor] == ",":
                    cursor += 1
                    continue
                if buffer[cursor] == "]":
                    return
                if eof:
                    raise ValueError(f"Unexpected trailing content in {path}")
                break

            buffer = buffer[cursor:]
            if eof:
                if buffer.strip():
                    raise ValueError(f"Unexpected EOF while parsing {path}")
                return


def _iter_m4_annotations():
    ann_path = _download_annotation_json()
    for ann in _iter_json_array(ann_path):
        yield ann


def load_m4_instruct_rows(max_rows: int = 1024) -> List[Dict[str, Any]]:
    """Collect M4-Instruct rows while preferring locally available/smaller zips."""
    if max_rows <= 0:
        return []

    from collections import defaultdict

    available_zips = _list_available_zips()
    offline = _offline_mode_enabled()
    by_zip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scanned = 0

    for ann in _iter_m4_annotations():
        scanned += 1
        images = ann.get("image")
        if not isinstance(images, list) or not images:
            continue
        zip_name = _get_zip_name(images[0])
        if offline and zip_name not in available_zips:
            continue
        conv = ann.get("conversations", [])
        if not isinstance(conv, list):
            continue
        question_raw, answer = _extract_question_and_answer(conv)
        if not question_raw:
            continue
        by_zip[zip_name].append(
            {
                "sample_id": ann.get("sample_id", f"row-{scanned}"),
                "question": _strip_image_tags(question_raw),
                "answer": answer,
                "org_text": _strip_image_tags(question_raw),
                "image_paths": images,
                "num_images": len(images),
                "metadata": ann.get("metadata", {}),
            }
        )

    def zip_sort_key(zip_name: str) -> tuple[int, int, str]:
        local_rank = 0 if zip_name in available_zips else 1
        preferred_rank = (
            _PREFERRED_ZIP_ORDER.index(zip_name) if zip_name in _PREFERRED_ZIP_ORDER else len(_PREFERRED_ZIP_ORDER)
        )
        return (local_rank, preferred_rank, zip_name)

    ordered_zips = sorted(by_zip.keys(), key=zip_sort_key)
    rows: List[Dict[str, Any]] = []
    for zip_name in ordered_zips:
        remaining = max_rows - len(rows)
        if remaining <= 0:
            break
        rows.extend(by_zip[zip_name][:remaining])

    if offline and len(rows) < max_rows:
        raise FileNotFoundError(
            f"Only found {len(rows)} M4-Instruct samples from locally cached zip files in offline mode, "
            f"but max_rows={max_rows}. Cached zips: {sorted(available_zips)}"
        )

    zips_used = sorted({_get_zip_name(r["image_paths"][0]) for r in rows})
    print(
        f"[M4-Instruct] Collected {len(rows)} rows (requested={max_rows}, scanned={scanned}, offline={offline}) "
        f"from {len(zips_used)} zip(s): {zips_used}"
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
        processed_visuals.append(images)
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
