import os
from glob import glob
from typing import Any, Dict, List

import pyarrow.parquet as pq
from PIL import Image


def _star_root() -> str:
    candidates = [
        os.environ.get("STAR_ROOT", "").strip(),
        "/home/data2/dyf/STAR/train_subset_256",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return "/home/data2/dyf/STAR/train_subset_256"


def _resize_image(image: Image.Image, max_long_side: int = 480) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_long_side:
        return image.convert("RGB")
    if h > w:
        new_h = max_long_side
        new_w = int(round(max_long_side * w / h))
    else:
        new_w = max_long_side
        new_h = int(round(max_long_side * h / w))
    return image.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)


def _sample_frame_paths(video_id: str, max_frames: int = 2) -> List[str]:
    root = _star_root()
    pattern = os.path.join(root, "train_videos", f"{video_id}-*.jpg")
    frame_paths = sorted(
        glob(pattern),
        key=lambda path: int(os.path.basename(path).rsplit(".", 1)[0].rsplit("-", 1)[1]),
    )
    if len(frame_paths) <= max_frames:
        return frame_paths

    selected = []
    for idx in range(max_frames):
        frame_idx = round(idx * (len(frame_paths) - 1) / (max_frames - 1))
        selected.append(frame_paths[frame_idx])
    return selected


def load_star_subset_rows() -> List[Dict[str, Any]]:
    parquet_path = os.path.join(_star_root(), "star_train_subset_256.parquet")
    table = pq.read_table(parquet_path)
    return table.to_pylist()


def star_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    visuals = []
    for frame_path in _sample_frame_paths(doc["video_id"]):
        with Image.open(frame_path) as image:
            visuals.append(_resize_image(image))
    return visuals


def star_doc_to_text(doc: Dict[str, Any]) -> str:
    choices = doc["choices"]["choice"]
    option_lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)]
    return (
        "Analyze the given video clip and select the best answer for the following question.\n"
        f"Question: {doc['question']}\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\nAnswer:"
    )


def star_doc_to_org_text(doc: Dict[str, Any]) -> str:
    return star_doc_to_text(doc)


def star_doc_to_full_answer(doc: Dict[str, Any]) -> str:
    answer = doc["answer"]
    choices = doc["choices"]["choice"]
    for idx, choice in enumerate(choices):
        if choice == answer:
            return f"{chr(ord('A') + idx)}. {answer}"
    return answer


def star_transform(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    processed_visuals = []
    processed_frames = []
    processed_texts = []
    processed_full_answers = []
    processed_org_texts = []

    for doc in batch["__raw_doc__"]:
        visuals = star_doc_to_visual(doc)
        text = star_doc_to_text(doc)
        full_answer = star_doc_to_full_answer(doc)
        org_text = star_doc_to_org_text(doc)

        processed_visuals.append(visuals)
        processed_frames.append(len(visuals))
        processed_texts.append(text)
        processed_full_answers.append(full_answer)
        processed_org_texts.append(org_text)

    return {
        "model_input_text": processed_texts,
        "model_input_visual": processed_visuals,
        "model_input_frames": processed_frames,
        "model_input_full_answer": processed_full_answers,
        "model_input_org_text": processed_org_texts,
    }
