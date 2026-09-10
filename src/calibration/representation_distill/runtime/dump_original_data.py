import os
import random
from typing import Any, Dict, List, Mapping, Sequence

from datasets import concatenate_datasets, load_dataset
import pyarrow.parquet as pq

from tasks.coco import coco_doc_to_text
from tasks.dataset_paths import get_hf_home, require_dataset_dir
from tasks.gqa import gqa_doc_to_text, gqa_doc_to_visual, load_gqa_instruction_rows
from tasks.m4_instruct import load_m4_instruct_rows
from tasks.video_mmmu import (
    _build_placeholder_frame,
    _image_payload_to_frame,
    get_cache_dir,
    process_media,
)

from src.calibration.representation_distill.common import dump_json, ensure_dir


SUPPORTED_DATASETS = ("gqa", "coco", "m4_instruct", "video_mmmu")
_VIDEOMMMU_ADAPTATION_IMAGE_BY_ID: Dict[str, object] | None = None


def _resolve_hub_snapshot_path(repo_type: str, repo_id: str, *parts: str) -> str | None:
    repo_dir = f"{repo_type}s--{repo_id.replace('/', '--')}"
    snapshot_root = os.path.join(get_hf_home(), "hub", repo_dir, "snapshots")
    if not os.path.isdir(snapshot_root):
        return None
    candidates = sorted(
        [
            os.path.join(snapshot_root, entry)
            for entry in os.listdir(snapshot_root)
            if os.path.isdir(os.path.join(snapshot_root, entry))
        ]
    )
    if not candidates:
        return None
    return os.path.join(candidates[-1], *parts)


def _resolve_dataset_source(dataset_name: str, *parts: str) -> str:
    dataset_repo_map = {
        "COCO-Caption2017": "lmms-lab/COCO-Caption2017",
        "VideoMMMU": "lmms-lab/VideoMMMU",
    }
    try:
        return require_dataset_dir(dataset_name, *parts)
    except FileNotFoundError:
        repo_id = dataset_repo_map.get(dataset_name)
        if repo_id is None:
            raise
        fallback = _resolve_hub_snapshot_path("dataset", repo_id, *parts)
        if fallback and os.path.exists(fallback):
            return fallback
        raise


def _load_videommmu_adaptation_image_index() -> Dict[str, object]:
    global _VIDEOMMMU_ADAPTATION_IMAGE_BY_ID
    if _VIDEOMMMU_ADAPTATION_IMAGE_BY_ID is None:
        path = os.path.join(
            _resolve_dataset_source("VideoMMMU", "Adaptation"),
            "test-00000-of-00001.parquet",
        )
        rows = pq.read_table(path, columns=["id", "image"]).to_pylist()
        _VIDEOMMMU_ADAPTATION_IMAGE_BY_ID = {row["id"]: row.get("image") for row in rows}
    return _VIDEOMMMU_ADAPTATION_IMAGE_BY_ID


def _sample_indices(total_size: int, sample_count: int, seed: int) -> List[int]:
    if sample_count > total_size:
        raise ValueError(
            f"Requested {sample_count} samples from dataset of size {total_size}."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_size), sample_count))


def _format_videommmu_text(row: Dict[str, Any]) -> str:
    question = row["question"].strip()
    options = row.get("options") or []
    if options:
        option_lines = [f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(options)]
        question = f"{question}\nOptions:\n" + "\n".join(option_lines)
    qtype = row.get("question_type", "")
    if qtype:
        question = f"[{qtype}] {question}"
    return question


def _load_videommmu_frames(row: Dict[str, Any], num_frames: int, max_long_side: int):
    videommmu_home = _resolve_dataset_source("VideoMMMU")
    subject = "_".join(row["id"].split("_")[1:-1])
    video_root = os.path.join(videommmu_home, get_cache_dir(subject))
    video_path = os.path.join(video_root, f"{row['id']}.mp4")
    if os.path.exists(video_path):
        frames, _ = process_media(
            video_path,
            max_frames=num_frames,
            max_long_side=max_long_side,
        )
        return frames

    frame = _image_payload_to_frame(row.get("image"))
    if frame is None:
        frame = _image_payload_to_frame(_load_videommmu_adaptation_image_index().get(row["id"]))
    if frame is None:
        frame = _build_placeholder_frame()
    return [frame]


def _build_gqa_sample(row: Dict[str, Any], dataset_index: int) -> Dict[str, Any]:
    return {
        "dataset_name": "gqa",
        "dataset_index": dataset_index,
        "sample_id": row.get("question_id", f"gqa_{dataset_index}"),
        "task_type": "vqa",
        "text": gqa_doc_to_text(row),
        "images": gqa_doc_to_visual(row),
        "video_frames": [],
        "choices": [],
        "answer": row.get("answer"),
        "metadata": {
            "question": row.get("question"),
            "full_answer": row.get("fullAnswer"),
            "image_id": row.get("imageId"),
        },
    }


def _build_coco_sample(row: Dict[str, Any], dataset_index: int) -> Dict[str, Any]:
    return {
        "dataset_name": "coco",
        "dataset_index": dataset_index,
        "sample_id": row.get("id", f"coco_{dataset_index}"),
        "task_type": "caption",
        "text": coco_doc_to_text(row),
        "images": [row["image"].convert("RGB")],
        "video_frames": [],
        "choices": [],
        "answer": row.get("answer", [""])[0],
        "metadata": {
            "captions": row.get("answer"),
        },
    }


def _build_m4_sample(row: Dict[str, Any], dataset_index: int) -> Dict[str, Any]:
    from tasks.m4_instruct import _load_image_from_zip

    images = [_load_image_from_zip(path) for path in row["image_paths"]]
    return {
        "dataset_name": "m4_instruct",
        "dataset_index": dataset_index,
        "sample_id": row.get("sample_id", f"m4_{dataset_index}"),
        "task_type": "instruction",
        "text": row["question"],
        "images": images,
        "video_frames": [],
        "choices": [],
        "answer": row.get("answer"),
        "metadata": {
            "num_images": row.get("num_images"),
            "org_text": row.get("org_text"),
            "metadata": row.get("metadata"),
        },
    }


def _build_videommmu_sample(
    row: Dict[str, Any],
    dataset_index: int,
    num_frames: int,
    max_long_side: int,
) -> Dict[str, Any]:
    return {
        "dataset_name": "video_mmmu",
        "dataset_index": dataset_index,
        "sample_id": row["id"],
        "task_type": "video_qa",
        "text": _format_videommmu_text(row),
        "images": [],
        "video_frames": _load_videommmu_frames(
            row,
            num_frames=num_frames,
            max_long_side=max_long_side,
        ),
        "choices": row.get("options") or [],
        "answer": row.get("answer"),
        "metadata": {
            "question_type": row.get("question_type"),
            "image_fallback": row.get("image") is not None,
        },
    }


def _load_dataset_rows(dataset_name: str, samples_per_dataset: int) -> Sequence[Dict[str, Any]]:
    if dataset_name == "gqa":
        return load_gqa_instruction_rows()
    if dataset_name == "coco":
        return load_dataset(
            _resolve_dataset_source("COCO-Caption2017", "data"),
            token=True,
        )["validation"]
    if dataset_name == "m4_instruct":
        return load_m4_instruct_rows(max_rows=max(samples_per_dataset * 2, samples_per_dataset))
    if dataset_name == "video_mmmu":
        adaptation = load_dataset(
            _resolve_dataset_source("VideoMMMU", "Adaptation"),
            token=True,
        )["test"]
        comprehension = load_dataset(
            _resolve_dataset_source("VideoMMMU", "Comprehension"),
            token=True,
        )["test"]
        perception = load_dataset(
            _resolve_dataset_source("VideoMMMU", "Perception"),
            token=True,
        )["test"]
        return concatenate_datasets([adaptation, comprehension, perception])
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset_rows(
    dataset_name: str,
    *,
    minimum_rows: int = 1,
) -> Sequence[Dict[str, Any]]:
    """Load source rows without materializing image/video payloads."""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets: {SUPPORTED_DATASETS}"
        )
    return _load_dataset_rows(dataset_name, samples_per_dataset=max(int(minimum_rows), 1))


def build_raw_sample(
    dataset_name: str,
    row: Dict[str, Any],
    dataset_index: int,
    *,
    num_video_frames: int = 8,
    video_max_long_side: int = 480,
) -> Dict[str, Any]:
    """Materialize one unified raw sample from a source row."""
    if dataset_name == "gqa":
        sample = _build_gqa_sample(row, dataset_index)
    elif dataset_name == "coco":
        sample = _build_coco_sample(row, dataset_index)
    elif dataset_name == "m4_instruct":
        sample = _build_m4_sample(row, dataset_index)
    elif dataset_name == "video_mmmu":
        sample = _build_videommmu_sample(
            row,
            dataset_index,
            num_frames=num_video_frames,
            max_long_side=video_max_long_side,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    sample["dataset_id"] = SUPPORTED_DATASETS.index(dataset_name)
    return sample


class ManifestRawDataset:
    """Lazily materialize unified raw samples referenced by a JSON manifest."""

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        num_video_frames: int = 8,
        video_max_long_side: int = 480,
        rows_by_dataset: Mapping[str, Sequence[Dict[str, Any]]] | None = None,
    ) -> None:
        self.entries = [dict(entry) for entry in entries]
        self.num_video_frames = int(num_video_frames)
        self.video_max_long_side = int(video_max_long_side)
        self._rows_by_dataset: Dict[str, Sequence[Dict[str, Any]]] = dict(
            rows_by_dataset or {}
        )

        max_index_by_dataset: Dict[str, int] = {}
        for entry in self.entries:
            dataset_name = str(entry["dataset_name"])
            if dataset_name not in SUPPORTED_DATASETS:
                raise ValueError(f"Unsupported manifest dataset: {dataset_name}")
            dataset_index = int(entry["dataset_index"])
            if dataset_index < 0:
                raise ValueError(f"dataset_index must be non-negative, got {dataset_index}")
            max_index_by_dataset[dataset_name] = max(
                max_index_by_dataset.get(dataset_name, -1), dataset_index
            )

        for dataset_name, max_index in max_index_by_dataset.items():
            rows = self._rows_by_dataset.get(dataset_name)
            if rows is None:
                rows = load_dataset_rows(dataset_name, minimum_rows=max_index + 1)
            if max_index >= len(rows):
                raise IndexError(
                    f"Manifest references {dataset_name}[{max_index}], but dataset has {len(rows)} rows."
                )
            self._rows_by_dataset[dataset_name] = rows

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        dataset_name = str(entry["dataset_name"])
        dataset_index = int(entry["dataset_index"])
        sample = build_raw_sample(
            dataset_name,
            self._rows_by_dataset[dataset_name][dataset_index],
            dataset_index,
            num_video_frames=self.num_video_frames,
            video_max_long_side=self.video_max_long_side,
        )
        expected_id = entry.get("sample_id")
        if expected_id is not None and str(sample["sample_id"]) != str(expected_id):
            raise ValueError(
                "Manifest sample identity mismatch for "
                f"{dataset_name}[{dataset_index}]: expected {expected_id!r}, "
                f"loaded {sample['sample_id']!r}."
            )
        return sample


def dump_original_data(
    *,
    output_dir: str,
    samples_per_dataset: int = 1024,
    seed: int = 42,
    num_video_frames: int = 8,
    video_max_long_side: int = 480,
    shuffle_seed: int | None = 1234,
    selected_datasets: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    index_dir = os.path.join(output_dir, "sample_indices")
    ensure_dir(index_dir)

    combined_samples: List[Dict[str, Any]] = []
    dataset_summary: Dict[str, Any] = {}
    dataset_ids = {name: idx for idx, name in enumerate(SUPPORTED_DATASETS)}
    dataset_names = list(selected_datasets) if selected_datasets is not None else list(SUPPORTED_DATASETS)

    invalid = [name for name in dataset_names if name not in dataset_ids]
    if invalid:
        raise ValueError(
            f"Unsupported datasets: {invalid}. Supported datasets: {list(SUPPORTED_DATASETS)}"
        )
    if not dataset_names:
        raise ValueError("selected_datasets must contain at least one dataset name.")

    for dataset_offset, dataset_name in enumerate(dataset_names):
        rows = _load_dataset_rows(dataset_name, samples_per_dataset=samples_per_dataset)
        indices = _sample_indices(
            total_size=len(rows),
            sample_count=samples_per_dataset,
            seed=seed + dataset_offset,
        )
        dump_json(
            os.path.join(index_dir, f"{dataset_name}.json"),
            {
                "dataset_name": dataset_name,
                "seed": seed + dataset_offset,
                "samples_per_dataset": samples_per_dataset,
                "indices": indices,
            },
        )

        dataset_samples = []
        for dataset_index in indices:
            row = rows[dataset_index]
            sample = build_raw_sample(
                dataset_name,
                row,
                dataset_index,
                num_video_frames=num_video_frames,
                video_max_long_side=video_max_long_side,
            )
            dataset_samples.append(sample)

        combined_samples.extend(dataset_samples)
        dataset_summary[dataset_name] = {
            "dataset_id": dataset_ids[dataset_name],
            "sample_count": len(dataset_samples),
        }

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(combined_samples)

    dump_json(
        os.path.join(output_dir, "dump_original_data.json"),
        {
            "samples_per_dataset": samples_per_dataset,
            "total_samples": len(combined_samples),
            "shuffle_seed": shuffle_seed,
            "num_video_frames": num_video_frames,
            "video_max_long_side": video_max_long_side,
            "selected_datasets": dataset_names,
            "datasets": dataset_summary,
        },
    )
    return combined_samples


# Backward compatibility for existing call sites.
build_teacher_pool = dump_original_data
