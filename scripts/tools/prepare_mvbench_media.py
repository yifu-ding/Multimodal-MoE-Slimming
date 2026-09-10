#!/usr/bin/env python3
"""Prepare the media layout expected by lmms-eval's MVBench video revision."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from datasets import Dataset
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm


PREPARE_SPECS = {
    "object_interaction": ("star/Charades_segment", ["star/Charades_v1_480", "data0613/star/Charades_v1_480"], "video"),
    "action_sequence": ("star/Charades_segment", ["star/Charades_v1_480", "data0613/star/Charades_v1_480"], "video"),
    "action_prediction": ("star/Charades_segment", ["star/Charades_v1_480", "data0613/star/Charades_v1_480"], "video"),
    "action_localization": ("sta/sta_video_segment", ["sta/sta_video"], "video"),
    "episodic_reasoning": ("tvqa/video_fps3_hq_segment", ["tvqa/frames_fps3_hq"], "frames"),
    "action_antonym": ("ssv2_video_mp4", ["ssv2_video"], "video"),
}

EXPECTED_DIRS = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}


@dataclass(frozen=True)
class Job:
    task: str
    source: str
    target: str
    kind: str
    start: float | None
    end: float | None
    source_fps: float | None = None


def parse_args() -> argparse.Namespace:
    default_hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, default=default_hf_home)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--arrow-cache", type=Path)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _arrow_dataset(cache_root: Path, task: str) -> Dataset:
    matches = sorted((cache_root / task).glob("*/*/*.arrow"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one Arrow file for {task}, found {len(matches)} under {cache_root / task}")
    return Dataset.from_file(str(matches[0]))


def _find_source(media_root: Path, folders: list[str], name: str) -> Path | None:
    for folder in folders:
        candidate = media_root / folder / name
        if candidate.exists():
            return candidate
    return None


def build_jobs(source_root: Path, media_root: Path, arrow_cache: Path) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    errors: list[str] = []
    annotation_root = source_root / "json"

    for task, (target_dir, source_dirs, kind) in PREPARE_SPECS.items():
        raw_docs = json.loads((annotation_root / f"{task}.json").read_text())
        expected_docs = _arrow_dataset(arrow_cache, task)
        if len(raw_docs) != len(expected_docs):
            errors.append(f"{task}: annotation count {len(raw_docs)} != Arrow count {len(expected_docs)}")
            continue

        for index, (raw_doc, expected_doc) in enumerate(zip(raw_docs, expected_docs)):
            if raw_doc["question"] != expected_doc["question"]:
                errors.append(f"{task}[{index}]: main/video revision row mismatch")
                continue
            source = _find_source(media_root, source_dirs, raw_doc["video"])
            if source is None:
                errors.append(f"{task}[{index}]: source missing: {raw_doc['video']}")
                continue
            jobs.append(
                Job(
                    task=task,
                    source=str(source),
                    target=str(media_root / target_dir / expected_doc["video"]),
                    kind=kind,
                    start=float(raw_doc["start"]) if "start" in raw_doc else None,
                    end=float(raw_doc["end"]) if "end" in raw_doc else None,
                    source_fps=float(raw_doc.get("fps", 0)) or None,
                )
            )
    return jobs, errors


def repair_data0613_files(media_root: Path, arrow_cache: Path, dry_run: bool) -> int:
    repaired = 0
    for task, folder in EXPECTED_DIRS.items():
        for doc in _arrow_dataset(arrow_cache, task):
            target = media_root / folder / doc["video"]
            patch_source = media_root / "data0613" / folder / doc["video"]
            if target.exists() or not patch_source.is_file():
                continue
            repaired += 1
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(patch_source, target)
    return repaired


def _uniform_indices(first: int, last: int, maximum: int) -> list[int]:
    if last < first:
        raise ValueError(f"empty frame interval: {first}..{last}")
    count = min(maximum, last - first + 1)
    return np.unique(np.linspace(first, last, count, dtype=int)).tolist()


def _read_video_frames(job: Job, max_frames: int) -> list[np.ndarray]:
    reader = VideoReader(job.source, ctx=cpu(0), num_threads=1)
    total = len(reader)
    if total == 0:
        raise ValueError("source has no frames")
    fps = float(reader.get_avg_fps())
    first = 0 if job.start is None else max(0, math.floor(job.start * fps))
    last = total - 1 if job.end is None else min(total - 1, math.ceil(job.end * fps) - 1)
    indices = _uniform_indices(first, last, max_frames)
    return list(reader.get_batch(indices).asnumpy())


def _read_image_frames(job: Job, max_frames: int) -> list[np.ndarray]:
    files = sorted(
        path
        for path in Path(job.source).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise ValueError("source frame directory is empty")
    fps = job.source_fps or 3.0
    first = 0 if job.start is None else max(0, math.floor(job.start * fps))
    last = len(files) - 1 if job.end is None else min(len(files) - 1, math.ceil(job.end * fps) - 1)
    indices = _uniform_indices(first, last, max_frames)
    frames = []
    for index in indices:
        with Image.open(files[index]) as image:
            frames.append(np.asarray(image.convert("RGB")))
    return frames


def _encode_mp4(frames: list[np.ndarray], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    try:
        with av.open(str(temporary), mode="w", format="mp4") as output:
            stream = output.add_stream("libx264", rate=2)
            height, width = frames[0].shape[:2]
            stream.width = width - width % 2
            stream.height = height - height % 2
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "veryfast", "threads": "1"}
            for array in frames:
                array = np.ascontiguousarray(array[: stream.height, : stream.width, :3])
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_one(job: Job, max_frames: int, overwrite: bool) -> tuple[str, str]:
    target = Path(job.target)
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return "skipped", job.target
    frames = _read_image_frames(job, max_frames) if job.kind == "frames" else _read_video_frames(job, max_frames)
    _encode_mp4(frames, target)
    return "created", job.target


def verify(media_root: Path, arrow_cache: Path) -> bool:
    total = 0
    ready = 0
    for task, folder in EXPECTED_DIRS.items():
        docs = _arrow_dataset(arrow_cache, task)
        task_ready = sum((media_root / folder / doc["video"]).is_file() for doc in docs)
        total += len(docs)
        ready += task_ready
        print(f"{task}: ready={task_ready}/{len(docs)}")
    print(f"MVBench total: ready={ready}/{total}, missing={total - ready}")
    return ready == total


def main() -> int:
    args = parse_args()
    source_root = args.source_root or args.hf_home / "datasets" / "MVBench"
    media_root = args.media_root or args.hf_home / "mvbench_video"
    arrow_cache = args.arrow_cache or args.hf_home / "datasets" / "OpenGVLab___mv_bench"

    if args.verify_only:
        return 0 if verify(media_root, arrow_cache) else 1

    repaired = repair_data0613_files(media_root, arrow_cache, args.dry_run)
    jobs, errors = build_jobs(source_root, media_root, arrow_cache)
    if errors:
        print("Cannot prepare all available-source jobs:")
        for error in errors[:50]:
            print(f"  {error}")
        return 1

    pending = [job for job in jobs if args.overwrite or not Path(job.target).is_file()]
    print(f"jobs={len(jobs)} pending={len(pending)} patch_files={repaired} max_frames={args.max_frames}")
    if args.dry_run:
        for job in pending[:20]:
            print(f"{job.task}: {job.source} -> {job.target}")
        return 0

    failures: list[str] = []
    created = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prepare_one, job, args.max_frames, args.overwrite): job for job in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing MVBench media"):
            job = futures[future]
            try:
                status, _ = future.result()
                created += status == "created"
            except Exception as exc:
                failures.append(f"{job.task}: {job.source} -> {job.target}: {exc}")

    print(f"created={created} failed={len(failures)}")
    for failure in failures[:50]:
        print(f"  {failure}")
    verify(media_root, arrow_cache)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
