import os
from pathlib import Path

from lmms_eval.tasks.egoschema.utils import (
    egoschema_aggregate_mc,
    egoschema_aggregate_score,
    egoschema_doc_to_answer,
    egoschema_doc_to_text,
    egoschema_process_results_generation,
)


def egoschema_local_doc_to_visual(doc):
    root = Path(
        os.environ.get(
            "EGOSCHEMA_ROOT",
            "/home/data/dyf/hf_cache/datasets/egoschema",
        )
    )
    video_path = root / "videos" / f"{doc['video_idx']}.mp4"
    if not video_path.is_file():
        upper_path = video_path.with_suffix(".MP4")
        if upper_path.is_file():
            video_path = upper_path
        else:
            raise FileNotFoundError(f"EgoSchema video does not exist: {video_path}")
    return [str(video_path)]


__all__ = [
    "egoschema_aggregate_mc",
    "egoschema_aggregate_score",
    "egoschema_doc_to_answer",
    "egoschema_doc_to_text",
    "egoschema_local_doc_to_visual",
    "egoschema_process_results_generation",
]
