import json
import os
from pathlib import Path

from lmms_eval.tasks.videommmu.utils import get_cache_dir


def video_mmmu_doc_to_visual_local(doc):
    root = Path(os.environ.get("VIDEO_MMMU_ROOT", "/home/data/dyf/hf_cache/datasets/VideoMMMU"))
    subject = "_".join(doc["id"].split("_")[1:-1])
    video_path = root / get_cache_dir(subject) / f"{doc['id']}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video-MMMU media is missing: {video_path}")
    media_log = os.environ.get("VIDEO_MMMU_MEDIA_LOG")
    if media_log:
        line = json.dumps({"id": doc["id"], "media_path": str(video_path)}) + "\n"
        descriptor = os.open(media_log, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)
    return [str(video_path)]
