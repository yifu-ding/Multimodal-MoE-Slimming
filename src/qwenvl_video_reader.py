"""Selective OpenCV video reader for qwen-vl-utils."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch


def install_into(vision_process):
    def read_video_opencv(ele):
        video_path = str(ele["video"])
        if video_path.startswith("file://"):
            video_path = video_path[7:]
        if not Path(video_path).is_file():
            raise FileNotFoundError(video_path)

        started = time.time()
        capture = cv2.VideoCapture(video_path)
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or frame_count <= 0 or video_fps <= 0:
                raise ValueError(
                    f"OpenCV could not read video metadata: path={video_path}, "
                    f"frames={frame_count}, fps={video_fps}"
                )
            start_frame, end_frame, selected_frame_count = vision_process.calculate_video_frame_range(
                ele, frame_count, video_fps
            )
            nframes = vision_process.smart_nframes(
                ele, total_frames=selected_frame_count, video_fps=video_fps
            )
            requested = np.linspace(start_frame, end_frame, nframes).round().astype(np.int64)
            frames = []
            actual_indices = []
            for requested_index in requested.tolist():
                frame = None
                actual_index = requested_index
                for offset in range(17):
                    candidate = max(start_frame, requested_index - offset)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                    ok, decoded = capture.read()
                    if ok and decoded is not None:
                        frame = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        actual_index = candidate
                        break
                if frame is None:
                    raise RuntimeError(
                        f"OpenCV failed to decode frame {requested_index} and its 16 predecessors "
                        f"from {video_path}"
                    )
                frames.append(frame)
                actual_indices.append(actual_index)
        finally:
            capture.release()

        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        sample_fps = nframes / max(selected_frame_count, 1e-6) * video_fps
        vision_process.logger.info(
            f"opencv: video_path={video_path!r}, total_frames={selected_frame_count}, "
            f"video_fps={video_fps}, time={time.time() - started:.3f}s"
        )
        metadata = {
            "fps": video_fps,
            "frames_indices": actual_indices,
            "total_num_frames": selected_frame_count,
            "video_backend": "opencv",
        }
        return video, metadata, sample_fps

    vision_process.VIDEO_READER_BACKENDS["opencv"] = read_video_opencv
