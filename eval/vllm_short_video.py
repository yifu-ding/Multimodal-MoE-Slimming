import re
from typing import Any

from loguru import logger


_NFRAMES_ERROR = re.compile(
    r"nframes should in interval \[(\d+), (\d+)\], but got (\d+)\."
)


def to_openai_messages_with_nframe_fallback(
    chat_messages: Any,
    video_kwargs: dict[str, Any],
) -> list[dict]:
    """Retry short videos with the largest valid even frame count."""
    adjusted_kwargs = dict(video_kwargs)
    for _ in range(8):
        try:
            return chat_messages.to_openai_messages(video_kwargs=adjusted_kwargs)
        except ValueError as exc:
            match = _NFRAMES_ERROR.search(str(exc))
            if match is None or "nframes" not in adjusted_kwargs:
                raise
            min_frames, total_frames, requested_frames = map(int, match.groups())
            valid_frames = min(total_frames, requested_frames)
            valid_frames -= valid_frames % 2
            valid_frames = max(min_frames, valid_frames)
            if valid_frames == adjusted_kwargs["nframes"]:
                raise
            logger.warning(
                "Short video has only {} frames; reducing nframes from {} to {}.",
                total_frames,
                adjusted_kwargs["nframes"],
                valid_frames,
            )
            adjusted_kwargs["nframes"] = valid_frames
    raise RuntimeError("Unable to find a valid nframes value after 8 retries.")
