import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from loguru import logger


QWEN3_VIDEOMME_TASK = "videomme_qwen3_vllm"
_PIXELS_PER_VISUAL_TOKEN = 32 * 32


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _env_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


@dataclass(frozen=True)
class Qwen3VideoMMEConfig:
    fps: float
    max_frames: int
    min_tokens_per_frame: int
    max_tokens_per_frame: int
    total_video_tokens: int

    @classmethod
    def from_env(cls) -> "Qwen3VideoMMEConfig":
        total_video_tokens = _env_int("QWEN3_VIDEOMME_TOTAL_VIDEO_TOKENS", 224000)
        recovery_total_video_tokens = os.getenv(
            "QWEN3_VIDEOMME_RECOVERY_TOTAL_VIDEO_TOKENS"
        )
        if recovery_total_video_tokens is not None:
            total_video_tokens = _env_int(
                "QWEN3_VIDEOMME_RECOVERY_TOTAL_VIDEO_TOKENS",
                total_video_tokens,
            )
        config = cls(
            fps=_env_float("QWEN3_VIDEOMME_FPS", 2.0),
            max_frames=_env_int("QWEN3_VIDEOMME_MAX_FRAMES", 2048),
            min_tokens_per_frame=_env_int("QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME", 128),
            max_tokens_per_frame=_env_int("QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME", 640),
            total_video_tokens=total_video_tokens,
        )
        if config.min_tokens_per_frame > config.max_tokens_per_frame:
            raise ValueError(
                "QWEN3_VIDEOMME_MIN_TOKENS_PER_FRAME cannot exceed "
                "QWEN3_VIDEOMME_MAX_TOKENS_PER_FRAME."
            )
        return config


def is_qwen3_videomme_native_request(model: str, task: str) -> bool:
    return task == QWEN3_VIDEOMME_TASK and "qwen3-vl" in model.lower()


@lru_cache(maxsize=1)
def _load_native_video(video_path: str, config: Qwen3VideoMMEConfig):
    from qwen_vl_utils import process_vision_info

    video_part = {
        "type": "video",
        "video": video_path,
        "fps": config.fps,
        "max_frames": config.max_frames,
        "min_pixels": config.min_tokens_per_frame * _PIXELS_PER_VISUAL_TOKEN,
        "max_pixels": config.max_tokens_per_frame * _PIXELS_PER_VISUAL_TOKEN,
        "total_pixels": config.total_video_tokens * _PIXELS_PER_VISUAL_TOKEN,
    }
    messages = [{"role": "user", "content": [video_part]}]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if image_inputs is not None or not video_inputs:
        raise RuntimeError(f"Expected one native video input for {video_path}.")

    # qwen-vl-utils already applied both spatial token budgets. Resizing the
    # resulting tensor again inside vLLM is the accuracy bug described in #1540.
    video_kwargs = dict(video_kwargs)
    video_kwargs["do_resize"] = False
    first_video = video_inputs[0]
    if isinstance(first_video, tuple):
        first_video = first_video[0]
    logger.info(
        "Qwen3 VideoMME native input: path={} fps={} max_frames={} "
        "max_tokens_per_frame={} total_video_tokens={} sampled_shape={} "
        "do_resize={}",
        video_path,
        config.fps,
        config.max_frames,
        config.max_tokens_per_frame,
        config.total_video_tokens,
        tuple(first_video.shape) if hasattr(first_video, "shape") else "unknown",
        video_kwargs["do_resize"],
    )
    return video_inputs, video_kwargs, video_part


def prepare_qwen3_videomme_input(
    client: Any,
    model: str,
    task: str,
    contexts: str,
    visuals: list[Any],
) -> dict[str, Any] | None:
    if not is_qwen3_videomme_native_request(model, task):
        return None
    if len(visuals) != 1 or not isinstance(visuals[0], str):
        raise ValueError(
            f"{QWEN3_VIDEOMME_TASK} expects exactly one video path, got {visuals!r}."
        )

    config = Qwen3VideoMMEConfig.from_env()
    video_inputs, video_kwargs, video_part = _load_native_video(visuals[0], config)
    messages = [
        {
            "role": "user",
            "content": [video_part, {"type": "text", "text": contexts}],
        }
    ]
    prompt = client.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"video": video_inputs},
        "mm_processor_kwargs": video_kwargs,
    }
