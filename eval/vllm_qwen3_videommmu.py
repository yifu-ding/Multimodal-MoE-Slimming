import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


QWEN3_VIDEOMMMU_TASK = "video_mmmu_local"
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
class Qwen3VideoMMMUConfig:
    fps: float
    max_frames: int
    max_tokens_per_frame: int
    total_video_tokens: int

    @classmethod
    def from_env(cls) -> "Qwen3VideoMMMUConfig":
        return cls(
            fps=_env_float("QWEN3_VIDEOMMMU_FPS", 2.0),
            max_frames=_env_int("QWEN3_VIDEOMMMU_MAX_FRAMES", 2048),
            max_tokens_per_frame=_env_int("QWEN3_VIDEOMMMU_MAX_TOKENS_PER_FRAME", 768),
            total_video_tokens=_env_int("QWEN3_VIDEOMMMU_TOTAL_VIDEO_TOKENS", 224000),
        )


def is_qwen3_videommmu_native_request(model: str, task: str) -> bool:
    return task.startswith("video_mmmu_") and "qwen3-vl" in model.lower()


@lru_cache(maxsize=1)
def _load_native_video(video_path: str, config: Qwen3VideoMMMUConfig):
    from qwen_vl_utils import process_vision_info

    video_part = {
        "type": "video",
        "video": video_path,
        "fps": config.fps,
        "max_frames": config.max_frames,
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

    # qwen-vl-utils has already applied the per-frame and total token budgets.
    video_kwargs = dict(video_kwargs)
    video_kwargs["do_resize"] = False
    return video_inputs, video_kwargs, video_part


def prepare_qwen3_videommmu_input(
    client: Any,
    model: str,
    task: str,
    contexts: str,
    visuals: list[Any],
) -> dict[str, Any] | None:
    if not is_qwen3_videommmu_native_request(model, task):
        return None
    if len(visuals) != 1 or not isinstance(visuals[0], str):
        raise ValueError(
            f"{QWEN3_VIDEOMMMU_TASK} expects exactly one video path, got {visuals!r}."
        )

    config = Qwen3VideoMMMUConfig.from_env()
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
