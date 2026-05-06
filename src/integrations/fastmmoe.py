import os
import sys
from typing import Optional

import torch
from loguru import logger


FASTMMOE_VENDOR_ROOT = "/home/dyf/code/distill/comparison_baselines/vendors/FastMMoE"
FASTMMOE_DEEPSEEK_ROOT = os.path.join(FASTMMOE_VENDOR_ROOT, "DeepSeek-VL2")
FASTMMOE_INTERNVL_ROOT = os.path.join(
    FASTMMOE_VENDOR_ROOT, "Internvl3_5", "VLMEvalKit"
)


def fastmmoe_enabled() -> bool:
    return os.getenv("FASTMMOE_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}


def fastmmoe_strategy() -> str:
    return os.getenv("TOKEN_MERGE_STRATEGY", "fastmmoe")


def _prepend_sys_path(path: str) -> None:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _purge_modules(prefix: str) -> None:
    stale = [name for name in sys.modules if name == prefix or name.startswith(prefix + ".")]
    for name in stale:
        sys.modules.pop(name, None)


def prepare_fastmmoe_vendor_imports(family: str) -> None:
    if family == "deepseek_vl":
        if not os.path.isdir(FASTMMOE_DEEPSEEK_ROOT):
            raise FileNotFoundError(f"FastMMoE DeepSeek vendor root not found: {FASTMMOE_DEEPSEEK_ROOT}")
        _purge_modules("deepseek_vl2")
        _prepend_sys_path(FASTMMOE_DEEPSEEK_ROOT)
        return
    if family == "internvl":
        if not os.path.isdir(FASTMMOE_INTERNVL_ROOT):
            raise FileNotFoundError(f"FastMMoE InternVL vendor root not found: {FASTMMOE_INTERNVL_ROOT}")
        _purge_modules("vlmeval")
        _prepend_sys_path(FASTMMOE_INTERNVL_ROOT)
        return
    raise ValueError(f"FastMMoE is not supported for family={family}")


def _resolve_internvl_img_context_token_id(processor) -> Optional[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    except Exception:
        return None
    if token_id is None:
        return None
    try:
        token_id = int(token_id)
    except Exception:
        return None
    return None if token_id < 0 else token_id


def configure_fastmmoe_internvl_runtime(model, processor) -> None:
    special_ids = torch.tensor(processor.tokenizer.all_special_ids)
    model.special_token_id_tensor = special_ids
    if hasattr(model, "language_model"):
        model.language_model.special_token_id_tensor = special_ids
    img_context_token_id = _resolve_internvl_img_context_token_id(processor)
    if img_context_token_id is not None:
        model.img_context_token_id = img_context_token_id
        logger.info(f"[FastMMoE] InternVL img_context_token_id set to {img_context_token_id}")
