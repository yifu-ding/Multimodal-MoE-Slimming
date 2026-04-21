import torch
from loguru import logger

try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
except Exception:
    logger.warning("DeepSeek-VL loading dependencies are not available.")
    AutoModel = None
    AutoModelForCausalLM = None
    AutoProcessor = None

try:
    from deepseek_vl2.models import DeepseekVLV2Processor
    _DEEPSEEK_VL2_IMPORT_ERROR = None
except Exception as exc:
    DeepseekVLV2Processor = None
    _DEEPSEEK_VL2_IMPORT_ERROR = exc


def _set_special_token_tensor(model, processor) -> None:
    token_ids = torch.tensor(processor.tokenizer.all_special_ids)
    if hasattr(model, "model"):
        model.model.special_token_id_tensor = token_ids
    elif hasattr(model, "language"):
        model.language.special_token_id_tensor = token_ids
    else:
        model.special_token_id_tensor = token_ids


def load_model(
    model_path: str,
    attn_implementation: str = "flash_attention_2",
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    if AutoProcessor is None or AutoModelForCausalLM is None or AutoModel is None:
        raise ImportError("DeepSeek-VL loading requires transformers AutoModel/AutoProcessor support.")

    load_errors = []

    # DeepSeek-VL2 checkpoints require the official deepseek_vl2 package.
    # Falling back to AutoModel is not reliable because transformers may not
    # recognize `deepseek_vl_v2` model_type.
    model_path_lower = str(model_path).lower()
    looks_like_vl2 = "deepseek-vl2" in model_path_lower or "deepseek_vl2" in model_path_lower
    if looks_like_vl2 and DeepseekVLV2Processor is None:
        raise RuntimeError(
            "DeepSeek-VL2 model loading requires `deepseek_vl2` in the current Python environment, "
            "but importing `deepseek_vl2.models.DeepseekVLV2Processor` failed. "
            f"Import error: {_DEEPSEEK_VL2_IMPORT_ERROR!r}. "
            "Please install DeepSeek-VL2 dependencies into the same env used by this script."
        )

    if DeepseekVLV2Processor is not None:
        try:
            processor = DeepseekVLV2Processor.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            model.eval()
            _set_special_token_tensor(model, processor)
            return model, processor
        except Exception as exc:
            load_errors.append(f"official_deepseek_vl2: {exc}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    common_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    if attn_implementation is not None:
        common_kwargs["attn_implementation"] = attn_implementation

    for loader in (AutoModelForCausalLM, AutoModel):
        try:
            model = loader.from_pretrained(model_path, **common_kwargs)
            model.eval()
            _set_special_token_tensor(model, processor)
            return model, processor
        except Exception as exc:
            load_errors.append(f"{loader.__name__}: {exc}")

    raise RuntimeError(
        "Failed to load DeepSeek-VL model. Tried AutoModelForCausalLM and AutoModel. "
        + " | ".join(load_errors)
    )
