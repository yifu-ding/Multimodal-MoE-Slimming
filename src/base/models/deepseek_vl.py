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
except Exception:
    DeepseekVLV2Processor = None


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
