import torch
from loguru import logger
from src.integrations import (
    configure_fastmmoe_internvl_runtime,
    fastmmoe_enabled,
    prepare_fastmmoe_vendor_imports,
)

try:
    from transformers import AutoProcessor, InternVLForConditionalGeneration
except Exception:
    logger.warning("InternVLForConditionalGeneration is not available.")
    AutoProcessor = None
    InternVLForConditionalGeneration = None


def load_model(
    model_path: str,
    attn_implementation: str = "flash_attention_2",
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    if fastmmoe_enabled():
        prepare_fastmmoe_vendor_imports("internvl")
        from vlmeval.vlm.internvl import InternVLChatModel

        if AutoProcessor is None:
            raise ImportError("InternVL FastMMoE loading requires transformers AutoProcessor.")
        model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        configure_fastmmoe_internvl_runtime(model, processor)
        logger.info("[FastMMoE] Loaded InternVL vendor model")
        return model, processor

    if InternVLForConditionalGeneration is None or AutoProcessor is None:
        try:
            import transformers

            transformers_version = transformers.__version__
        except Exception:
            transformers_version = "unknown"
        raise ImportError(
            "InternVL loading requires a transformers build that provides "
            "`InternVLForConditionalGeneration`. "
            f"Installed transformers version: {transformers_version}."
        )

    model = InternVLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.model.special_token_id_tensor = torch.tensor(
        processor.tokenizer.all_special_ids
    )
    return model, processor
