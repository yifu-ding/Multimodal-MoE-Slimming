import torch
from loguru import logger

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
