from observations.common import infer_model_family, resolve_model_name_or_path


def auto_load_model(model_path: str, device_map="auto", attn_implementation="flash_attention_2"):
    """Auto-dispatch model loading based on model family."""
    family = infer_model_family(model_path)
    resolved = resolve_model_name_or_path(model_path)
    print(f"[Run] Detected model family: {family}")
    if family == "kimi":
        from src.base.models.kimi import load_model
        return load_model(resolved, device_map=device_map, attn_implementation=attn_implementation)
    elif family == "qwen3":
        from src.base.models.qwen3 import load_model
        return load_model(resolved, device_map=device_map, attn_implementation=attn_implementation)
    elif family == "internvl":
        from src.base.models.internvl import load_model
        return load_model(resolved, device_map=device_map, attn_implementation=attn_implementation)
    else:
        raise ValueError(f"Unsupported model family: {family}")
