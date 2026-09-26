import torch
import torch.nn.functional as F
from loguru import logger
from src.integrations import (
    fastmmoe_enabled,
    fastmmoe_strategy,
    prepare_fastmmoe_vendor_imports,
)

try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
except Exception:
    logger.warning("DeepSeek-VL loading dependencies are not available.")
    AutoModel = None
    AutoModelForCausalLM = None
    AutoProcessor = None

try:
    from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor
    _DEEPSEEK_VL2_IMPORT_ERROR = None
except Exception as exc:
    DeepseekVLV2Processor = None
    _DEEPSEEK_VL2_IMPORT_ERROR = exc


_DEEPSEEK_VL2_VISION_PATCHED = False
_FLASH_ATTN_GUARD_PATCHED = False


def _guard_against_broken_flash_attn() -> None:
    global _FLASH_ATTN_GUARD_PATCHED
    if _FLASH_ATTN_GUARD_PATCHED:
        return
    try:
        import flash_attn_2_cuda  # noqa: F401
        _FLASH_ATTN_GUARD_PATCHED = True
        return
    except Exception:
        pass

    try:
        import transformers.modeling_utils as modeling_utils

        modeling_utils.is_flash_attn_2_available = lambda: False
    except Exception:
        pass
    try:
        import transformers.utils as tf_utils

        tf_utils.is_flash_attn_2_available = lambda: False
    except Exception:
        pass
    try:
        import transformers.utils.import_utils as import_utils

        import_utils.is_flash_attn_2_available = lambda: False
    except Exception:
        pass
    _FLASH_ATTN_GUARD_PATCHED = True


def _patch_deepseek_vl2_vision_attention() -> None:
    global _DEEPSEEK_VL2_VISION_PATCHED
    if _DEEPSEEK_VL2_VISION_PATCHED:
        return
    _guard_against_broken_flash_attn()
    try:
        import importlib

        siglip_vit = importlib.import_module("deepseek_vl2.models.siglip_vit")
    except Exception as exc:
        logger.warning(f"Skipping DeepSeek-VL2 vision attention patch because siglip_vit import failed: {exc!r}")
        return

    original_forward = siglip_vit.Attention.forward

    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if not self.qk_norm:
            if self.head_dim % 32 == 0 and siglip_vit.is_flash_attn_2_available():
                x = siglip_vit.flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    deterministic=self.deterministic,
                )
            else:
                q, k, v = qkv.unbind(2)
                used_xformers = False
                if q.device.type == "cuda":
                    try:
                        from xformers.ops import memory_efficient_attention

                        x = memory_efficient_attention(
                            q, k, v, p=self.attn_drop.p if self.training else 0.0
                        )
                        used_xformers = True
                    except Exception:
                        used_xformers = False
                if not used_xformers:
                    q = q.permute(0, 2, 1, 3)
                    k = k.permute(0, 2, 1, 3)
                    v = v.permute(0, 2, 1, 3)
                    x = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                    ).permute(0, 2, 1, 3)
            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    siglip_vit.Attention.forward = patched_forward
    _DEEPSEEK_VL2_VISION_PATCHED = True


def _retry_import_deepseek_vl2_processor():
    global DeepseekVLV2Processor, _DEEPSEEK_VL2_IMPORT_ERROR
    if DeepseekVLV2Processor is not None:
        return DeepseekVLV2Processor
    _guard_against_broken_flash_attn()
    try:
        import importlib

        module = importlib.import_module("deepseek_vl2.models.processing_deepseek_vl_v2")
        DeepseekVLV2Processor = module.DeepseekVLV2Processor
        _DEEPSEEK_VL2_IMPORT_ERROR = None
    except Exception as exc:
        _DEEPSEEK_VL2_IMPORT_ERROR = exc
    return DeepseekVLV2Processor


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
    if fastmmoe_enabled():
        prepare_fastmmoe_vendor_imports("deepseek_vl")
        from deepseek_vl2.models import (
            DeepseekVLV2FastMMoEForCausalLM,
            DeepseekVLV2FastVForCausalLM,
            DeepseekVLV2Processor,
            DeepseekVLV2SparseVLMForCausalLM,
        )

        strategy = fastmmoe_strategy()
        if strategy == "sparsevlm":
            model_cls = DeepseekVLV2SparseVLMForCausalLM
        elif strategy == "fastv":
            model_cls = DeepseekVLV2FastVForCausalLM
        else:
            model_cls = DeepseekVLV2FastMMoEForCausalLM

        processor = DeepseekVLV2Processor.from_pretrained(model_path)
        model = model_cls.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()
        _set_special_token_tensor(model, processor)
        logger.info(f"[FastMMoE] Loaded DeepSeek-VL2 with strategy={strategy}")
        return model, processor

    _guard_against_broken_flash_attn()
    _patch_deepseek_vl2_vision_attention()
    _retry_import_deepseek_vl2_processor()

    load_errors = []

    # DeepSeek-VL2 checkpoints require the official deepseek_vl2 package.
    # Falling back to AutoModel is not reliable because transformers may not
    # recognize `deepseek_vl_v2` model_type.
    model_path_lower = str(model_path).lower()
    looks_like_vl2 = "deepseek-vl2" in model_path_lower or "deepseek_vl2" in model_path_lower
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
    elif looks_like_vl2:
        load_errors.append(f"official_deepseek_vl2_import: {_DEEPSEEK_VL2_IMPORT_ERROR!r}")

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
