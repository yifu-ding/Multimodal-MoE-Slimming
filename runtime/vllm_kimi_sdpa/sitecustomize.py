"""Bound Kimi-VL MoonViT SDPA memory when external flash-attn is absent."""

from __future__ import annotations

import importlib.util
import os


if os.environ.get("MAES_EP4_PLAN") or os.environ.get("MAES_MASK_PLAN"):
    # Python imports only the first sitecustomize on PYTHONPATH. Compose the
    # EP4 import hook here before importing any vLLM module for the SDPA patch.
    import maes_ep4_bootstrap  # noqa: F401


def _install_kimi_sdpa_fallback() -> None:
    if importlib.util.find_spec("flash_attn") is not None:
        return

    try:
        import torch
        import torch.nn.functional as F
        from vllm.model_executor.models import moonvit
    except ModuleNotFoundError:
        # ``conda run`` starts helper interpreters outside the target env. They
        # also import sitecustomize, but do not have the target dependencies.
        return

    def segmented_sdpa_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_cu_seqlens: torch.Tensor | None = None,
        k_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_cu_seqlens is None or k_cu_seqlens is None:
            raise ValueError("MoonViT packed SDPA requires cumulative sequence lengths")
        q_bounds = q_cu_seqlens.detach().cpu().tolist()
        k_bounds = k_cu_seqlens.detach().cpu().tolist()
        if len(q_bounds) != len(k_bounds):
            raise ValueError("MoonViT query/key sequence counts differ")

        outputs = []
        for q_start, q_end, k_start, k_end in zip(
            q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]
        ):
            q_part = q[q_start:q_end].transpose(0, 1)
            k_part = k[k_start:k_end].transpose(0, 1)
            v_part = v[k_start:k_end].transpose(0, 1)
            part = F.scaled_dot_product_attention(
                q_part,
                k_part,
                v_part,
                dropout_p=0.0,
                is_causal=False,
            )
            outputs.append(part.transpose(0, 1).reshape(q_end - q_start, -1))
        return torch.cat(outputs, dim=0)

    moonvit.sdpa_attention = segmented_sdpa_attention
    moonvit.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = segmented_sdpa_attention


_install_kimi_sdpa_fallback()
