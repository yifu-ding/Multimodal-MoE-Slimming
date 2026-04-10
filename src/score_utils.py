"""Per-channel importance scoring utilities.

Ported from LLM-Distillation/src/calibration/channel_scoring/collector/utils.py.
Pure tensor math — no model-specific dependencies.
"""
import torch
import torch.nn as nn


def weight_rms(weight: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
    """L2 norm per output channel.

    weight: [..., I, ...], channel_dim indicates which axis is I.
    Returns: [I]
    """
    x = weight.detach().float()
    reduce_dims = [d for d in range(x.ndim) if d != channel_dim]
    return x.pow(2).sum(dim=reduce_dims).sqrt()


def channel_rms(act: torch.Tensor) -> torch.Tensor:
    """RMS over all non-channel (token/batch) dimensions.

    act: [..., I]  (last dim = channels)
    Returns: [I]
    """
    x = act.detach().float()
    dims = tuple(range(x.dim() - 1))
    return x.pow(2).sum(dim=dims).sqrt()


def channel_saliency(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Mean |act * grad| per channel.

    act, grad: [..., I]
    Returns: [I]
    """
    s = (act * grad).abs().detach().float()
    dims = tuple(range(s.dim() - 1))
    return s.mean(dim=dims)


def safe_add_with_ema(
    target: torch.Tensor,
    ema: float,
    value: torch.Tensor,
) -> torch.Tensor:
    """EMA update: target = target * ema + value * (1 - ema).

    If target is None, initialise from value (clone).
    Returns the updated tensor.
    """
    value = value.detach()
    if target is None:
        return value.clone()
    return target.mul_(ema).add_(value, alpha=1.0 - ema)
