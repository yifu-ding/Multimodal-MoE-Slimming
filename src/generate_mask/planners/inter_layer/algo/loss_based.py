import torch

from src.base.shared_utils import _print


def smooth_layerwise_loss(
    layerwise_loss: torch.Tensor,
    smooth_times: int = 0,
    *,
    power: float = 0.5,
) -> torch.Tensor:
    if layerwise_loss is None:
        return None
    out = layerwise_loss.clone().float().clamp_min(0.0)
    for _ in range(smooth_times):
        out = torch.pow(out, power)
        mean = out.mean()
        std = out.std()
        out = out.clamp(min=mean - std, max=mean + std)
    denom = out.mean().clamp_min(1e-6)
    return out / denom


def smooth_layerwise_loss_with_fn(
    layerwise_loss: torch.Tensor,
    smooth_times: int = 0,
    smooth_fn: str = "sqrt",
) -> torch.Tensor:
    smooth_fn = (smooth_fn or "sqrt").lower()
    if smooth_fn == "raw":
        return layerwise_loss.clone().float()
    if smooth_fn == "sqrt":
        return smooth_layerwise_loss(layerwise_loss, smooth_times=smooth_times, power=0.5)
    if smooth_fn == "cbrt":
        return smooth_layerwise_loss(layerwise_loss, smooth_times=smooth_times, power=1.0 / 3.0)
    if smooth_fn == "fourth_root":
        return smooth_layerwise_loss(layerwise_loss, smooth_times=smooth_times, power=0.25)
    if smooth_fn == "log":
        out = torch.log1p(layerwise_loss.clone().float().clamp_min(0.0))
        return out / out.mean().clamp_min(1e-6)
    raise ValueError(f"Unsupported smooth_fn: {smooth_fn}")


def loss_based_importance_keep_plan(
    layerwise_loss,
    p_target,
    L: int,
    tol: float = 1e-5,
    smooth_times: int = 0,
    smooth_fn: str = "sqrt",
    verbose: bool = False,
):
    assert layerwise_loss.shape == (L,), f"layerwise_loss must have shape ({L},)"
    smoothed = smooth_layerwise_loss_with_fn(
        layerwise_loss,
        smooth_times=smooth_times,
        smooth_fn=smooth_fn,
    )
    smoothed = smoothed.clamp_min(0.0)
    denom = smoothed.sum().clamp_min(1e-6)
    total_keep_ratio = (1.0 - float(p_target)) * L
    layerwise_keep_ratio = total_keep_ratio * (smoothed / denom)
    layerwise_keep_ratio = layerwise_keep_ratio.clamp(0.0, 1.0)

    diff = abs(float(layerwise_keep_ratio.mean().item()) - (1.0 - float(p_target)))
    if diff > tol and verbose:
        _print(
            "[loss_based_importance_keep_plan] keep mean differs from target: "
            f"diff={diff:.6f}"
        )
    if verbose:
        _print(f"[loss_based_importance_keep_plan] {layerwise_keep_ratio}")
    return layerwise_keep_ratio
