import torch
from src.base.shared_utils import _print
from observation.rebuttal_exp4.smooth_variants import (
    smooth_layerwise_loss_clip,
    smooth_layerwise_loss_huber_style,
    smooth_layerwise_loss_log,
)
# def smooth_scaling(s, tau=1.0, eps=1e-6):
#     # s: [I], non-negative
#     s = torch.log(s + eps)
#     s = (s - s.mean()) / (s.std() + eps)
#     s = torch.softmax(s / tau, dim=-1)
#     return s

def smooth_layerwise_loss(
    layerwise_loss: torch.Tensor,
    smooth_times=0,
    *,
    power: float = 1/4,
) -> torch.Tensor:
    """Repeatedly apply x**power then clamp to [mean-std, mean+std]. Default power=0.5 is sqrt."""
    if layerwise_loss is None:
        return None

    for _ in range(smooth_times):
        layerwise_loss = torch.pow(layerwise_loss, power)  # [L]; e.g. 0.5 sqrt, 1/3 cbrt, 0.25 fourth root
        _std = layerwise_loss.std()
        _mean = layerwise_loss.mean()
        layerwise_loss = layerwise_loss.clamp(min=_mean - _std, max=_mean + _std)

    final_mean = layerwise_loss.mean()
    if final_mean > 0:
        layerwise_loss = layerwise_loss / final_mean
    return layerwise_loss

def smooth_layerwise_loss_with_fn(
    layerwise_loss: torch.Tensor,
    smooth_times: int = 0,
    smooth_fn: str = "sqrt",
) -> torch.Tensor:
    smooth_fn = (smooth_fn or "sqrt").lower()

    if smooth_fn == "sqrt":
        return smooth_layerwise_loss(layerwise_loss, smooth_times=smooth_times)
    if smooth_fn == "log":
        return smooth_layerwise_loss_log(layerwise_loss)
    if smooth_fn == "clip":
        return smooth_layerwise_loss_clip(layerwise_loss)
    if smooth_fn in {"huber", "huber_style"}:
        return smooth_layerwise_loss_huber_style(layerwise_loss)

    raise ValueError(
        f"Unsupported smooth_fn: {smooth_fn}.  "
    )


def loss_based_importance_keep_plan(
    layerwise_loss,
    p_target,
    L: int,
    tol: float = 1e-5,
    smooth_times=0,
    smooth_fn: str = "sqrt",
    verbose=False,
):
    # import ipdb; ipdb.set_trace()
    assert layerwise_loss.shape == (L,), "layerwise_loss must have shape (L,)"
    layerwise_loss = smooth_layerwise_loss_with_fn(
        layerwise_loss,
        smooth_times=smooth_times,
        smooth_fn=smooth_fn,
    )
    loss_sum = sum(layerwise_loss)
    total_keep_ratio = (1 - p_target) * L
    layerwise_keep_ratio = total_keep_ratio * (layerwise_loss / loss_sum)
    try:
        assert abs(sum(layerwise_keep_ratio)/L - (1-p_target)) < tol, "loss_based_importance_keep_plan: layerwise_keep_ratio.mean() - p_target is not close to 0"
    except:
        _print(f"\t loss_based_importance_keep_plan: layerwise_keep_ratio.mean() is not close to 1-p_target, diff={abs(sum(layerwise_keep_ratio)/L - (1-p_target))}")
        return None
    
    try:
        assert all(layerwise_keep_ratio >= 0) and all(layerwise_keep_ratio <= 1), "loss_based_importance_keep_plan: layerwise_keep_ratio is not in [0, 1]"
    except:
        _print(f"\t loss_based_importance_keep_plan: layerwise_keep_ratio is not in [0, 1], {layerwise_keep_ratio}")
        return None
    
    if verbose:
        _print(f"\t loss_based_importance_keep_plan: {layerwise_keep_ratio}")
    return layerwise_keep_ratio
