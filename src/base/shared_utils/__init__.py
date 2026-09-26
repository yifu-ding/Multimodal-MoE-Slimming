import torch
import torch.nn.functional as F


def _print(*args, **kwargs):
    print(*args, **kwargs)


def dict_to_tensor(mapping):
    if isinstance(mapping, torch.Tensor):
        return mapping
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected dict or tensor, got {type(mapping)}")
    keys = sorted(mapping.keys())
    values = [mapping[k] for k in keys]
    return torch.stack(
        [
            v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        for v in values
        ],
        dim=0,
    )


def angle_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()
    cos = F.cosine_similarity(pred_f, target_f, dim=-1, eps=eps)
    return 1.0 - cos
