import torch


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
