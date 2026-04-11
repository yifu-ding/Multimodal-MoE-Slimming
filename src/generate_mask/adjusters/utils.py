import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

def _largest_remainder_alloc(ideals: torch.Tensor, caps: torch.Tensor, target: int) -> torch.Tensor:
    """
    标准最大余数法分配整数，受 caps 约束。
    ideals: float64 非负，和为期望值
    caps: int64 非负，逐元素上界
    target: 需要分配的总整数
    返回:
      int64, sum 等于 min(target, sum(caps))，且尽量逼近 ideals。
    """
    assert ideals.dtype in (torch.float64, torch.float32)
    assert caps.dtype == torch.int64
    n = ideals.numel()
    base = torch.floor(ideals).to(torch.int64)
    base = torch.minimum(base, caps)
    s = int(base.sum().item())
    remain = min(target, int(caps.sum().item())) - s
    if remain <= 0:
        return base

    frac = (ideals - base.to(ideals.dtype))
    # 不能超过 cap 的位置不再参与
    mask = (base < caps)
    # 对不可分配位置打分设为 -inf
    scores = torch.where(mask, frac, torch.full_like(frac, -1e9))
    order = torch.argsort(scores, descending=True)
    res = base.clone()
    j = 0
    while remain > 0 and j < n:
        idx = int(order[j].item())
        if res[idx] < caps[idx]:
            res[idx] += 1
            remain -= 1
        j += 1
    return res


