from collections import defaultdict
import torch
import math
import numpy as np
from src.base.shared_utils import _print

def build_masks_layerwise(scores,   # scores[layer][expert] = 1D torch.Tensor[D] 本层每个专家打分 , scores 可以是 tensor 也可以是 dict
                             keep_ratio=None, 
                             top_k=None, 
                             L=None, 
                             E=None, 
                             I=None,
                             verbose=False,
                             ) -> tuple[torch.Tensor, torch.Tensor]:

    masks = torch.zeros((L, E, I), dtype=torch.float32).to(scores.device)
    K_E = torch.zeros(L, E, dtype=torch.int64)

    if isinstance(keep_ratio, (float, int)):
        keep_ratio = torch.tensor([keep_ratio for _ in range(L)]).clamp(max=1.0)

    assert len(keep_ratio) == L, "keep_ratio must be a list or tensor with length L"

    if verbose:
        _print(f"Building masks layerwise keep_ratio={keep_ratio}")
        
    for lid in range(L):
        sizes = torch.tensor([I for _ in range(E)], dtype=torch.long)
        D = int(sizes.sum().item())
        # 2) 计算该层要保留的维度数量
        if top_k is not None:
            k = min(top_k, D)
        else:
            k = max(1, int(math.ceil(D * keep_ratio[lid])))  # 该层要保留的维度数量
        # 3) 拼接该层所有 expert 的打分向量（按 eids 顺序）
        flat = torch.cat([scores[lid][e].reshape(-1) for e in range(E)], dim=0)  # [D]

        # 4) 选出全层 Top-k 的全局索引
        top_idx = torch.topk(flat, k=k, largest=True).indices  # [k]

        # 5) 将全局索引回映到各 expert 的局部通道索引
        #    cum = [0, size0, size0+size1, ...]，使用 bucketize 定位所属 expert
        device = flat.device
        cum = torch.cat([torch.tensor([0], dtype=torch.long), torch.cumsum(sizes, dim=0)]).to(device)  # [n_expert+1]
        owner = torch.bucketize(top_idx, cum[1:], right=True).to(device)                 # [k]，每个 top_idx 属于哪个 expert (0..n-1)
        local = top_idx - cum[:-1][owner]                              # [k]，该 expert 内的局部通道索引

        # 6) 构建 masks[lid, eid]，逐 expert 设置 0/1
        for e in range(E):
            masks[lid, e] = torch.zeros(I, dtype=torch.float32).to(device)

        # 向量化填充：每个 expert 内将被选中的局部索引置 1
        for e in range(E):
            selected = local[owner == e]
            if selected.numel() > 0:
                masks[lid, e].index_fill_(0, selected, 1.0)
                K_E[lid, e] = selected.numel()
    
                # import ipdb; ipdb.set_trace()
                # t = scores[lid][e][selected]
                # print(f"t: {t.min()}")
    
    return masks, K_E