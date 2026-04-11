import torch
from src.base.shared_utils import _print

# ---------- 主函数：层级与专家均为可变剪枝率 ----------
def build_masks_globally(
    scores,                     # scores[layer][expert] = 1D torch.Tensor[D] 本层每个专家打分 , scores 可以是 tensor 也可以是 dict
    global_keep_ratio: float = 0.5,        # 全模型保留比例 (0~1) 
    L=None, 
    E=None, 
    I=None,
    verbose=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ---- 统计结构维度（来自 scores_mat 的真实可剪对象）----
    layers = list(range(L))
    D_layer = {lid: E * I for lid in layers}                   # 每层总通道数

    assert isinstance(global_keep_ratio, (float, int)), "global_keep_ratio must be a float or int"
  
    sum_dims = sum(D_layer.values())
    assert sum_dims > 0, "No channels found."
    
    wE = torch.zeros((L, E), dtype=torch.float32)
    for lid in range(L):
        for eid in range(E):
            wE[lid, eid] = sum(scores[lid][eid])
    wE = wE / wE.sum()  # [L, E] 归一化 每层占比

    K_total = int(round(global_keep_ratio * sum_dims))   # 全模型保留通道数
    K_E = torch.floor(wE * K_total).to(torch.int64)  # [L, E] 整数化

    if verbose:
        _print(f"[build_masks_globally] K_E: {K_E}")
        
    masks = torch.zeros((L, E, I), dtype=torch.float32).to(scores.device)

    for lid in len(scores):
        layer_scores = scores[lid]
        for eid in range(E):
            expert_scores = layer_scores[eid]
            top_idx = torch.topk(expert_scores, k=K_E[lid][eid], largest=True).indices  # [k]
            masks[lid, eid] = torch.zeros(I, dtype=torch.float32).to(scores.device)
            masks[lid, eid].index_fill_(0, top_idx, 1.0)

    return masks, K_E
