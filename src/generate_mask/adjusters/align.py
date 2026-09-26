import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .utils import _largest_remainder_alloc
from time import time
from src.base.shared_utils import _print

@torch.no_grad()
def adjust_with_align(
    K_E: torch.Tensor,                      # [L, E] int64
    mat: dict,                   # [L, E, I] float32
    D_layer: Dict[int, int],               # {layer_idx: layer_cap}
    cap_per_expert: int = 768,
    align: int = 64,                       # 取 64 或 128
    min_per_expert: int = 0,               # 若需要启动下限，建议给成 64 或 128 的倍数
    layers: Optional[List[int]] = None,
    prefer_keep_active: bool = False,       # 回收时尽量不让已激活掉到 0
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    将每个专家通道数对齐到 align 的倍数，并尽量保持总体不变。
    同时满足：
      0 <= K[l,e] <= cap_per_expert
      sum_e K[l,e] <= D_layer[l]
      K[l,e] == 0 或 K[l,e] >= min_per_expert (如启用) 
      K[l,e] % align == 0
    返回:
      K_aligned, info 字典：{'target_total','final_total','added','removed','align','unmet_diff'}
    """
    start_time = time()
    
    assert K_E.dtype == torch.int64
    L, E = K_E.shape
    if layers is None:
        layers = sorted(D_layer.keys())
    assert len(layers) == L
    device = K_E.device

    caps_e = torch.full((E,), int(cap_per_expert), dtype=torch.int64, device=device)
    layer_caps_vec = torch.tensor([int(D_layer[l]) for l in layers], dtype=torch.int64, device=device)

    # 原目标总量
    target_total = int(K_E.sum().item())

    # 对齐下限
    if min_per_expert > 0:
        # 保证是 align 的倍数
        if min_per_expert % align != 0:
            min_per_expert = ((min_per_expert + align - 1) // align) * align

    # 0) 逐点先 clamp 到专家上限
    K = torch.clamp(K_E.clone(), min=torch.zeros_like(caps_e), max=caps_e)

    # 1) 基线向下对齐到 align 的倍数，并处理启用下限
    #    对每个槽位：base = floor(K/align)*align
    base = (K // align) * align

    # 若开启下限：把 0 < base < min_per_expert 归零
    if min_per_expert > 0:
        mask_low = (base > 0) & (base < min_per_expert)
        base[mask_low] = 0

    # 分层容量约束的基线修正：若某层 base 和超过层 cap，按块回收
    for i in range(L):
        row = base[i]
        cap = int(layer_caps_vec[i].item())
        s = int(row.sum().item())
        if s <= cap:
            continue
        # 需要回收的块数 (按 align 计) 
        over = s - cap
        over_chunks = (over + align - 1) // align  # 按块回收，至少回收到不超过cap
        # 优先从“离 0 最近”的位置回收，即数值较小的已激活项
        candidates = torch.nonzero(row > 0, as_tuple=False).flatten()
        # 排序：小的优先回收；若开启 prefer_keep_active，可再按与 min 的距离排序
        if prefer_keep_active and min_per_expert > 0:
            key = row[candidates] - min_per_expert
        else:
            key = row[candidates]
        order = candidates[torch.argsort(key)]  # 由小到大
        j = 0
        while over_chunks > 0 and j < order.numel():
            idx = int(order[j].item())
            if row[idx] >= align:
                # 回收 1 块
                new_val = row[idx] - align
                if min_per_expert > 0 and new_val > 0 and new_val < min_per_expert:
                    # 若会落到 0<val<min，则直接回收到 0
                    over_chunks -= int((row[idx] + align - 1) // align)  # 全部清空所需的块
                    row[idx] = 0
                else:
                    row[idx] = new_val
                    over_chunks -= 1
            j += 1
        base[i] = row

    base_total = int(base.sum().item())
    base_time = time()

    # 2) 计算需要“加块”还是“减块”
    diff = target_total - base_total
    # 只能以 align 为粒度调整
    # 如果 diff 不是 align 的倍数，则最接近可达的是 diff_adj = floor(diff/align)*align 或 ceil
    # 这里尽量选择最接近 0 的调整方向，同时满足层与专家约束
    def round_to_chunks(x: int, step: int) -> int:
        # 返回整块数
        return int(np.round(x / step))

    need_chunks = diff // align  # 向下取整的块数
    # 如果 diff 不是整块，选择最接近的方向：|diff - need_chunks*align| vs |diff - (need_chunks+1)*align|
    if diff % align != 0:
        cand1 = need_chunks
        cand2 = need_chunks + 1 if diff > 0 else need_chunks - 1
        e1 = abs(diff - cand1 * align)
        e2 = abs(diff - cand2 * align)
        if e2 < e1:
            need_chunks = cand2

    # 3) 按块“加”或“减”，同时考虑层与专家 headroom
    added_chunks = 0
    removed_chunks = 0

    # 预计算各层 headroom (按值) ，再转为块数
    def layer_headroom_chunks(curr: torch.Tensor) -> torch.Tensor:
        used = curr.sum(dim=1)
        hr_val = torch.clamp(layer_caps_vec - used, min=torch.zeros_like(layer_caps_vec))
        return hr_val // align

    # 每槽位最大还能加的块数
    def slot_headroom_chunks(curr_row: torch.Tensor) -> torch.Tensor:
        hr_e = torch.clamp(caps_e - curr_row, min=torch.zeros_like(caps_e))
        return hr_e // align

    # 计算“欲望”分数：倾向于把 base 补到最靠近原始 K 的 ceil 对齐
    frac = (K % align).to(torch.int64)  # 每槽位对齐的“小数部分”
    desire = frac  # 分数越大越应该优先获得 1 块补偿
    
    compute_slot_headroom_time = time() 

    # 3.1 增加路径
    if need_chunks > 0:
        remain = need_chunks
        # 先分层，层预算为 headroom 的块数
        while remain > 0:
            layer_hr = layer_headroom_chunks(base)
            if int(layer_hr.sum().item()) == 0:
                break  # 无法继续加
            # 分层按照 headroom 比例的最大余数法
            proportional = (layer_hr.double() * (remain / max(int(layer_hr.sum().item()), 1))).to(torch.float64)
            layer_add = _largest_remainder_alloc(proportional, caps=layer_hr, target=remain).to(torch.int64)

            # 层内分配：两阶段
            for i in range(L):
                c_i = int(layer_add[i].item())
                if c_i <= 0:
                    continue

                row = base[i]
                row_hr_chunks = slot_headroom_chunks(row)

                # 第一阶段：只在“已激活或可直接升到 min”的槽位里补
                # 候选1：已激活的槽位
                active = row > 0
                # 候选2：当前为 0 但原始 K >= min_per_expert，且至少能给到 min_per_expert 的块数
                can_new = torch.zeros_like(row, dtype=torch.bool)
                if min_per_expert > 0:
                    need_min_chunks = min_per_expert // align
                    can_new = (row == 0) & (K[i] >= min_per_expert) & (row_hr_chunks >= need_min_chunks)

                cand = torch.where(active | can_new, torch.ones_like(row, dtype=torch.bool), torch.zeros_like(row, dtype=torch.bool))
                if cand.any():
                    # 候选的 desirability：优先把 frac 大的补齐
                    scores = torch.where(cand, desire[i], torch.full_like(row, -1))
                    order = torch.argsort(scores, descending=True)

                    j = 0
                    while c_i > 0 and j < order.numel():
                        idx = int(order[j].item())
                        if not cand[idx]:
                            j += 1
                            continue
                        if row_hr_chunks[idx] <= 0:
                            j += 1
                            continue

                        if row[idx] == 0 and min_per_expert > 0:
                            # 直接激活到 min
                            need_chunks_here = min(min_per_expert // align, int(row_hr_chunks[idx].item()))
                            give = min(need_chunks_here, c_i)
                            if give == 0:
                                j += 1
                                continue
                            row[idx] += give * align
                            c_i -= give
                            added_chunks += give
                        else:
                            # 已激活，补 1 块
                            row[idx] += align
                            c_i -= 1
                            added_chunks += 1
                        # 更新 per-slot headroom
                        row_hr_chunks = slot_headroom_chunks(row)
                        j += 1

                # 第二阶段：若仍有块未分配，放宽到任意有 headroom 的槽位，按 headroom 大小分配
                if c_i > 0:
                    row_hr_chunks = slot_headroom_chunks(row)
                    if int(row_hr_chunks.sum().item()) > 0:
                        order = torch.argsort(row_hr_chunks, descending=True)
                        j = 0
                        while c_i > 0 and j < order.numel():
                            idx = int(order[j].item())
                            if row_hr_chunks[idx] <= 0:
                                j += 1
                                continue
                            row[idx] += align
                            c_i -= 1
                            added_chunks += 1
                            row_hr_chunks[idx] -= 1
                            j += 1

                base[i] = row
                remain = int(remain - layer_add[i].item() + c_i)  # 未用完的块返还到 remain

            # 继续外层 while 循环，直到 remain 用尽或没有 headroom

    # 3.2 减少路径
    if need_chunks < 0:
        need = -need_chunks
        remain = need
        # 优先从“最不需要”的位置回收：分两阶段
        while remain > 0:
            any_pos = torch.nonzero(base > 0, as_tuple=False)
            if any_pos.numel() == 0:
                break

            # 第一阶段：优先回收那些“并未接近原始 K 的 ceil”的槽位
            # 定义回收分数：frac 小的优先回收；若 prefer_keep_active，尽量不让落到 min 以下，如不可避免则选值最小的
            scores = []
            indices = []
            for i in range(L):
                row = base[i]
                cand = torch.nonzero(row > 0, as_tuple=False).flatten()
                for idx in cand.tolist():
                    v = int(row[idx])
                    # 若回收 1 块会落入 0<val<min，则回收到 0
                    penalty = 0
                    if min_per_expert > 0:
                        if v - align > 0 and v - align < min_per_expert:
                            penalty = 2  # 增加一点惩罚，尽量晚回收
                    score = penalty * 10 + int(align - (desire[i, idx].item()))  # desire 越小越先回收
                    scores.append(score)
                    indices.append((i, idx))
            if len(indices) == 0:
                break
            order = np.argsort(np.array(scores))

            j = 0
            while remain > 0 and j < len(order):
                i, idx = indices[order[j]]
                v = int(base[i, idx].item())
                if v <= 0:
                    j += 1
                    continue
                # 执行回收
                new_v = v - align
                if min_per_expert > 0 and new_v > 0 and new_v < min_per_expert:
                    # 回收到 0
                    base[i, idx] = 0
                    removed = v // align  # 实际回收块数
                    removed_chunks += removed
                    remain -= removed
                else:
                    base[i, idx] = new_v
                    removed_chunks += 1
                    remain -= 1
                j += 1
            # 直到 remain 为 0 或无可回收
    
    allocate_chunks_time = time()
    
    # 4) 最终保险：逐层不超过 cap，逐点不超过专家 cap，满足对齐与下限
    K_final = torch.clamp(base, min=torch.zeros_like(caps_e), max=caps_e)
    if min_per_expert > 0:
        K_final[(K_final > 0) & (K_final < min_per_expert)] = 0
    # 再做一次层 cap 保险 (理论上此时不应超过) 
    for i in range(L):
        row = K_final[i]
        cap = int(layer_caps_vec[i].item())
        s = int(row.sum().item())
        if s > cap:
            over = s - cap
            over_chunks = (over + align - 1) // align
            # 从值小的先回收
            cand = torch.nonzero(row > 0, as_tuple=False).flatten()
            order = cand[torch.argsort(row[cand])]
            j = 0
            while over_chunks > 0 and j < order.numel():
                idx = int(order[j].item())
                v = int(row[idx].item())
                if v <= 0:
                    j += 1
                    continue
                nv = v - align
                if min_per_expert > 0 and nv > 0 and nv < min_per_expert:
                    row[idx] = 0
                    over_chunks -= int((v + align - 1) // align)
                else:
                    row[idx] = nv
                    over_chunks -= 1
                j += 1
            K_final[i] = row

    final_total = int(K_final.sum().item())
    total_time = time() 
    
    info = {
        "method": "largest_channel",
        "align": int(align),
        "min_per_expert": int(min_per_expert),
        "target_total": int(target_total),
        "final_total": int(final_total),
        "added": int(added_chunks * align),
        "removed": int(removed_chunks * align),
        "unmet_diff": int(final_total - target_total),  # 若非 0，则说明精确保持不可达
    }
    
    if verbose:
        _print("[Time Summary] align time summary ")
        _print(f"    base time: {(base_time - start_time) * 1000:.2f}ms")
        _print(f"    compute_slot_headroom time: {(compute_slot_headroom_time - base_time) * 1000:.2f}ms")
        _print(f"    allocate_chunks time: {(allocate_chunks_time - compute_slot_headroom_time) * 1000:.2f}ms")
        _print(f"    finalize_clamp_and_cap time: {(total_time - allocate_chunks_time) * 1000:.2f}ms")
        _print(f"    total time: {(total_time - start_time) * 1000:.2f}ms")
    
    # 根据 K_final 和 score 制作 mask
    I = cap_per_expert
    masks = torch.zeros((L, E, I), dtype=torch.bool, device=K_final.device)
    for lid in range(L):
        for eid in range(E):
            score_vec = mat[lid][eid].to(device=K_final.device)
            topk_idx = torch.topk(score_vec, k=K_final[lid][eid], largest=True).indices
            masks[lid][eid].index_fill_(0, topk_idx, True)
    
    return K_final, masks, info

