import numpy as np

def quadratic_u_schedule(
    L: int,
    keep_global: float = 0.5,   # 全局平均保留率
    beta: float = 0.35,         # 曲率/对比度，越大两端越高（建议 0.2 ~ 0.6）
    center: float = 0.5,        # U 型中心（0~1），默认居中
    rmin: float | None = None,  # 每层下限（绝对值），如 0.2；None 表示不设
    rmax: float | None = None,  # 每层上限（绝对值），如 0.8；None 表示不设
    renorm: bool = True,        # 夹紧后是否再归一到 keep_global 的均值
    verbose: bool = False,
) -> np.ndarray:
    """
    返回 shape=(L,) 的 per-layer keep_ratio: 
      r[l] ≈ keep_global + beta * std_normalized((x-center)^2)
    先生成U形, 再按 rmin/rmax 夹紧, 最后可选地重新归一到全局均值。
    """
    # assert L > 0 and 0.0 < keep_global < 1.0
    x = np.linspace(0.0, 1.0, L)
    base = (x - center) ** 2
    # 标准化基底到零均值、单位方差，便于用 beta 控制对比度
    base = (base - base.mean()) / (base.std() + 1e-12)

    # 生成初始曲线（开口向上）
    r = keep_global + beta * keep_global * base
    # 边界夹紧
    if rmin is not None:
        r = np.maximum(r, rmin)
    if rmax is not None:
        r = np.minimum(r, rmax)

    if verbose:
        print(f"\t beta={beta}, center={center}, rmax={r.max()}, rmin={r.min()}")
    return r


def u_shaped_keep_plan(p_target: float, beta: float = 0.02, rmin: float = 0.10, center: float = 0.5, L: int = None, tol: float = 1e-5, verbose: bool = False):  
    layerwise_scores = quadratic_u_schedule(L, keep_global=1-p_target, beta=beta, rmin=rmin, center=center, verbose=verbose)
    total_keep_ratio = (1-p_target) * L 
    total_scores = sum(layerwise_scores)
    layerwise_keep_ratio = [total_keep_ratio * (layerwise_scores[lid] / total_scores) for lid in range(L)]
    
    assert abs(sum(layerwise_keep_ratio)/L - p_target) < tol, "u_shaped_keep_plan: layerwise_keep_ratio.mean() - p_target is not close to 0"

    if verbose:
        print(f"\t p_target={p_target}, beta={beta}, rmin={rmin}, center={center}, L={L}")
        print(f"\t u_shaped_keep_plan: {layerwise_keep_ratio}")
    return layerwise_keep_ratio