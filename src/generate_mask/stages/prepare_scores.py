import csv
import os
import json
import re
import torch
from typing import List, Dict, Optional, Tuple, Any, Union
from src.base.shared_utils.dict_to_tensor import dict_to_tensor
from src.generate_mask.planners import inter_layer_planner, intra_layer_planner
from src.base.shared_utils import _print

__all__ = [
    "load_expert_evict_loss",
    "load_channel_scores",
    "load_layerwise_loss",
    "load_attention_head_scores", 
    "prepare_scores"
]

def load_expert_evict_loss(path: str, L: int, E: int) -> torch.Tensor:
    """
    从 CSV 文件加载 loss 数据作为 expertwise_scores。
    
    Args:
        path: CSV 文件路径，应包含 layer_idx, expert_idx, delta_nll 列
        L: 层数
        E: 每层专家数
    
    Returns:
        expertwise_scores: [L, E] 张量
    """
    tensor = torch.zeros((L, E), dtype=torch.float32)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"{path} header is empty.")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if "delta_nll" not in reader.fieldnames:
            raise ValueError(f"{path} must contain delta_nll column.")
        for row in reader:
            lid_raw = row.get("layer_idx")
            eid_raw = row.get("expert_idx")
            val_raw = row.get("delta_nll")
            if lid_raw is None or eid_raw is None or val_raw is None:
                continue
            try:
                lid = int(float(lid_raw.strip()))
                eid = int(float(eid_raw.strip()))
                delta = float(val_raw)
            except (TypeError, ValueError):
                continue
            if not (0 <= lid < L and 0 <= eid < E):
                continue
            delta = max(0.0, delta)
            if delta > tensor[lid, eid]:
                tensor[lid, eid] = delta
    return tensor


def load_layerwise_inter_hidden_prune_ratios(
    json_path: str,
    layerwise_keep_plan: List[float],
    expertwise_scores: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    return_type: str = "tensor", # "tensor" or "dict"
) -> Tuple[Union[torch.Tensor, torch.Tensor], Union[torch.Tensor, torch.Tensor]]:
    """
    从 JSON 文件读取 inter 和 hidden prune ratios。支持三种格式:
    
    格式 1 (Golden 格式 - layerwise):
      {
        "0": {"best_inter_prune_ratio": 0.283, ...},
        ...
      }
      p_inter = best_inter_prune_ratio
      p_hidden = solve_hidden_ratio_from_inter_ratio(prune_ratio, p_inter, clamp=False)
    
    格式 2 (Pareto 格式 - layerwise):
      {
        "0": {
          "best_point": {"p_inter": 0.283391, "p_hidden": 0.52992, ...},
          ...
        },
        ...
      }
      p_inter 和 p_hidden 直接从 best_point 中读取
    
    格式 3 (Greedy 格式 - expertwise):
      {
        "0": {
          "p_inter_vec": [0.37, 0.32, ...],  # 每个专家的 inter prune ratio
          "p_hidden_vec": [0.35, 0.35, ...], # 每个专家的 hidden prune ratio
          ...
        },
        ...
      }
      每层每个专家都有不同的 p_inter 和 p_hidden

    返回:
      - 格式 1/2: layerwise_inter_prune_ratio: [L], layerwise_hidden_prune_ratio: [L]
      - 格式 3: expertwise_inter_prune_ratio: [L, E], expertwise_hidden_prune_ratio: [L, E]
    """
    assert os.path.exists(json_path), f"HI ratio json file {json_path} does not exist"
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # keys 形如 "0","1","2"...，不保证连续时也能 work
    layer_ids = sorted(int(k) for k in d.keys())
    if len(layer_ids) == 0:
        return None, None
    L = layer_ids[-1] + 1

    from src.calibration.hi_ratio_search.common.solve_ratio import solve_hidden_ratio_from_inter_ratio, init_expert_prune_ratio, solve_ratio
    
    # 检测格式：查看第一层的数据结构
    first_layer_data = d[str(layer_ids[0])]
    
    # 格式 3: expertwise vectors (greedy 格式)
    if "p_inter_vec" in first_layer_data and "p_hidden_vec" in first_layer_data:
        # 获取专家数量 E
        E = len(first_layer_data["p_inter_vec"])
        
        if return_type == "tensor":
            inter = torch.zeros((L, E), device=device, dtype=dtype)
            hidden = torch.zeros((L, E), device=device, dtype=dtype)
        else:
            inter = {}
            hidden = {}

        for k_str, v in d.items():
            layer_idx = int(k_str)
            # layer_prune_ratio_csv = float(v["layer_prune_ratio"])
            layer_keep_ratio = layerwise_keep_plan[layer_idx]
            layer_prune_ratio = 1.0 - layer_keep_ratio
            p_inter_vec = v["p_inter_vec"]
            p_hidden_vec = v["p_hidden_vec"]
            expertwise_keep_ratio_layer = expertwise_scores[layer_idx].to(device=device)
            # import ipdb; ipdb.set_trace()
            expertwise_prune_ratio_layer = 1.0 - expertwise_keep_ratio_layer
            # expertwise_prune_ratio = init_expert_prune_ratio(expertwise_scores_layer, layer_prune_ratio)

            # 将列表转换为张量
            inter[layer_idx] = torch.tensor(p_inter_vec, device=device, dtype=dtype)
            hidden[layer_idx] = torch.tensor(p_hidden_vec, device=device, dtype=dtype)
           
            # layer_ratio_sum = inter[layer_idx] + hidden[layer_idx] 
            # inter[layer_idx] = (inter[layer_idx] / layer_ratio_sum) * layer_prune_ratio
            for eid in range(E):
                hidden[layer_idx][eid] = solve_ratio(prune_ratio=expertwise_prune_ratio_layer[eid],
                                                     source_ratio=inter[layer_idx][eid],
                                                     source="inter", 
                                                     target="hidden", 
                                                     clamp=True)
           
        return inter, hidden
    
    # 格式 1 或 2: layerwise scalars
    else:
        inter = torch.zeros((L,), device=device, dtype=dtype)
        hidden = torch.zeros((L,), device=device, dtype=dtype)
        
        for k_str, v in d.items():
            layer_idx = int(k_str)
            layer_keep_ratio = layerwise_keep_plan[layer_idx]
            
            # 检测使用哪种格式
            if "best_point" in v:
                # 格式 2: 直接从 best_point 中读取
                p_inter = float(v["best_point"]["p_inter"])
                p_hidden = float(v["best_point"]["p_hidden"])
            else:
                # 格式 1: 从 best_inter_prune_ratio 读取并计算 p_hidden
                p_inter = float(v["best_inter_prune_ratio"])
                p_hidden = float(
                    solve_hidden_ratio_from_inter_ratio(
                        prune_ratio=1-layer_keep_ratio,
                        inter_prune_ratio=p_inter,
                        clamp=False,
                    )
                )
            inter[layer_idx] = p_inter
            hidden[layer_idx] = p_hidden

        return inter, hidden


def load_attention_head_scores(scores_dir: str, verbose: bool = True) -> Optional[Dict[str, torch.Tensor]]:
  
    attn_head_score_file = os.path.join(scores_dir, "attn_head_scores.pth")
    if os.path.exists(attn_head_score_file):
        if verbose:
            _print(f"  - 加载 attention head scores: attn_head_scores.pth")
        return torch.load(attn_head_score_file, map_location="cpu")
  
    if verbose:
        raise FileNotFoundError(f"找不到 attention head scores 文件: {scores_dir}")
    return None

def load_channel_scores(scores_dir: str, prune_hidden, device, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    if verbose:
        _print(f"[Score Loading] Loading channel scores from {scores_dir}")
    # 加载 scores
    expert_scores = torch.load(os.path.join(scores_dir, "expert_scores.pth"), map_location=device)
    if prune_hidden:
        assert os.path.exists(os.path.join(scores_dir, "H_scores.pth")), f"H_scores.pth not found in {scores_dir}"
        H_scores = torch.load(os.path.join(scores_dir, "H_scores.pth"), map_location=device)
    else:
        H_scores = None
    return expert_scores, H_scores

def load_layerwise_loss(
    scores_dir: str,
    inter_layer_method: str,
    smooth_fn: str,
    device: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    m = re.match(r"loss_smooth_(\d+)", inter_layer_method)
    smooth_times = int(m.group(1)) if m else 0
    loss_based_kwargs = {"layerwise_loss": None, "smooth_times": 0, "smooth_fn": smooth_fn}
    if 'loss' in inter_layer_method:
        layerwise_loss = torch.load(os.path.join(scores_dir, "layerwise_loss.pth"), map_location=device)
        if verbose: 
            _print(
                f"[Score Loading] Loading layerwise_loss "
                f"(shape: {layerwise_loss.shape}), smooth_fn={smooth_fn}, smooth_times={smooth_times}"
            )
        loss_based_kwargs = {
            "layerwise_loss": layerwise_loss,
            "smooth_times": smooth_times,
            "smooth_fn": smooth_fn,
        }

    return loss_based_kwargs


def prepare_scores(
    scores_dir: str,
    mask_method_kwargs: Dict[str, Any],
    HI_ratio_kwargs: Dict[str, Any],
    prune_ratio: float,
    prune_hidden: bool,
    prune_gqa: bool,
    smooth_fn: str = "sqrt",
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    """
    根据 mask_method_kwargs 准备各种 scores。
    
    Args:
        scores_dir: scores 目录路径
        mask_method_kwargs: 包含 intra_expert_metric, hidden_score_metric, intra_layer_method 等配置
        device: 目标设备
        verbose: 是否打印详细信息
    
    Returns:
        intermediate_scores: [L, E, I] 张量
        hidden_scores: [L, E, H] 张量
        expertwise_scores: [L, E] 张量
        L: 层数
        E: 专家数
        I: 中间维度大小
        H: 隐藏维度大小
    """
    
    expert_scores, H_scores = load_channel_scores(scores_dir, prune_hidden, device, verbose)
    
    # 获取配置
    intra_expert_metric = mask_method_kwargs["intra_expert_metric"]
    hidden_score_metric = HI_ratio_kwargs.get("hidden_score_metric", "wa")
    hidden_score_metric = "H_" + hidden_score_metric
    intra_layer_method = mask_method_kwargs["intra_layer_method"]
    
    # 1) intermediate scores
    intermediate_scores = expert_scores[intra_expert_metric]
    intermediate_scores = dict_to_tensor(intermediate_scores)
    L, E, I = intermediate_scores.shape
    
    if prune_hidden:
        # 2) hidden scores
        hidden_metric_key = hidden_score_metric
        if hidden_metric_key not in H_scores:
            hidden_metric_key = f"H_{hidden_metric_key}"
        hidden_scores = H_scores[hidden_metric_key]
        hidden_scores = dict_to_tensor(hidden_scores)
        H = hidden_scores.shape[-1]
        _print(f"[Score Loading] Using hidden metric: {hidden_metric_key}")
    else:
        hidden_scores = None
        H = None
        
    if verbose:
        _print(f"[Score Loading] Using intermediate metric: {intra_expert_metric}")
        if hidden_scores is not None:
            _text = f", Hidden shape: {hidden_scores.shape}"
        else:
            _text = ""
        _print(f"[Score Loading] Intermediate shape: {intermediate_scores.shape}{_text}")
    
    # 3) expertwise_scores
    if intra_layer_method == "attr_coverage":
        expertwise_scores = expert_scores["expert_out_token_contrib"]
        expertwise_scores = dict_to_tensor(expertwise_scores)
        expertwise_scores = -expertwise_scores
    elif intra_layer_method == "second_attr_coverage":
        second_attr_key = "second_exact_attr" if "second_exact_attr" in expert_scores else "second_attr"
        expertwise_scores = expert_scores[second_attr_key]
        expertwise_scores = dict_to_tensor(expertwise_scores)
    elif intra_layer_method in ("true_ablate", "true_ablate_coverage"):
        expertwise_scores = expert_scores["true_ablate"]
        expertwise_scores = dict_to_tensor(expertwise_scores)
    elif 'loss' in intra_layer_method:  
        loss_file_path = os.path.join(os.path.dirname(os.path.dirname(scores_dir)), "loss_csv_files", "single_expert_nll_results.csv")
        if not os.path.exists(loss_file_path):
            raise FileNotFoundError(f"loss file {loss_file_path} not found")
        if verbose:
            _print(f"Loading loss from {loss_file_path}")
        expertwise_scores = load_expert_evict_loss(path=loss_file_path, L=L, E=E)
        expertwise_scores = expertwise_scores.to(device=device)
    elif "usage" in intra_layer_method:  # usage, usage_coverage
        gate_scores = torch.load(os.path.join(scores_dir, "gate_scores.pth"), map_location=device)
        expertwise_scores = gate_scores["usage"]
        expertwise_scores = dict_to_tensor(expertwise_scores)
        expertwise_scores = expertwise_scores.to(device=device)
    elif "router" in intra_layer_method:  # router, router_coverage
        gate_scores = torch.load(os.path.join(scores_dir, "gate_scores.pth"), map_location=device)
        expertwise_scores = gate_scores["out"]
        expertwise_scores = dict_to_tensor(expertwise_scores).squeeze()
        expertwise_scores = expertwise_scores.to(device=device)
    else: # uniform, uniform_coverage, channel_ranking
        expertwise_scores = torch.ones((L, E), dtype=torch.float32, device=device)

    inter_layer_method = mask_method_kwargs["inter_layer_method"]
    # 加载 layerwise_loss（如果需要）
    loss_based_kwargs = load_layerwise_loss(scores_dir, inter_layer_method, smooth_fn, device, verbose)
    
    layerwise_keep_plan = inter_layer_planner(
        intermediate_scores,
        p_target=prune_ratio,
        method=inter_layer_method,
        L=L,
        loss_based_importance_kwargs=loss_based_kwargs,
        tol=0.1,
        verbose=verbose,
    )
    loss_based_kwargs["layerwise_keep_plan"] = layerwise_keep_plan

    if intra_layer_method == "channel_ranking":
        intermediate_masks_float, K_E = intra_layer_planner(
            scores=intermediate_scores,
            expertwise_scores=expertwise_scores,
            keep_ratio=layerwise_keep_plan,
            method="channel_ranking",
            L=L,
            E=E,
            I=I,
            verbose=verbose,
        )   # -> intermediate_masks_float [L, E, I]
        expertwise_scores = intermediate_masks_float.sum(dim=-1) / (I)

    return intermediate_scores, hidden_scores, expertwise_scores, L, E, I, H, loss_based_kwargs
