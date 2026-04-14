import torch
import torch.nn as nn
from .utils import *

def masked_channel_rms(
    act: torch.Tensor | None,
    token_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if act is None:
        return None
    if token_mask is None:
        return channel_rms(act)
    token_mask = token_mask.to(device=act.device).view(-1).bool()
    if token_mask.numel() != act.shape[0] or not bool(token_mask.any()):
        return None
    return channel_rms(act[token_mask])

def wa_score(weight: torch.Tensor, activation: torch.Tensor, sum_dim=0) -> torch.Tensor:
    return (weight.abs() * activation.unsqueeze(0)).sum(dim=sum_dim)  # [channels]

def snip_score(
    weight: torch.Tensor,
    grad: torch.Tensor,
    channel_dim: int = 0,
) -> torch.Tensor:
    score = (weight * grad).abs()
    reduce_dims = [d for d in range(score.ndim) if d != channel_dim]
    return score.sum(dim=reduce_dims)

def token_contrib_old(g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(z.dim() - 1))  # 平均 batch, seq 维度
    token_contrib = (g * z).sum(dim=dims) / z.size(-1)  # [S, I] -> [I]
    # Q_e = token_contrib.mean()
    return token_contrib

def token_contrib(
    g: torch.Tensor,
    z: torch.Tensor,
    trim_head: float = 0.01,  # clip 掉绝对值最大的 top p%
    trim_tail: float = 0.00,  # 现在不再用,保留接口以兼容
) -> torch.Tensor:
    """
    g, z: [..., I]
    返回: [I], 每个 channel 的 clipped mean 贡献

    做的事情:
    - c = g * z
    - 按绝对值算每个 channel 的 (1 - trim_head) 分位数 q_high
    - 对每个 channel 做带符号 clip 到 [-q_high, q_high]
    - 然后对 token 维度取均值
    """
    assert g.shape == z.shape, "g 和 z 的形状必须一致"
    I = z.size(-1)
    orig_dtype = z.dtype

    # 所有非最后一维都看作 token 维度, 展平
    gz = g * z                               # [..., I]
    gz_flat = gz.view(-1, I).to(torch.float32)  # [N_tokens, I], 用 float32 以支持 quantile

    N = gz_flat.size(0)
    # 样本太少或不开启 clip, 退化为普通均值
    if N <= 2 or trim_head <= 0.0:
        return gz_flat.mean(dim=0).to(orig_dtype)  # [I]

    # 限制一下比例, 避免奇怪超参
    trim_head = float(max(0.0, min(trim_head, 0.49)))

    abs_gz = gz_flat.abs()  # [N_tokens, I]

    # 每个 channel 的 (1 - trim_head) 分位数, 例如 trim_head=0.01 -> 99% 分位
    q_high = torch.quantile(
        abs_gz, 1.0 - trim_head, dim=0, keepdim=True
    )  # [1, I]

    # 防止某些 channel 全是 0, 分位数为 0 导致全截成 0
    # 加一个极小下界
    q_high = torch.clamp(q_high, min=1e-25)

    # 带符号 clip 到 [-q_high, q_high]
    clipped = torch.clamp(gz_flat, min=-q_high, max=q_high)  # [N_tokens, I]

    # 对 token 维度取均值
    contrib_mean = clipped.mean(dim=0)  # [I], float32

    return contrib_mean.to(orig_dtype)


# 通道 saliency, 使用 act * grad
def channel_saliency(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    # act, grad 形状类似 [..., I], 最后一维是通道
    s = (act * grad).abs().detach()
    dims = tuple[int, ...](range(s.dim() - 1))  # 平均 batch, seq 维度
    return s.mean(dim=dims)


def channel_saliency_masked(
    act: torch.Tensor,
    grad: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """与 channel_saliency 相同，但仅在 token_mask 为 True 的 token 上平均（首维为 token，对齐 compute_gateup_act）。"""
    if act is None or grad is None:
        return None
    s = (act * grad).abs().detach()
    if token_mask is not None:
        token_mask = token_mask.to(device=s.device).view(-1).bool()
        if token_mask.numel() != s.shape[0] or not bool(token_mask.any()):
            return None
        s = s[token_mask]
    dims = tuple[int, ...](range(s.dim() - 1))
    return s.mean(dim=dims).to(torch.float32)


def compute_token_contrib_I(down_input: torch.Tensor = None, 
                            down_grad: torch.Tensor = None, 
                            up_output: torch.Tensor = None,
                            up_out_grad: torch.Tensor = None, 
                            gate_output: torch.Tensor = None,
                            gate_grad: torch.Tensor = None):
    down_token_contrib = token_contrib(down_grad, down_input)   # [I]
    up_token_contrib = token_contrib(up_out_grad, up_output)   # [I]
    gate_token_contrib = token_contrib(gate_grad, gate_output) # [I]
    token_contrib_mean = (down_token_contrib + up_token_contrib + gate_token_contrib) / 3.0  # [I]
    return token_contrib_mean.to(torch.float32)

def compute_grad_I(down_grad: torch.Tensor = None, 
                    up_out_grad: torch.Tensor = None, 
                    gate_grad: torch.Tensor = None):

    down_grad = channel_rms(down_grad)
    up_out_grad = channel_rms(up_out_grad)
    gate_grad = channel_rms(gate_grad)
    grad_mean = (down_grad + up_out_grad + gate_grad) / 3.0
    return grad_mean.to(torch.float32), down_grad.to(torch.float32), up_out_grad.to(torch.float32), gate_grad.to(torch.float32)


def compute_grad_I_masked(
    down_grad: torch.Tensor = None,
    up_out_grad: torch.Tensor = None,
    gate_grad: torch.Tensor = None,
    token_mask: torch.Tensor | None = None,
):
    down_grad_ch = masked_channel_rms(down_grad, token_mask)
    up_out_grad_ch = masked_channel_rms(up_out_grad, token_mask)
    gate_grad_ch = masked_channel_rms(gate_grad, token_mask)
    if down_grad_ch is None or up_out_grad_ch is None or gate_grad_ch is None:
        return None
    ret = (down_grad_ch + up_out_grad_ch + gate_grad_ch) / 3.0
    return ret.to(torch.float32)
        
def compute_saliency_I(down_input: torch.Tensor = None, 
                        down_grad: torch.Tensor = None,
                        # up_output: torch.Tensor = None, 
                        # up_out_grad: torch.Tensor = None,
                        # gate_output: torch.Tensor = None,
                        # gate_grad: torch.Tensor = None):
                    ):

    # 1) 三个 proj 的通道 saliency
    down_sal = channel_saliency(down_input, down_grad)
    # up_sal   = channel_saliency(up_output, up_out_grad)
    # gate_sal = channel_saliency(gate_output, gate_grad)

    # 三者通道数应该一致
    # assert down_sal.shape == up_sal.shape == gate_sal.shape
    # sal_mean = (down_sal + up_sal + gate_sal) / 3.0  # [I]
    # return sal_mean
    return down_sal.to(torch.float32)


def compute_3proj_saliency_I(
    down_input: torch.Tensor = None,
    down_grad: torch.Tensor = None,
    up_output: torch.Tensor = None,
    up_out_grad: torch.Tensor = None,
    gate_output: torch.Tensor = None,
    gate_grad: torch.Tensor = None,
):
    down_sal = channel_saliency(down_input, down_grad)
    up_sal = channel_saliency(up_output, up_out_grad)
    gate_sal = channel_saliency(gate_output, gate_grad)
    assert down_sal.shape == up_sal.shape == gate_sal.shape
    ret = (down_sal + up_sal + gate_sal) / 3.0
    return ret.to(torch.float32)


def compute_3proj_saliency_I_masked(
    down_input: torch.Tensor = None,
    down_grad: torch.Tensor = None,
    up_output: torch.Tensor = None,
    up_out_grad: torch.Tensor = None,
    gate_output: torch.Tensor = None,
    gate_grad: torch.Tensor = None,
    token_mask: torch.Tensor | None = None,
):
    down_sal = channel_saliency_masked(down_input, down_grad, token_mask)
    up_sal = channel_saliency_masked(up_output, up_out_grad, token_mask)
    gate_sal = channel_saliency_masked(gate_output, gate_grad, token_mask)
    if down_sal is None or up_sal is None or gate_sal is None:
        return None
    ret = (down_sal + up_sal + gate_sal) / 3.0
    return ret.to(torch.float32)
            
def compute_activation_I(down_input: torch.Tensor = None, 
                          up_output: torch.Tensor = None,
                          gate_output: torch.Tensor = None):
    
    down_act = channel_rms(down_input)   # down_act.shape = [S, I]
    up_act   = channel_rms(up_output)    # up_act.shape = [S, I]
    gate_act = channel_rms(gate_output)  # gate_act.shape = [S, I]

    assert down_act.shape == up_act.shape == gate_act.shape
    act_mean = (down_act + up_act + gate_act) / 3.0  # [I]

    return act_mean.to(torch.float32), down_act.to(torch.float32)


def compute_activation_I_masked(
    down_input: torch.Tensor = None,
    up_output: torch.Tensor = None,
    gate_output: torch.Tensor = None,
    token_mask: torch.Tensor | None = None,
):
    down_act = masked_channel_rms(down_input, token_mask)
    up_act = masked_channel_rms(up_output, token_mask)
    gate_act = masked_channel_rms(gate_output, token_mask)
    if down_act is None or up_act is None or gate_act is None:
        return None
    ret = (down_act + up_act + gate_act) / 3.0
    return ret.to(torch.float32)

def compute_wa_I(W_down: torch.Tensor = None, 
                 W_up: torch.Tensor = None, 
                 W_gate: torch.Tensor = None, 
                 down_ch_act: torch.Tensor = None, 
                 up_input: torch.Tensor = None, 
                 gate_input: torch.Tensor = None):

    # 通道维度一致
    assert W_down.size(1) == W_up.size(0) == W_gate.size(0)   # [H, I], [I, H], [I, H]. 如果 load_in_4bit 的话不能这么算 wa_score, 目前不支持量化 model

    up_input = channel_rms(up_input)      # up_input.shape = [S, H]
    gate_input = channel_rms(gate_input)  # gate_input.shape = [S, H]
    wa_down = wa_score(W_down, down_ch_act, sum_dim=0)     # [H, I] * [1, I] -> [H, I] -> [I]
    wa_up   = wa_score(W_up, up_input, sum_dim=1)       # [I, H] * [1, H] -> [I, H] -> [I]
    wa_gate = wa_score(W_gate, gate_input, sum_dim=1)   # [I, H] * [1, H] -> [I, H] -> [I]

    wa_mean = (wa_down + wa_up + wa_gate) / 3.0  # [I]

    return wa_mean.to(torch.float32)


def compute_wa_I_masked(
    W_down: torch.Tensor = None,
    W_up: torch.Tensor = None,
    W_gate: torch.Tensor = None,
    down_input: torch.Tensor = None,
    up_input: torch.Tensor = None,
    gate_input: torch.Tensor = None,
    token_mask: torch.Tensor | None = None,
):
    down_ch_act = masked_channel_rms(down_input, token_mask)
    up_in = masked_channel_rms(up_input, token_mask)
    gate_in = masked_channel_rms(gate_input, token_mask)
    if down_ch_act is None or up_in is None or gate_in is None:
        return None
    wa_down = wa_score(W_down, down_ch_act, sum_dim=0)
    wa_up = wa_score(W_up, up_in, sum_dim=1)
    wa_gate = wa_score(W_gate, gate_in, sum_dim=1)
    ret = (wa_down + wa_up + wa_gate) / 3.0
    return ret.to(torch.float32)


def compute_wg_I(W_down: torch.Tensor = None, 
                 W_up: torch.Tensor = None, 
                 W_gate: torch.Tensor = None, 
                 W_down_grad: torch.Tensor = None, 
                 W_up_grad: torch.Tensor = None, 
                 W_gate_grad: torch.Tensor = None, 
                 ):

    # 通道维度一致
    assert W_down.size(1) == W_up.size(0) == W_gate.size(0)   # [H, I], [I, H], [I, H]. 如果 load_in_4bit 的话不能这么算 wa_score, 目前不支持量化 model
    if W_down_grad is None or W_up_grad is None or W_gate_grad is None:
        return torch.zeros(W_down.size(1), dtype=torch.float32, device=W_down.device)

    W_down_grad = W_down_grad.detach()
    W_up_grad = W_up_grad.detach()
    W_gate_grad = W_gate_grad.detach()
    
    # 对齐通道维度:
    # - W_down: 输入通道在 dim=1, 输出通道在 dim=0
    # - W_up / W_gate: 输入通道在 dim=0
    snip_score_down = snip_score(W_down, W_down_grad, channel_dim=1)  # [I]
    snip_score_up   = snip_score(W_up, W_up_grad, channel_dim=0)      # [I]
    snip_score_gate = snip_score(W_gate, W_gate_grad, channel_dim=0)  # [I]
    snip_score_mean = (snip_score_down + snip_score_up + snip_score_gate) / 3.0  # [I]

    return snip_score_mean.to(torch.float32)

def compute_grad_H(down_out_grad: torch.Tensor = None, 
                   up_in_grad: torch.Tensor = None,     # shape [S, H]
                   gate_in_grad: torch.Tensor = None,   # shape [S, H]
                   ) -> torch.Tensor:
   
    down_grad = channel_rms(down_out_grad)
    up_grad = channel_rms(up_in_grad)
    gate_grad = channel_rms(gate_in_grad)
    grad_mean = (down_grad + up_grad + gate_grad) / 3.0

    return grad_mean.to(torch.float32)

def compute_saliency_H(down_output: torch.Tensor = None, 
                       down_out_grad: torch.Tensor = None, 
                       up_input: torch.Tensor = None, 
                       up_in_grad: torch.Tensor = None, 
                       gate_input: torch.Tensor = None, 
                       gate_in_grad: torch.Tensor = None):   # shape [S, H] -> [H]
    
    down_sal = channel_saliency(down_output, down_out_grad)
    up_sal   = channel_saliency(up_input, up_in_grad)
    gate_sal = channel_saliency(gate_input, gate_in_grad)

    assert down_sal.shape == up_sal.shape == gate_sal.shape
    sal_mean = (down_sal + up_sal + gate_sal) / 3.0  # [I]

    return sal_mean.to(torch.float32)

def compute_activation_H(down_output: torch.Tensor = None, 
                         up_input: torch.Tensor = None,
                         gate_input: torch.Tensor = None):
    
    # 通道 activation, Wanda / MoE-Pruner 中的 ‖X_j‖ (L2 范數)
    down_act = channel_rms(down_output)   # shape [S, H] -> [H]
    up_act   = channel_rms(up_input)    # shape [S, H] -> [H]
    gate_act = channel_rms(gate_input)  # shape [S, H] -> [H]

    assert down_act.shape == up_act.shape == gate_act.shape
    act_mean = (down_act + up_act + gate_act) / 3.0  # [H]

    return act_mean.to(torch.float32)

def compute_wa_H(down_input: torch.Tensor = None, 
                 up_input: torch.Tensor = None,     # shape [S, H]
                 gate_input: torch.Tensor = None,   # shape [S, H]
                 W_down: torch.Tensor = None,   # [H, I]
                 W_up: torch.Tensor = None,     # [I, H]
                 W_gate: torch.Tensor = None):   # [I, H]
    
    down_input = channel_rms(down_input)   # shape [S, I] -> [I]
    up_input   = channel_rms(up_input)    # shape [S, H] -> [H]
    gate_input = channel_rms(gate_input)  # shape [S, H] -> [H]

    wa_down = wa_score(W_down, down_input, sum_dim=1)     # [H, I] * [1, I] -> [H, I] -> [H]
    wa_up   = wa_score(W_up, up_input, sum_dim=0)       # [I, H] * [1, H] -> [I, H] -> [H]
    wa_gate = wa_score(W_gate, gate_input, sum_dim=0)   # [I, H] * [1, H] -> [I, H] -> [H]
    wa_mean = (wa_down + wa_up + wa_gate) / 3.0  # [H]

    return wa_mean.to(torch.float32)
        

def resolve_activation_fn(expert: nn.Module):
    for attr in ("act_fn", "activation_fn"):
        fn = getattr(expert, attr, None)
        if fn is not None:
            return fn
    return torch.nn.functional.silu


def compute_channel_hessian_diag(
    W_down: torch.Tensor,
    down_input: torch.Tensor,
    token_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Analytic Hessian diagonal: importance_j = ‖W_down[:,j]‖² · Σ_t h_{t,j}²."""
    w_col_norm2 = W_down.detach().float().pow(2).sum(dim=0)  # [I]
    h = down_input.detach().float()
    if token_mask is not None:
        token_mask = token_mask.to(device=h.device).view(-1).bool()
        if not bool(token_mask.any()):
            # return torch.zeros(h.shape[-1], dtype=torch.float32, device=h.device)
            return None
        h = h[token_mask]
    return (w_col_norm2 * h.pow(2).sum(dim=0)).to(torch.float32)


def compute_3linear_hessian_diag(
    W_down: torch.Tensor,
    W_up: torch.Tensor,
    W_gate: torch.Tensor,
    down_input: torch.Tensor,
    up_output: torch.Tensor,
    gate_output: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate Hessian diag over down/up/gate to channel dim I."""
    down_h = compute_channel_hessian_diag(W_down, down_input, token_mask)
    up_h = compute_channel_hessian_diag(W_up.transpose(0, 1), up_output, token_mask)
    gate_h = compute_channel_hessian_diag(W_gate.transpose(0, 1), gate_output, token_mask)
    ret = (down_h + up_h + gate_h) / 3.0
    return ret.to(torch.float32)


def compute_gateup_act(
    expert: nn.Module,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    token_mask: torch.Tensor | None = None,
):
    if gate_output is None or up_output is None:
        return None
    activation = resolve_activation_fn(expert)(gate_output.detach()) * up_output.detach()
    if token_mask is not None:
        token_mask = token_mask.to(device=activation.device).view(-1).bool()
        if token_mask.numel() != activation.shape[0] or not bool(token_mask.any()):
            return None
        activation = activation[token_mask]
    dims = tuple(range(activation.dim() - 1))
    ret = activation.abs().to(torch.float32).mean(dim=dims)
    return ret.to(torch.float32)