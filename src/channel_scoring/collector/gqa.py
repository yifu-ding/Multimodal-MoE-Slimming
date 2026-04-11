"""
GQA (Grouped Query Attention) score collection utilities.

Provides functions to collect GQA-related scores including:
- WO group scores (KV head group scores)
- Attention head similarity matrix
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.shared_utils.safe_isinstance import _get_text_cfg, _is_ds_model

def get_o_proj(module: torch.nn.Module) -> torch.nn.Linear:
    # 适配常见命名
    candidates = []
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(module, name):
            candidates.append(getattr(module, name))
    candidates.append(module)

    for m in candidates:
        if hasattr(m, "o_proj") and isinstance(getattr(m, "o_proj"), torch.nn.Linear):
            return getattr(m, "o_proj")
    raise AttributeError("Cannot find o_proj Linear inside cnt_block.")


@torch.no_grad()
def update_wo_gqa_group_scores(
    cnt_block: torch.nn.Module,
    scores_ema_dict: dict[str, torch.Tensor] | None,   # dict with keys: mean, max, sum; values: [num_heads]
    ema: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """
    Compute per query-head scores without GQA grouping.
    
    Returns:
        dict with keys 'mean', 'max', 'sum', each containing a tensor of shape [num_heads]
        Note: All three strategies return identical values for per-head scoring.
    """
    o_proj = get_o_proj(cnt_block)
    W = o_proj.weight
    g = o_proj.saved_weight_grad
    o_proj.saved_weight_grad = None

    if g is None:
        raise RuntimeError("o_proj.weight.grad is None. Did you call backward()?")

    # sensitivity tensor: s = dL/dW * W
    s = (W.float() * g.float())  # [d_model, inner_dim]

    # Eq.4 on W^O rows in paper -> column-wise l2 on PyTorch weight
    col_score = torch.sqrt((s * s).sum(dim=0) + 1e-10)  # [inner_dim]

    inner_dim = col_score.numel()
    expected = num_heads * head_dim
    if inner_dim != expected:
        raise RuntimeError(f"inner_dim mismatch: got {inner_dim}, expected {expected} = H*D")

    # per query-head score: mean over head_dim
    head_score = col_score.view(num_heads, head_dim).mean(dim=1)  # [H]

    # Apply EMA
    # For per-head scoring, all three strategies return the same value
    cur = head_score.detach()
    result = {}
    for strategy in ["mean", "max", "sum"]:
        if scores_ema_dict is None or strategy not in scores_ema_dict:
            result[strategy] = cur
        else:
            result[strategy] = (scores_ema_dict[strategy] * ema + cur * (1.0 - ema))
    return result


@torch.no_grad()
def compute_attention_head_similarity(
    cnt_block: torch.nn.Module,
    similarity_matrix_ema: torch.Tensor | None,
    ema: float,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算同一层内不同 attention heads 之间的相似度矩阵。
    
    使用注意力模式相似度来判断 head 冗余性：
    - 如果两个 head 的注意力模式高度相似，说明它们可能是冗余的
    - 使用余弦相似度来衡量 attention map 的相似性
    
    Args:
        cnt_block: 包含 self_attn 的 block
        similarity_matrix_ema: 上一次的相似度矩阵 EMA，shape [num_heads, num_heads]
        ema: EMA 系数
        attn_mask: attention mask，shape [B, S]
        
    Returns:
        similarity_matrix: 相似度矩阵，shape [num_heads, num_heads]
                          element [i, j] 表示 head i 和 head j 的相似度（余弦相似度）
    """
    self_attn = cnt_block.self_attn
    
    # 获取 hook 保存的输入
    self_attn_input = self_attn.saved_input  # [B, S, H]
    if self_attn_input is None:
        raise RuntimeError("self_attn.saved_input is None. Did you add the hook?")
    
    # 获取 q_proj 和 k_proj 的权重
    q_proj_weight = self_attn.q_proj.weight  # [num_heads * head_dim, H]
    k_proj_weight = self_attn.k_proj.weight  # [num_kv_heads * head_dim, H]
    
    # 获取配置参数
    if hasattr(self_attn, 'num_heads'):
        num_heads = self_attn.num_heads
        num_kv_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim
    elif hasattr(cnt_block, 'self_attn') and hasattr(cnt_block.self_attn, 'config'):
        config = cnt_block.self_attn.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, 'head_dim', config.hidden_size // num_heads)
    else:
        # 从权重形状推断
        num_heads = q_proj_weight.shape[0] // (k_proj_weight.shape[0] // k_proj_weight.shape[0])
        raise RuntimeError("Cannot infer num_heads, num_kv_heads, head_dim from module")
    
    B, S, H = self_attn_input.shape
    
    # 1. 计算 Q 和 K
    # Q: [B, S, H] @ [H, num_heads * head_dim]^T -> [B, S, num_heads * head_dim]
    Q = F.linear(self_attn_input.float(), q_proj_weight.float())  # [B, S, num_heads * head_dim]
    Q = Q.view(B, S, num_heads, head_dim)  # [B, S, num_heads, head_dim]
    
    # K: [B, S, H] @ [H, num_kv_heads * head_dim]^T -> [B, S, num_kv_heads * head_dim]
    K = F.linear(self_attn_input.float(), k_proj_weight.float())  # [B, S, num_kv_heads * head_dim]
    K = K.view(B, S, num_kv_heads, head_dim)  # [B, S, num_kv_heads, head_dim]
    
    # 2. 处理 GQA：扩展 K 到所有 query heads
    if num_heads != num_kv_heads:
        # GQA: 需要将 K 扩展到所有 query heads
        group_size = num_heads // num_kv_heads
        K = K.repeat_interleave(group_size, dim=2)  # [B, S, num_heads, head_dim]
    
    # 3. 计算 attention scores (不需要完整的 softmax，只需要相对模式)
    # Q @ K^T: [B, S, num_heads, head_dim] @ [B, S, num_heads, head_dim]^T
    # -> [B, num_heads, S, S]
    Q = Q.transpose(1, 2)  # [B, num_heads, S, head_dim]
    K = K.transpose(1, 2)  # [B, num_heads, S, head_dim]
    
    scaling_factor = head_dim ** -0.5
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * scaling_factor  # [B, num_heads, S, S]
    
    # 4. 应用 attention mask（可选）
    if attn_mask is not None:
        # attn_mask: [B, S]
        # 扩展为 [B, 1, 1, S]，然后 broadcast 到 [B, num_heads, S, S]
        attn_mask_expanded = attn_mask[:, None, None, :].to(attention_scores.dtype)
        # 将 mask=0 的位置设为很小的值（不影响相似度计算）
        attention_scores = attention_scores * attn_mask_expanded
    
    # 5. 对每个 head，将 attention_scores 展平为向量
    # [B, num_heads, S, S] -> [B, num_heads, S*S]
    attention_vectors = attention_scores.view(B, num_heads, S * S)
    
    # 6. 对 batch 求平均
    # [B, num_heads, S*S] -> [num_heads, S*S]
    attention_vectors_mean = attention_vectors.mean(dim=0)
    
    # 7. 计算 head 之间的余弦相似度矩阵
    # [num_heads, S*S] -> [num_heads, num_heads]
    # 先归一化
    attention_vectors_norm = F.normalize(attention_vectors_mean, p=2, dim=1)  # [num_heads, S*S]
    
    # 计算余弦相似度：cosine_sim = A @ A^T (after normalization)
    similarity_matrix = torch.matmul(attention_vectors_norm, attention_vectors_norm.t())  # [num_heads, num_heads]
    
    # 8. 应用 EMA
    if similarity_matrix_ema is None:
        return similarity_matrix.detach()
    else:
        return (similarity_matrix_ema * ema + similarity_matrix.detach() * (1.0 - ema))

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def compute_attention_head_similarity_deepseekv2(
    cnt_block: nn.Module,
    similarity_matrix_ema: torch.Tensor | None,
    ema: float,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    DeepSeek-V2 attention head similarity matrix using attention score patterns.

    This version matches a DeepseekV2Attention that has:
      - self_attn.q_head_dim
      - self_attn.qk_nope_head_dim, self_attn.qk_rope_head_dim
      - self_attn.kv_lora_rank, self_attn.v_head_dim
      - self_attn.num_heads
      - q path: q_proj OR (q_a_proj, q_a_layernorm, q_b_proj) controlled by q_lora_rank
      - kv path: kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj
      - hook: self_attn.saved_input exists [B,S,H]

    RoPE rotation is NOT applied (no position_embeddings needed).
    """

    self_attn = getattr(cnt_block, "self_attn", None)
    if self_attn is None:
        raise RuntimeError("cnt_block.self_attn not found (expected DeepseekV2Attention).")

    x = getattr(self_attn, "saved_input", None)  # [B, S, hidden]
    if x is None:
        raise RuntimeError("self_attn.saved_input is None. Did you add the hook?")
    B, S, _ = x.shape

    num_heads = int(self_attn.num_heads)

    # IMPORTANT: your runtime module uses q_head_dim
    q_head_dim = int(self_attn.q_head_dim)

    qk_nope = int(self_attn.qk_nope_head_dim)
    qk_rope = int(self_attn.qk_rope_head_dim)

    kv_lora_rank = int(self_attn.kv_lora_rank)
    v_head_dim = int(self_attn.v_head_dim)

    if qk_nope + qk_rope != q_head_dim:
        raise RuntimeError(
            f"Dim mismatch: qk_nope({qk_nope}) + qk_rope({qk_rope}) != q_head_dim({q_head_dim})."
        )

    # 1) Q path (same structure as forward, excluding RoPE rotation)
    if self_attn.q_lora_rank is None:
        q = self_attn.q_proj(x)  # [B, S, num_heads*q_head_dim]
    else:
        q = self_attn.q_b_proj(self_attn.q_a_layernorm(self_attn.q_a_proj(x)))  # [B, S, num_heads*q_head_dim]

    q = q.view(B, S, num_heads, q_head_dim).transpose(1, 2)  # [B, H, S, q_head_dim]
    q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)

    # 2) KV path (same structure as forward, excluding RoPE rotation)
    compressed_kv = self_attn.kv_a_proj_with_mqa(x)  # [B, S, kv_lora_rank + qk_rope]
    k_nope_latent, k_pe = torch.split(compressed_kv, [kv_lora_rank, qk_rope], dim=-1)

    kv_dec = self_attn.kv_b_proj(self_attn.kv_a_layernorm(k_nope_latent))  # [B, S, H*(qk_nope+v)]
    expected_kv_out = num_heads * (qk_nope + v_head_dim)
    if int(kv_dec.shape[-1]) != expected_kv_out:
        raise RuntimeError(
            f"kv_b_proj output dim mismatch: got {int(kv_dec.shape[-1])}, expected {expected_kv_out} "
            f"(=num_heads*(qk_nope+v_head_dim))."
        )

    kv_dec = kv_dec.view(B, S, num_heads, qk_nope + v_head_dim).transpose(1, 2)  # [B,H,S,qk_nope+v]
    k_nope, _value_states = torch.split(kv_dec, [qk_nope, v_head_dim], dim=-1)

    # shared rope key expanded to all heads
    k_pe = k_pe.view(B, 1, S, qk_rope).expand(B, num_heads, S, qk_rope)

    query_states = torch.cat((q_nope, q_pe), dim=-1)  # [B,H,S,q_head_dim]
    key_states = torch.cat((k_nope, k_pe), dim=-1)    # [B,H,S,q_head_dim]

    # 3) Attention scores
    scaling = float(q_head_dim) ** -0.5
    attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling  # [B,H,S,S]

    # 4) Apply mask (same semantics as your original: multiply)
    if attn_mask is not None:
        if attn_mask.dim() != 2 or attn_mask.shape[0] != B or attn_mask.shape[1] != S:
            raise RuntimeError(f"attn_mask expected shape [B,S]=[{B},{S}], got {tuple(attn_mask.shape)}")
        m = attn_mask[:, None, None, :].to(dtype=attention_scores.dtype, device=attention_scores.device)
        attention_scores = attention_scores * m

    # 5) Flatten per head, average over batch, cosine similarity
    vec = attention_scores.view(B, num_heads, S * S).mean(dim=0)  # [H, S*S]
    vec = F.normalize(vec, p=2, dim=1)
    sim = torch.matmul(vec, vec.t())  # [H, H]

    cur = sim.detach()
    if similarity_matrix_ema is None:
        return cur

    if similarity_matrix_ema.shape != cur.shape:
        raise RuntimeError(
            f"similarity_matrix_ema shape mismatch: ema={tuple(similarity_matrix_ema.shape)} vs cur={tuple(cur.shape)}"
        )
    return similarity_matrix_ema.to(cur.device, cur.dtype) * float(ema) + cur * (1.0 - float(ema))


@torch.no_grad()
def update_wo_head_scores_deepseekv2(
    cnt_block: nn.Module,
    scores_ema_dict: dict[str, torch.Tensor] | None,  # keys: mean, max, sum; values: [num_heads]
    ema: float,
    num_heads: int,
    v_head_dim: int,
) -> dict[str, torch.Tensor]:
    """
    DeepSeek-V2 MLA attention: compute per-attention-head scores from o_proj using sensitivity s = W * dL/dW.

    This implements "方案 A": return per-head scores [num_heads], no KV grouping.

    Args:
        cnt_block: a block or attention module that contains o_proj, and o_proj.saved_weight_grad is populated.
        scores_ema_dict: optional EMA state dict with keys {'mean','max','sum'} and tensors [num_heads].
        ema: EMA factor. new = old*ema + cur*(1-ema)
        num_heads: attention head count to score (the heads you will structurally prune).
        v_head_dim: per-head value dimension. For DeepSeek-V2, o_proj input inner_dim should be num_heads * v_head_dim.

    Returns:
        dict with keys {'mean','max','sum'}, each a tensor of shape [num_heads].
        For per-head scoring, 'mean','max','sum' are identical by construction, kept for interface compatibility.
    """
    # You likely already have this helper in your codebase.
    # It should return the attention output projection module.
    o_proj = get_o_proj(cnt_block)

    W = o_proj.weight
    g = getattr(o_proj, "saved_weight_grad", None)
    o_proj.saved_weight_grad = None

    if g is None:
        raise RuntimeError("o_proj.saved_weight_grad is None. Did you run backward and save the grad hook?")

    # sensitivity tensor: s = dL/dW * W
    s = (W.float() * g.float())  # [d_model, inner_dim]

    # Column-wise l2 norm (PyTorch Linear: weight shape [out_features, in_features])
    col_score = torch.sqrt((s * s).sum(dim=0) + 1e-10)  # [inner_dim]

    inner_dim = int(col_score.numel())
    expected = int(num_heads) * int(v_head_dim)
    if inner_dim != expected:
        raise RuntimeError(
            f"inner_dim mismatch on o_proj: got {inner_dim}, expected {expected} (=num_heads*v_head_dim). "
            f"num_heads={num_heads}, v_head_dim={v_head_dim}."
        )

    # per-head score: mean over v_head_dim within each head block
    head_score = col_score.view(int(num_heads), int(v_head_dim)).mean(dim=1)  # [num_heads]

    # Keep the same return keys as the GQA version for minimal downstream disruption.
    # For per-head scoring, these three are equivalent; downstream can pick any.
    cur = head_score.detach()

    result: dict[str, torch.Tensor] = {}
    for strategy in ("mean", "max", "sum"):
        if scores_ema_dict is None or strategy not in scores_ema_dict:
            result[strategy] = cur
        else:
            prev = scores_ema_dict[strategy].to(device=cur.device, dtype=cur.dtype)
            if prev.numel() != cur.numel():
                raise RuntimeError(
                    f"EMA buffer shape mismatch for '{strategy}': prev={prev.numel()} vs cur={cur.numel()}."
                )
            result[strategy] = prev * float(ema) + cur * (1.0 - float(ema))

    return result


@torch.no_grad()
def collect_scores_gqa(
    cnt_block: nn.Module,
    model: nn.Module,
    saliency_ema: float,
    attn_mask: torch.Tensor,
    device: str = "cuda",
) -> None:
    """
    Collect GQA-related scores for a block.

    This function:
    1. Initializes and updates WO group scores (KV head group scores)
    2. Initializes and updates attention head similarity matrix

    Args:
        cnt_block: The block module to collect scores from
        model: The full model (for accessing config)
        saliency_ema: EMA coefficient for score updates
        attn_mask: Attention mask tensor [B, T]
        device: Target device for tensors
    """
    model_cfg = _get_text_cfg(model)
    # Initialize wo_group_scores dict if not exists
    if not hasattr(cnt_block, "wo_group_scores_dict"):
        cnt_block.wo_group_scores_dict = {
            "mean": torch.zeros(model_cfg.num_attention_heads, device=device),
            "max": torch.zeros(model_cfg.num_attention_heads, device=device),
            "sum": torch.zeros(model_cfg.num_attention_heads, device=device),
        }
    
    if _is_ds_model(model):
        cnt_block.wo_group_scores_dict = update_wo_head_scores_deepseekv2(
            cnt_block,
            scores_ema_dict=cnt_block.wo_group_scores_dict,
            ema=saliency_ema,
            num_heads=model_cfg.num_attention_heads,
            v_head_dim=getattr(model_cfg, "head_dim", model_cfg.hidden_size // model_cfg.num_attention_heads),
        )
    else:
        cnt_block.wo_group_scores_dict = update_wo_gqa_group_scores(
            cnt_block,
            scores_ema_dict=cnt_block.wo_group_scores_dict,
            ema=saliency_ema,
            num_heads=model_cfg.num_attention_heads,
            num_kv_heads=model_cfg.num_key_value_heads,
            head_dim=getattr(model_cfg, "head_dim", model_cfg.hidden_size // model_cfg.num_attention_heads),
        )
    
    # Compute attention head similarity matrix
    if not hasattr(cnt_block, "attention_head_similarity_matrix"):
        num_attn_heads = model_cfg.num_attention_heads
        cnt_block.attention_head_similarity_matrix = torch.zeros(
            num_attn_heads, num_attn_heads, device=device
        )
    
    if _is_ds_model(model):
        cnt_block.attention_head_similarity_matrix = compute_attention_head_similarity_deepseekv2(
            cnt_block,
            similarity_matrix_ema=cnt_block.attention_head_similarity_matrix,
            ema=saliency_ema,
            attn_mask=attn_mask,
        )
    else:
        cnt_block.attention_head_similarity_matrix = compute_attention_head_similarity(
            cnt_block,
            similarity_matrix_ema=cnt_block.attention_head_similarity_matrix,
            ema=saliency_ema,
            attn_mask=attn_mask,
        )
