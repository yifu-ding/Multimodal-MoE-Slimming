"""Phase 1: Per-channel importance score collection for Kimi-VL MoE experts.

Runs forward passes over VL calibration data and accumulates per-channel
activation scores (or weight-norm scores) for every routed expert in every
MoE layer.  Output is saved to ``scores.pt``.

Optionally also collects contribution scores (--collect_contrib):
  - layerwise_loss      : per-MoE-layer MLP reconstruction loss (grad-based)
  - expert_out_contrib  : |down_out * down_out_grad|.sum() × usage per expert
  - expert_usage        : routing fraction per expert (gradient-free)
  - layerwise_repr_change: ||h_out - h_in|| / ||h_in|| per MoE layer (gradient-free)

Usage
-----
    # Activation scoring + contribution scores
    python src/collect_scores.py \\
        --model_name_or_path moonshotai/Kimi-VL-A3B-Instruct \\
        --output_dir storage/prune/scores/kimi_gqa \\
        --num_samples 128 \\
        --score_type activation \\
        --collect_contrib

    # Weight scoring only (no forward pass)
    python src/collect_scores.py \\
        --model_name_or_path moonshotai/Kimi-VL-A3B-Instruct \\
        --output_dir storage/prune/scores/kimi_gqa \\
        --score_type weight

Score types
-----------
activation  (default)
    channel_rms of act_fn(gate_proj(x)) * up_proj(x) for each routed token.

weight
    weight_rms of gate_proj + up_proj weights.  No forward pass needed.
"""

import argparse
import os
import sys
import types
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from observations.common import (
    build_dataset,
    compute_generic_expert_activation,
    custom_collate_fn,
    discover_layer_structure,
    ensure_dir,
    load_model_bundle,
    move_inputs_to_model_device,
    prepare_inputs,
    resolve_model_name_or_path,
)
from src.score_utils import channel_rms, safe_add_with_ema, weight_rms


# ---------------------------------------------------------------------------
# Channel-score accumulator
# ---------------------------------------------------------------------------

class ScoreAccumulator:
    """Accumulates per-channel importance scores for each (layer, expert).

    scores[layer_idx][expert_idx]: Tensor[I] (EMA-updated) or None (unseen).
    """

    def __init__(
        self,
        layer_to_num_experts: Dict[int, int],
        layer_to_num_channels: Dict[int, int],
    ) -> None:
        self.layer_to_num_experts = layer_to_num_experts
        self.layer_to_num_channels = layer_to_num_channels
        self.layers: List[int] = sorted(layer_to_num_experts.keys())
        self.scores: Dict[int, Dict[int, Optional[torch.Tensor]]] = {
            layer: {eid: None for eid in range(layer_to_num_experts[layer])}
            for layer in self.layers
        }
        self.counts: Dict[int, Dict[int, int]] = {
            layer: {eid: 0 for eid in range(layer_to_num_experts[layer])}
            for layer in self.layers
        }

    def update(
        self,
        layer_idx: int,
        expert_idx: int,
        score: torch.Tensor,
        ema: float,
    ) -> None:
        score = score.detach().cpu().float()
        self.scores[layer_idx][expert_idx] = safe_add_with_ema(
            self.scores[layer_idx][expert_idx], ema, score
        )
        self.counts[layer_idx][expert_idx] += 1

    def to_payload(self) -> dict:
        return {
            "scores": self.scores,
            "counts": self.counts,
            "layer_to_num_experts": self.layer_to_num_experts,
            "layer_to_num_channels": self.layer_to_num_channels,
            "layers": self.layers,
        }


# ---------------------------------------------------------------------------
# Contribution accumulator
# ---------------------------------------------------------------------------

class ContribAccumulator:
    """Accumulates per-expert contribution scores and per-layer losses.

    Gradient-free (always collected with score_type=activation):
      expert_usage[l][e]         : EMA routing fraction (tokens to expert / total)
      layerwise_repr_change[l]   : EMA of ||h_out - h_in|| / ||h_in|| per MoE layer

    Gradient-based (only when --collect_contrib, via block-forward pass):
      layerwise_loss[l]          : MLP-level reconstruction loss vs teacher
      expert_out_contrib[l][e]   : |down_out * down_out_grad|.sum() * usage

    All scalars are float (None until first update).
    """

    def __init__(self, layer_to_num_experts: Dict[int, int]) -> None:
        self.layers = sorted(layer_to_num_experts.keys())
        E = layer_to_num_experts

        self.expert_usage: Dict[int, Dict[int, Optional[float]]] = {
            l: {e: None for e in range(E[l])} for l in self.layers
        }
        self.layerwise_repr_change: Dict[int, Optional[float]] = {
            l: None for l in self.layers
        }
        self.layerwise_loss: Dict[int, Optional[float]] = {
            l: None for l in self.layers
        }
        self.expert_out_contrib: Dict[int, Dict[int, Optional[float]]] = {
            l: {e: None for e in range(E[l])} for l in self.layers
        }

    def _ema_scalar(self, old: Optional[float], new: float, ema: float) -> float:
        return new if old is None else old * ema + new * (1.0 - ema)

    def update_usage(self, l: int, e: int, usage: float, ema: float) -> None:
        self.expert_usage[l][e] = self._ema_scalar(self.expert_usage[l][e], usage, ema)

    def update_repr_change(self, l: int, rc: float, ema: float) -> None:
        self.layerwise_repr_change[l] = self._ema_scalar(
            self.layerwise_repr_change[l], rc, ema
        )

    def update_layer_loss(self, l: int, loss: float, ema: float) -> None:
        self.layerwise_loss[l] = self._ema_scalar(self.layerwise_loss[l], loss, ema)

    def update_expert_contrib(self, l: int, e: int, contrib: float, ema: float) -> None:
        self.expert_out_contrib[l][e] = self._ema_scalar(
            self.expert_out_contrib[l][e], contrib, ema
        )

    def to_payload(self) -> dict:
        return {
            "expert_usage": self.expert_usage,
            "layerwise_repr_change": self.layerwise_repr_change,
            "layerwise_loss": self.layerwise_loss,
            "expert_out_contrib": self.expert_out_contrib,
        }


# ---------------------------------------------------------------------------
# Modality affinity computation  (gradient-free routing computation)
# ---------------------------------------------------------------------------

def compute_modality_affinity(
    model,
    config,
    loader: DataLoader,
    prepare_fn,
    layer_to_num_experts: Dict[int, int],
    eps: float = 1e-8,
) -> Dict[int, Dict[int, float]]:
    """Run a lightweight forward-pass computation to compute expert modality affinity.

    For every MoE layer and expert, counts how many visual vs text tokens are
    routed to it, then computes::

        affinity[layer][expert] = (visual_freq - text_freq) / (visual_freq + text_freq + eps)

    Values are in [-1, +1]: +1 = purely visual, -1 = purely text, 0 = balanced.

    Uses `moe_text_mask` / `moe_media_mask` that Kimi-VL's patched model_forward
    sets on `layer.mlp` before each `moe_infer` call.
    """
    moe_layers = sorted(layer_to_num_experts.keys())

    # routing_counts[layer][modality] : Tensor[num_experts], float64, cumulative token count
    routing_counts: Dict[int, Dict[str, torch.Tensor]] = {
        l: {
            "text":   torch.zeros(layer_to_num_experts[l], dtype=torch.float64),
            "visual": torch.zeros(layer_to_num_experts[l], dtype=torch.float64),
        }
        for l in moe_layers
    }
    # token_counts[layer][modality] : float, total tokens seen per modality
    token_counts: Dict[int, Dict[str, float]] = {
        l: {"text": 0.0, "visual": 0.0} for l in moe_layers
    }

    # Patch moe_infer on every MoE layer to record routing counts
    orig_infers: Dict[int, object] = {}
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if layer_idx not in moe_layers:
            continue
        layer.mlp.freq_save_dir = "__observation__"
        layer.mlp.gate.layer_idx = layer_idx

        original_moe_infer = layer.mlp.moe_infer
        orig_infers[layer_idx] = original_moe_infer

        def _survey_moe_infer(
            self,
            x,
            topk_ids,
            topk_weight,
            *args,
            __orig=original_moe_infer,
            __lidx=layer_idx,
            __rc=routing_counts,
            __tc=token_counts,
            **kwargs,
        ):
            num_experts = len(self.experts)

            # Retrieve per-token modality masks (set by patched model_forward)
            text_mask = getattr(self, "moe_text_mask", None)
            visual_mask = getattr(self, "moe_media_mask", None)
            T = x.shape[0]
            if text_mask is None:
                text_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
            else:
                text_mask = text_mask.to(x.device).view(-1)[:T]
            if visual_mask is None:
                visual_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
            else:
                visual_mask = visual_mask.to(x.device).view(-1)[:T]

            __tc[__lidx]["text"]   += float(text_mask.sum().item())
            __tc[__lidx]["visual"] += float(visual_mask.sum().item())

            for modality, mask in (("text", text_mask), ("visual", visual_mask)):
                if mask.sum().item() == 0:
                    continue
                # topk_ids: [T, K] — collect all expert assignments for masked tokens
                selected = topk_ids[mask].reshape(-1).clamp(max=num_experts - 1).detach().cpu()
                counts = torch.bincount(selected, minlength=num_experts).to(torch.float64)
                __rc[__lidx][modality] += counts

            # Run original (no_grad) moe_infer for the actual output
            saved = getattr(self, "freq_save_dir", None)
            self.freq_save_dir = None
            out = __orig(x, topk_ids, topk_weight, *args, **kwargs)
            if saved is not None:
                self.freq_save_dir = saved
            return out

        layer.mlp.moe_infer = types.MethodType(_survey_moe_infer, layer.mlp)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Affinity computation", unit="batch"):
            inputs = prepare_fn(batch)
            inputs = move_inputs_to_model_device(model, inputs)
            try:
                model(**inputs, use_cache=False, return_dict=True)
            except Exception:
                pass

    # Restore original moe_infer
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if layer_idx not in orig_infers:
            continue
        orig = orig_infers[layer_idx]
        if orig is not None:
            layer.mlp.moe_infer = orig
        elif "moe_infer" in layer.mlp.__dict__:
            del layer.mlp.__dict__["moe_infer"]

    # Compute affinity: (visual_freq - text_freq) / (visual_freq + text_freq + eps)
    affinity: Dict[int, Dict[int, float]] = {}
    for l in moe_layers:
        n_exp = layer_to_num_experts[l]
        text_denom   = token_counts[l]["text"]
        visual_denom = token_counts[l]["visual"]

        text_freq   = routing_counts[l]["text"]   / max(text_denom,   1.0)
        visual_freq = routing_counts[l]["visual"] / max(visual_denom, 1.0)

        aff_tensor = (visual_freq - text_freq) / (visual_freq + text_freq + eps)
        affinity[l] = {e: float(aff_tensor[e].item()) for e in range(n_exp)}

    return affinity


# ---------------------------------------------------------------------------
# Activation-scoring hook  (gradient-free)
# ---------------------------------------------------------------------------

def attach_scoring_hooks(
    model,
    config,
    accumulator: ScoreAccumulator,
    contrib_acc: ContribAccumulator,
    ema: float,
    affinity: Optional[Dict[int, Dict[int, float]]] = None,
    affinity_mode: str = "threshold", # threshold | scalar
    affinity_threshold: float = 0.9,
) -> None:
    """Monkey-patch moe_infer on every MoE layer.

    Collects per-channel activation scores and gradient-free contribution
    proxies (expert_usage, layerwise_repr_change) in a single forward pass.
    No backward pass is needed.
    """
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            continue

        # Enable the mask-population path in kimi model_forward.
        layer.mlp.freq_save_dir = "__observation__"
        layer.mlp.gate.layer_idx = layer_idx

        # --- MLP-level repr_change hooks ---
        # Decoder layers are called with all keyword args → inp=() in a layer-level
        # forward hook. Use MLP pre/post hooks instead; MLP is always called
        # positionally so inp[0] = hidden_states reliably.
        def _make_mlp_repr_hooks(lidx, cacc, _ema):
            state: Dict = {}

            def _pre(module, inp):
                if inp and isinstance(inp[0], torch.Tensor):
                    state["h_in"] = inp[0].detach()

            def _post(module, inp, out):
                h_out = out[0] if isinstance(out, tuple) else out
                h_in = state.pop("h_in", None)
                if h_in is None or not isinstance(h_out, torch.Tensor):
                    return
                h_in_f = h_in.float()
                h_out_f = h_out.detach().float()
                norm_in = h_in_f.norm()
                diff = (h_out_f - h_in_f).norm()
                rc = float((diff / (norm_in + 1e-8)).item())
                cacc.update_repr_change(lidx, rc, _ema)

            return _pre, _post

        _mlp_pre, _mlp_post = _make_mlp_repr_hooks(layer_idx, contrib_acc, ema)
        layer.mlp.register_forward_pre_hook(_mlp_pre)
        layer.mlp.register_forward_hook(_mlp_post)

        # --- moe_infer replacement: channel scores + usage ---
        original_moe_infer = layer.mlp.moe_infer

        def observed_moe_infer(
            self,
            x,
            topk_ids,
            topk_weight,
            *args,
            __orig=original_moe_infer,
            __layer_idx=layer_idx,
            __acc=accumulator,
            __cacc=contrib_acc,
            __ema=ema,
            __affinity=affinity,
            __threshold=affinity_threshold,
            __mode=affinity_mode,
            **kwargs,
        ):
            num_experts = len(self.experts)
            total_tokens = max(x.shape[0], 1)

            expert_mask = F.one_hot(
                topk_ids.clamp(max=num_experts - 1), num_classes=num_experts
            ).permute(2, 1, 0)  # [E, topk, T]

            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for et in expert_hit:
                eid = int(et[0].item())
                _, token_idx = torch.where(expert_mask[eid])
                if token_idx.numel() == 0:
                    continue

                aff = __affinity.get(__layer_idx, {}).get(eid, 0.0) if __affinity is not None else 0.0

                if __mode == "threshold" and __affinity is not None:
                    # Hard filter: route this expert's tokens to preferred modality only
                    score_token_idx = token_idx
                    if aff > __threshold:
                        vis_flat = getattr(self, "moe_media_mask", None)
                        if vis_flat is not None:
                            vis_flat = vis_flat.to(x.device).view(-1)
                            keep = vis_flat[token_idx]
                            if keep.any():
                                score_token_idx = token_idx[keep]
                    elif aff < -__threshold:
                        txt_flat = getattr(self, "moe_text_mask", None)
                        if txt_flat is not None:
                            txt_flat = txt_flat.to(x.device).view(-1)
                            keep = txt_flat[token_idx]
                            if keep.any():
                                score_token_idx = token_idx[keep]

                    with torch.no_grad():
                        act = compute_generic_expert_activation(
                            self.experts[eid], x[score_token_idx]
                        )  # [T', I]
                    score = channel_rms(act)  # [I]

                elif __mode == "scalar" and __affinity is not None:
                    # Soft weighting: compute RMS separately for visual and text tokens
                    # that hit this expert, then blend by normalised affinity weights.
                    #
                    # Normalisation maps aff ∈ [-1, +1] to per-modality weights in [0, 1]:
                    #   vis_weight = max(aff, 0)   → 0 when text-only, 1 when visual-only
                    #   txt_weight = max(-aff, 0)  → 0 when visual-only, 1 when text-only
                    # Balanced experts (aff ≈ 0) contribute equal weight from both modalities.
                    vis_flat = getattr(self, "moe_media_mask", None)
                    txt_flat = getattr(self, "moe_text_mask", None)
                    T_seq = x.shape[0]
                    if vis_flat is not None:
                        vis_flat = vis_flat.to(x.device).view(-1)[:T_seq]
                    else:
                        vis_flat = torch.zeros(T_seq, dtype=torch.bool, device=x.device)
                    if txt_flat is not None:
                        txt_flat = txt_flat.to(x.device).view(-1)[:T_seq]
                    else:
                        txt_flat = torch.zeros(T_seq, dtype=torch.bool, device=x.device)

                    # Boolean masks over the tokens that actually hit this expert
                    vis_sel = vis_flat[token_idx]   # [T']
                    txt_sel = txt_flat[token_idx]   # [T']

                    with torch.no_grad():
                        act = compute_generic_expert_activation(
                            self.experts[eid], x[token_idx]
                        )  # [T', I]

                    vis_weight = max(aff, 0.0)    # ∈ [0, 1]
                    txt_weight = max(-aff, 0.0)   # ∈ [0, 1]

                    if vis_sel.any() and vis_weight > 0.0:
                        vis_score = channel_rms(act[vis_sel]) * vis_weight
                    else:
                        vis_score = None
                    if txt_sel.any() and txt_weight > 0.0:
                        txt_score = channel_rms(act[txt_sel]) * txt_weight
                    else:
                        txt_score = None

                    if vis_score is not None and txt_score is not None:
                        score = vis_score + txt_score
                    elif vis_score is not None:
                        score = vis_score
                    elif txt_score is not None:
                        score = txt_score
                    else:
                        # Balanced expert with no strong affinity: plain RMS over all tokens
                        score = channel_rms(act)

                else:
                    # No affinity or affinity_mode=None: plain RMS over all routed tokens
                    with torch.no_grad():
                        act = compute_generic_expert_activation(
                            self.experts[eid], x[token_idx]
                        )  # [T', I]
                    score = channel_rms(act)  # [I]

                __acc.update(__layer_idx, eid, score, __ema)

                # Gradient-free contribution proxy: routing fraction
                usage = float(token_idx.numel()) / float(total_tokens)
                __cacc.update_usage(__layer_idx, eid, usage, __ema)

            # Run original moe_infer (which has its own no_grad context).
            saved = getattr(self, "freq_save_dir", None)
            self.freq_save_dir = None
            try:
                return __orig(x, topk_ids, topk_weight, *args, **kwargs)
            finally:
                self.freq_save_dir = saved

        layer.mlp.moe_infer = observed_moe_infer.__get__(layer.mlp)


# ---------------------------------------------------------------------------
# Weight-based scoring  (no forward pass)
# ---------------------------------------------------------------------------

def collect_weight_scores(
    model,
    config,
    accumulator: ScoreAccumulator,
) -> None:
    """Compute weight_rms scores directly from expert parameters.

    No forward pass needed.  Fast, but ignores activation statistics.
    """
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            continue
        for eid, expert in enumerate(layer.mlp.experts):
            g = expert.gate_proj.weight  # [I, H]
            u = expert.up_proj.weight    # [I, H]
            score = (weight_rms(g) + weight_rms(u)) / 2.0  # [I]
            accumulator.update(layer_idx, eid, score, ema=1.0)


# ---------------------------------------------------------------------------
# Gradient-enabled MoE forward  (for block-forward contribution pass)
# ---------------------------------------------------------------------------

def _moe_infer_with_grad(self, x, topk_ids, topk_weight, **kwargs):
    """Gradient-enabled replacement for @torch.no_grad() moe_infer.

    Same routing logic as the original (sorted expert dispatch), but without
    the no_grad decorator so that loss.backward() reaches expert parameters.
    Simplified: no skip_expert_idx, no ep_rank offset, no freq tracking.

    x           : [T, H]   (sequence of hidden states after flattening)
    topk_ids    : [T, K]   expert assignments (K = num_experts_per_tok)
    topk_weight : [T, K]   routing weights
    Returns     : [T, H]
    """
    T, H = x.shape
    E = len(self.experts)
    K = topk_ids.shape[1]

    # Flatten routing: each (token, topk-slot) entry → (expert_id, weight)
    flat_ids = topk_ids.view(-1)      # [T*K]
    flat_w   = topk_weight.view(-1)   # [T*K]
    # Corresponding token index for each slot
    orig_tok = (
        torch.arange(T, device=x.device)
        .unsqueeze(1).expand(-1, K).reshape(-1)
    )  # [T*K]

    # Sort by expert ID (skip sentinel E which means "no expert")
    sort_idx  = flat_ids.argsort()           # [T*K]
    s_ids     = flat_ids[sort_idx]           # sorted expert IDs
    s_x       = x[orig_tok[sort_idx]]        # [T*K, H]
    s_w       = flat_w[sort_idx]             # [T*K]
    s_tok     = orig_tok[sort_idx]           # [T*K]

    # Count tokens per expert (including sentinel E)
    counts = torch.bincount(s_ids.clamp(min=0, max=E), minlength=E + 1)

    # Process each expert, collecting outputs and back-token indices
    out_parts  = []   # each: [n, H]
    tok_parts  = []   # each: [n]
    w_parts    = []   # each: [n]

    start = 0
    for eid in range(E):
        n = int(counts[eid].item())
        if n == 0:
            start += n
            continue
        x_eid = s_x[start : start + n]           # [n, H]
        out_eid = self.experts[eid](x_eid)        # [n, H]  ← gradient flows
        out_parts.append(out_eid)
        tok_parts.append(s_tok[start : start + n])
        w_parts.append(s_w[start : start + n])
        start += n

    if not out_parts:
        return torch.zeros(T, H, device=x.device, dtype=x.dtype)

    all_out = torch.cat(out_parts, dim=0)   # [N, H]  differentiable
    all_tok = torch.cat(tok_parts, dim=0)   # [N]
    all_w   = torch.cat(w_parts,   dim=0)   # [N]

    # Weighted scatter back to [T, H]  (index_add is differentiable)
    result = torch.zeros(T, H, device=x.device, dtype=all_out.dtype)
    # Cast weights to match all_out dtype before multiply (topk_weight may be float32
    # while expert outputs are bfloat16 — index_add requires identical dtypes).
    weighted = all_out * all_w.to(all_out.dtype).unsqueeze(-1)
    result = result.index_add(0, all_tok, weighted.to(result.dtype))
    return result


# ---------------------------------------------------------------------------
# Block-forward contribution collection  (gradient-based)
# ---------------------------------------------------------------------------

def _rel_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Relative L2 reconstruction loss (mean over valid tokens).

    pred, target : [B, T, H]  or  [T, H]
    attn_mask    : [B, T]  or  [T]  (1=valid), optional
    Returns      : scalar
    """
    p = pred.float().reshape(-1, pred.shape[-1])   # [N, H]
    t = target.float().reshape(-1, target.shape[-1])
    diff2 = (p - t).pow(2).sum(dim=-1)             # [N]
    base2 = t.pow(2).sum(dim=-1)                   # [N]
    rel   = diff2 / (base2 + eps)                  # [N]

    if attn_mask is not None:
        mask = attn_mask.float().reshape(-1).to(pred.device)
        valid = mask > 0
        return rel[valid].mean() if valid.any() else rel.mean()
    return rel.mean()


def collect_block_contrib_scores(
    model,
    config,
    loader,
    prepare_fn,
    dataset_name: str,
    ema: float,
    contrib_acc: ContribAccumulator,
) -> None:
    """MLP-level block-forward contribution collection (requires one extra pass).

    For each batch:
      1. Teacher pass (no grad): hook every MoE MLP to capture (h_in, h_out).
         Both tensors are kept on the model's device.
      2. For each MoE layer: temporarily patch layer.mlp.moe_infer with a
         gradient-enabled version, enable grad on expert params, forward
         layer.mlp(h_in), compute rel-L2 loss vs h_out, backward.
      3. Collect layerwise_loss and per-expert output contribution.
      4. Restore original moe_infer and param grad states.

    No deepcopy is performed — the original layer is used in-place during the
    contribution pass, then fully restored before moving to the next layer.
    """
    import traceback as _tb

    sorted_moe_layers = sorted(contrib_acc.layerwise_loss.keys())

    for batch in tqdm(loader, desc="Block contrib", unit="batch"):
        inputs = prepare_fn(batch)
        inputs = move_inputs_to_model_device(model, inputs)

        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.detach()   # stay on model device

        # ── Teacher pass: capture MLP input / output (kept on model device) ──
        mlp_captures: Dict[int, Dict] = {}

        def _make_mlp_hooks(lidx):
            def _pre(module, inp):
                if inp and isinstance(inp[0], torch.Tensor):
                    mlp_captures[lidx] = {"in": inp[0].detach(), "out": None}

            def _post(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor) and lidx in mlp_captures:
                    mlp_captures[lidx]["out"] = h.detach()

            return _pre, _post

        hooks = []
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            if layer_idx not in sorted_moe_layers:
                continue
            pre_fn, post_fn = _make_mlp_hooks(layer_idx)
            hooks.append(layer.mlp.register_forward_pre_hook(pre_fn))
            hooks.append(layer.mlp.register_forward_hook(post_fn))

        with torch.no_grad():
            try:
                model(**inputs, use_cache=False, return_dict=True)
            except Exception:
                pass

        for h in hooks:
            h.remove()

        if not mlp_captures:
            continue

        # ── Block-forward pass for each MoE layer (in-place, no deepcopy) ──
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            cap = mlp_captures.get(layer_idx)
            if cap is None or cap["out"] is None:
                continue

            h_in  = cap["in"]   # [B, T, H] on model device
            h_out = cap["out"]  # [B, T, H] on model device
            mlp   = layer.mlp

            # Save and replace moe_infer with gradient-enabled version
            orig_moe_infer = mlp.__dict__.get("moe_infer", None)
            mlp.moe_infer = types.MethodType(_moe_infer_with_grad, mlp)

            # Temporarily enable gradients on all expert parameters
            param_grad_states: Dict = {}
            for eid, expert in enumerate(mlp.experts):
                for name, param in expert.named_parameters():
                    param_grad_states[(eid, name)] = param.requires_grad
                    param.requires_grad_(True)

            # Register per-expert output forward+backward hooks
            expert_out_saves: Dict[int, Dict] = {
                e: {"out": None, "grad": None, "n_tokens": 0}
                for e in range(len(mlp.experts))
            }

            def _make_expert_hooks(eid, saves):
                def _fwd(module, inp, out):
                    saves[eid]["out"] = out.detach()
                    saves[eid]["n_tokens"] = out.shape[0]

                    def _bwd(g):
                        saves[eid]["grad"] = g.detach()

                    out.register_hook(_bwd)

                return _fwd

            expert_fwd_hooks = []
            for eid, expert in enumerate(mlp.experts):
                expert_fwd_hooks.append(
                    expert.register_forward_hook(_make_expert_hooks(eid, expert_out_saves))
                )

            # Forward with grad + backward
            mlp.zero_grad()
            try:
                with torch.enable_grad():
                    pred = mlp(h_in)
                    loss = _rel_l2_loss(pred, h_out, attn_mask)
                loss.backward()

                contrib_acc.update_layer_loss(layer_idx, float(loss.item()), ema)

                # Collect per-expert output contribution
                # T_total = total tokens in the batch (for usage fraction)
                T_total = int(h_in.shape[-2]) if h_in.ndim >= 2 else int(h_in.shape[0])
                for eid, saves in expert_out_saves.items():
                    if saves["out"] is not None and saves["grad"] is not None:
                        saliency = float(
                            (saves["out"].float() * saves["grad"].float())
                            .abs().sum().item()
                        )
                        usage = saves["n_tokens"] / max(T_total, 1)
                        contrib_acc.update_expert_contrib(
                            layer_idx, eid, saliency * usage, ema
                        )

            except Exception as exc:
                print(
                    f"[contrib] Layer {layer_idx} block-forward failed "
                    f"({type(exc).__name__}: {exc})\n"
                    + _tb.format_exc()
                )
            finally:
                for h in expert_fwd_hooks:
                    h.remove()
                mlp.zero_grad()

                # Restore moe_infer
                if orig_moe_infer is not None:
                    mlp.moe_infer = orig_moe_infer
                elif "moe_infer" in mlp.__dict__:
                    del mlp.__dict__["moe_infer"]

                # Restore param grad states
                for eid, expert in enumerate(mlp.experts):
                    for name, param in expert.named_parameters():
                        param.requires_grad_(param_grad_states.get((eid, name), False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect per-channel importance scores for Kimi-VL MoE experts."
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--dataset",
        type=str,
        default="gqa",
        choices=["gqa", "coco"],
    )
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="Random seed for sampling. Set to -1 to use sequential slice.",
    )
    p.add_argument(
        "--score_type",
        type=str,
        default="activation",
        choices=["activation", "weight"],
    )
    p.add_argument(
        "--ema",
        type=float,
        default=0.9,
        help="EMA decay for score accumulation.",
    )
    p.add_argument(
        "--collect_contrib",
        action="store_true",
        help=(
            "Also run the gradient-based block-forward pass to collect "
            "layerwise_loss and expert_out_contrib. Requires an extra pass "
            "per MoE layer with deepcopy; slower but provides richer signals."
        ),
    )
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument(
        "--modality_aware",
        action="store_true",
        help=(
            "Enable modality-aware channel scoring. Runs a preliminary "
            "routing-computation pass to compute per-expert affinity, then scores "
            "each expert using the strategy set by --affinity_mode. "
            "Saves affinity.pt to --output_dir."
        ),
    )
    p.add_argument(
        "--affinity_mode",
        type=str,
        default="threshold",
        choices=["threshold", "scalar"],
        help=(
            "How the affinity score is applied when --modality_aware is set. "
            "'threshold': hard-filter tokens to the preferred modality for experts "
            "whose |affinity| > --affinity_threshold (default). "
            "'scalar': soft-weight visual and text RMS scores by normalised affinity "
            "for all experts regardless of threshold."
        ),
    )
    p.add_argument(
        "--affinity_threshold",
        type=float,
        default=0.9,
        help=(
            "Affinity magnitude above which an expert is considered modality-specialised. "
            "Range [0, 1]. Default: 0.9."
        ),
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dir(args.output_dir)
    if not args.modality_aware:
        out_filename = "scores.pt"
    elif args.affinity_mode == "threshold":
        out_filename = f"affinity_threshold{args.affinity_threshold}_scores.pt"
    else:
        out_filename = f"affinity_{args.affinity_mode}_scores.pt"
    out_path = os.path.join(args.output_dir, out_filename)

    if os.path.exists(out_path) and not args.force_recompute:
        print(f"[collect_scores] Found existing scores at {out_path} "
              "Pass --force_recompute to overwrite.")
        return

    # Compute modality affinity if modality-aware scoring is requested
    affinity: Optional[Dict[int, Dict[int, float]]] = None

    print(f"[collect_scores] Resolving model: {args.model_name_or_path}")
    bundle = load_model_bundle(args.model_name_or_path)
    model = bundle.model
    config = model.config.text_config

    layer_to_num_experts, layer_to_num_channels = discover_layer_structure(bundle)
    accumulator = ScoreAccumulator(layer_to_num_experts, layer_to_num_channels)
    contrib_acc  = ContribAccumulator(layer_to_num_experts)

    print(
        f"[collect_scores] Discovered {len(layer_to_num_experts)} MoE layers, "
        f"{sum(layer_to_num_experts.values())} experts total."
    )

    if args.score_type == "weight":
        print("[collect_scores] score_type=weight: scoring from expert weight norms "
              "(no forward pass).")
        collect_weight_scores(model, config, accumulator)

    else:  # activation
        import random
        data = build_dataset(args.dataset, bundle.family)
        pool = list(range(args.start_idx, len(data)))
        if args.subset_seed >= 0:
            rng = random.Random(args.subset_seed)
            indices = rng.sample(pool, min(args.num_samples, len(pool)))
        else:
            indices = pool[: args.num_samples]

        subset = Subset(data, indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        def _prepare_fn(batch):
            return prepare_inputs(bundle, batch, args.dataset)

        # ── Optional: preliminary routing-computation pass to compute modality affinity ──
        if args.modality_aware:
            print("[collect_scores] modality_aware=True: running routing-computation pass "
                  "to compute expert modality affinity.")
            affinity = compute_modality_affinity(
                model, config, loader, _prepare_fn, layer_to_num_experts,
            )
            aff_path = os.path.join(args.output_dir, "affinity.pt")
            torch.save({"affinity": affinity, "threshold": args.affinity_threshold}, aff_path)
            print(f"[collect_scores] Affinity saved to {aff_path}.")

        print(f"[collect_scores] score_type=activation: attaching hooks, "
              f"dataset={args.dataset}.")
        attach_scoring_hooks(
            model, config, accumulator, contrib_acc, args.ema,
            affinity=affinity,
            affinity_mode=args.affinity_mode,
            affinity_threshold=args.affinity_threshold,
        )

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting scores", unit="batch"):
                inputs = _prepare_fn(batch)
                inputs = move_inputs_to_model_device(model, inputs)
                model(**inputs, use_cache=False, return_dict=True)

        # ── Optional: gradient-based block contribution pass ──
        if args.collect_contrib:
            print("[collect_scores] collect_contrib=True: running MLP block-forward "
                  "contribution pass (slower).")

            collect_block_contrib_scores(
                model, config,
                loader=loader,
                prepare_fn=_prepare_fn,
                dataset_name=args.dataset,
                ema=args.ema,
                contrib_acc=contrib_acc,
            )

    # Summarise coverage
    total = sum(len(v) for v in accumulator.scores.values())
    seen  = sum(
        1
        for layer_scores in accumulator.scores.values()
        for s in layer_scores.values()
        if s is not None
    )
    print(f"[collect_scores] Experts with channel scores: {seen}/{total}")

    # Summarise gradient-free contrib scores
    usage_seen = sum(
        1
        for d in contrib_acc.expert_usage.values()
        for v in d.values()
        if v is not None
    )
    print(f"[collect_scores] Experts with usage scores: {usage_seen}/{total}")
    rc_seen = sum(
        1 for v in contrib_acc.layerwise_repr_change.values() if v is not None
    )
    print(f"[collect_scores] Layers with repr_change: {rc_seen}/{len(contrib_acc.layers)}")

    if args.collect_contrib:
        loss_seen = sum(
            1 for v in contrib_acc.layerwise_loss.values() if v is not None
        )
        print(f"[collect_scores] Layers with block loss: {loss_seen}/{len(contrib_acc.layers)}")

    # Merge and save
    payload = accumulator.to_payload()
    payload.update(contrib_acc.to_payload())
    payload["score_type"]          = args.score_type
    payload["collect_contrib"]     = args.collect_contrib
    payload["model_name_or_path"]  = args.model_name_or_path
    payload["dataset"]             = args.dataset
    payload["num_samples"]         = args.num_samples
    payload["modality_aware"]      = args.modality_aware
    payload["affinity_mode"]       = args.affinity_mode
    payload["affinity_threshold"]  = args.affinity_threshold

    torch.save(payload, out_path)
    print(f"[collect_scores] Set SCORES_PATH={out_path} bash scripts/run_prune.sh to use the score in pruning. ")


if __name__ == "__main__":
    main()
