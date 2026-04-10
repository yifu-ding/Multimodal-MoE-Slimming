"""Phase 2: Generate keep-masks and apply structural channel pruning to Kimi-VL.

Loads pre-computed channel scores (from collect_scores.py), generates per-expert
top-k keep-masks using a two-level planning strategy (inter-layer + intra-layer),
applies structural pruning in-place, keeps the original config-level MoE sizes,
and saves with save_pretrained.

Reloading relies on a patched modeling_kimi_vl.py saved alongside the checkpoint:
routed experts are rebuilt from checkpoint tensor shapes before weight loading,
while shared experts keep the original config-defined size.

Usage
-----
    # Uniform pruning (default, same as before)
    python src/prune.py \\
        --model_path moonshotai/Kimi-VL-A3B-Instruct \\
        --scores_path storage/prune/scores/kimi_gqa/scores.pt \\
        --prune_ratio 0.30 \\
        --output_dir storage/prune/pruned_models/kimi_gqa_p30

    # Coverage inter-layer + coverage intra-layer
    python src/prune.py \\
        --model_path moonshotai/Kimi-VL-A3B-Instruct \\
        --scores_path storage/prune/scores/kimi_gqa/scores.pt \\
        --prune_ratio 0.30 \\
        --inter_method coverage \\
        --intra_method coverage \\
        --output_dir storage/prune/pruned_models/kimi_gqa_p30_cov_cov

Planning methods
----------------
Inter-layer  (--inter_method):
  uniform    : identical prune_ratio for every MoE layer  [default]
  coverage   : binary-search for saliency coverage fraction s — layers with
               concentrated scores are pruned more aggressively

Intra-layer  (--intra_method):
  expertwise : each expert independently top-k  [default]
  layerwise  : pool E×I scores per layer, global top-k back-assigned to experts
  global     : budget proportional to each expert's total score mass (cross-layer)
  coverage   : per-expert coverage-based selection, binary-search for scale t

Config / loading notes
----------------------
- moe_intermediate_size in config is left unchanged.
- Shared experts are not pruned and still use config-defined shapes.
- Routed experts are re-created from checkpoint tensor shapes at load time via a
  patched modeling_kimi_vl.py copied into the saved directory.

Pruning scope
-------------
- Only routed experts (layer.mlp.experts[eid]) are structurally pruned.
- Shared experts (layer.mlp.shared_experts) are left untouched.
- Layer 0 is dense (no MoE); only layers 1–26 are pruned.
"""

import argparse
import glob
import os
import shutil
import sys
from typing import Dict

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from models.kimi import _normalize_kimi_config_for_remote_code
from observations.common import resolve_model_name_or_path
from src.planners import (
    INTER_LAYER_METHODS,
    INTRA_LAYER_METHODS,
    generate_masks,
)


_ROUTED_EXPERT_LOAD_PATCH = """

# ---------------------------------------------------------------------------
# MoDES patch: restore routed experts from checkpoint tensor shapes
# ---------------------------------------------------------------------------

import re as _modes_re
from transformers import modeling_utils as _modes_modeling_utils


_MODES_ROUTED_WEIGHT_RE = _modes_re.compile(
    r"^(.*)\\.experts\\.(\\d+)\\.(gate_proj|up_proj|down_proj)\\.weight$"
)


def _modes_resolve_experts_container(model, moe_path):
    # Try model.get_submodule first (new HF transformers API)
    try:
        module = model.get_submodule(moe_path)
    except AttributeError:
        module, _ = _modes_modeling_utils.get_module_from_name(model, moe_path)
    if hasattr(module, "experts"):
        return module.experts
    if hasattr(module, "mlp") and hasattr(module.mlp, "experts"):
        return module.mlp.experts
    raise AttributeError(
        f"Could not resolve experts container from '{moe_path}' "
        f"(got {type(module).__name__})."
    )


def _modes_maybe_resize_routed_expert(model, param_name, tensor):
    match = _MODES_ROUTED_WEIGHT_RE.match(param_name)
    if match is None:
        return

    moe_path, expert_idx_str, proj_name = match.groups()
    expert_idx = int(expert_idx_str)
    experts = _modes_resolve_experts_container(model, moe_path)
    expert = experts[expert_idx]
    base_config = getattr(model.config, "text_config", model.config)

    if proj_name in ("gate_proj", "up_proj"):
        target_intermediate = int(tensor.shape[0])
        target_hidden = int(tensor.shape[1])
    else:
        target_hidden = int(tensor.shape[0])
        target_intermediate = int(tensor.shape[1])

    if expert is not None:
        current_hidden = int(expert.gate_proj.in_features)
        current_intermediate = int(expert.gate_proj.out_features)
        if (
            current_hidden == target_hidden
            and current_intermediate == target_intermediate
        ):
            return
        expert_device = expert.gate_proj.weight.device
        expert_dtype = expert.gate_proj.weight.dtype
    else:
        expert_device = tensor.device
        expert_dtype = tensor.dtype

    new_expert = DeepseekV3MLP(
        expert.config if expert is not None else base_config,
        hidden_size=target_hidden,
        intermediate_size=target_intermediate,
    )
    new_expert = new_expert.to(device=expert_device, dtype=expert_dtype)
    experts[expert_idx] = new_expert


# ---------------------------------------------------------------------------
# Patch 1: new-style HF transformers (core_model_loading.set_param_for_module)
# This is the primary loading path in transformers >= 4.50.
# ---------------------------------------------------------------------------
try:
    from transformers import core_model_loading as _modes_core_loading
    _modes_orig_set_param = _modes_core_loading.set_param_for_module

    def _modes_patched_set_param(model, target_name, param_value, loading_info,
                                  distributed_operation, hf_quantizer):
        _modes_maybe_resize_routed_expert(model, target_name, param_value)
        return _modes_orig_set_param(model, target_name, param_value, loading_info,
                                     distributed_operation, hf_quantizer)

    _modes_core_loading.set_param_for_module = _modes_patched_set_param
except (ImportError, AttributeError):
    pass

# ---------------------------------------------------------------------------
# Patch 2: old-style HF transformers (_load_parameter_into_model)
# Kept for backward compatibility with transformers < 4.50.
# ---------------------------------------------------------------------------
if hasattr(_modes_modeling_utils, "_load_parameter_into_model"):
    _modes_orig_load_parameter_into_model = _modes_modeling_utils._load_parameter_into_model

    def _modes_load_parameter_into_model(model, param_name, tensor):
        _modes_maybe_resize_routed_expert(model, param_name, tensor)
        return _modes_orig_load_parameter_into_model(model, param_name, tensor)

    _modes_modeling_utils._load_parameter_into_model = _modes_load_parameter_into_model
"""


# ---------------------------------------------------------------------------
# MoE layer predicate
# ---------------------------------------------------------------------------

def _is_moe_layer(layer_idx: int, config) -> bool:
    return (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_structural_pruning(
    model: nn.Module,
    masks: Dict[int, torch.Tensor],
    config,
) -> None:
    """Structurally prune routed expert intermediate dims in-place.

    For each routed expert at every MoE layer:
        gate_proj : [I, H] -> [I', H]   (keep rows where mask=True)
        up_proj   : [I, H] -> [I', H]   (same rows)
        down_proj : [H, I] -> [H, I']   (keep cols where mask=True)

    Shared experts are left untouched.

    """
    layers = model.language_model.model.layers
    pbar = tqdm(total=len(layers), desc="Pruning experts", unit="layer")

    params_removed = 0
    params_kept = 0

    for layer_idx, layer in enumerate(layers):
        pbar.update(1)
        if not _is_moe_layer(layer_idx, config):
            continue
        if layer_idx not in masks:
            continue

        layer_mask = masks[layer_idx]   # [E, I]

        # ---- Prune routed experts ----
        for eid, expert in enumerate(layer.mlp.experts):
            m_inter = layer_mask[eid].to(
                device=expert.gate_proj.weight.device, dtype=torch.bool
            )
            I_prime = int(m_inter.sum().item())
            if I_prime == 0:
                m_inter[0] = True
                I_prime = 1

            dtype  = expert.gate_proj.weight.dtype
            device = expert.gate_proj.weight.device
            H      = expert.gate_proj.in_features
            I_old  = expert.gate_proj.out_features

            W_gate = expert.gate_proj.weight.data[m_inter, :]
            W_up   = expert.up_proj.weight.data[m_inter, :]
            W_down = expert.down_proj.weight.data[:, m_inter]

            params_removed += int((I_old - I_prime) * H * 2 + H * (I_old - I_prime))
            params_kept    += int(I_prime * H * 2 + H * I_prime)

            new_gate = nn.Linear(H, I_prime, bias=False, device=device, dtype=dtype)
            new_up   = nn.Linear(H, I_prime, bias=False, device=device, dtype=dtype)
            new_down = nn.Linear(I_prime, H, bias=False, device=device, dtype=dtype)
            new_gate.weight = nn.Parameter(W_gate.contiguous())
            new_up.weight   = nn.Parameter(W_up.contiguous())
            new_down.weight = nn.Parameter(W_down.contiguous())
            expert.gate_proj = new_gate
            expert.up_proj   = new_up
            expert.down_proj = new_down

    pbar.close()
    total = params_removed + params_kept
    pct = 100.0 * params_removed / total if total > 0 else 0.0
    print(
        f"[prune] Expert params removed: {params_removed:,}  "
        f"kept: {params_kept:,}  "
        f"({pct:.1f}% removed)"
    )


def patch_saved_remote_code(output_dir: str) -> bool:
    """Patch copied modeling_kimi_vl.py to restore routed experts from ckpt shapes."""
    fp = os.path.join(output_dir, "modeling_kimi_vl.py")
    if not os.path.exists(fp):
        print(f"[prune] WARNING: {fp} not found; skipping modeling patch")
        return False

    with open(fp, "r", encoding="utf-8") as f:
        code = f.read()

    marker = "MoDES patch: restore routed experts from checkpoint tensor shapes"
    new_patch = _ROUTED_EXPERT_LOAD_PATCH.lstrip("\n")

    if new_patch in code:
        print("[prune] modeling_kimi_vl.py already patched.")
        return False

    if marker in code:
        marker_idx = code.index(marker)
        section_start = code.rfind("# ---------------------------------------------------------------------------", 0, marker_idx)
        if section_start == -1:
            section_start = marker_idx
        updated = code[:section_start].rstrip() + "\n\n" + new_patch
        action = "Updated"
    else:
        updated = code.rstrip() + "\n\n" + new_patch
        action = "Patched"

    with open(fp, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"[prune] {action} modeling_kimi_vl.py for variable routed-expert reload.")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structural channel pruning of Kimi-VL MoE experts."
    )
    p.add_argument("--model_path", type=str, required=True,
                   help="HF model id or local path to the original (un-pruned) model.")
    p.add_argument("--scores_path", type=str, required=True,
                   help="Path to scores.pt produced by collect_scores.py.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--prune_ratio", type=float, default=0.30,
                   help="Fraction of intermediate channels to remove globally.")

    # Planning strategy
    p.add_argument("--inter_method", type=str, default="uniform",
                   choices=list(INTER_LAYER_METHODS),
                   help="Inter-layer budget allocation: uniform | coverage.")
    p.add_argument("--intra_method", type=str, default="expertwise",
                   choices=list(INTRA_LAYER_METHODS),
                   help="Intra-layer channel selection: expertwise | layerwise | global | coverage.")
    p.add_argument(
        "--layerwise_weight_source",
        type=str,
        default=None,
        choices=["repr_change", "block_loss", None],
        help=(
            "Source for layerwise_weights passed to coverage inter-layer planner. "
            "repr_change: use layerwise_repr_change from scores file (gradient-free). "
            "block_loss: use layerwise_loss from scores file (requires --collect_contrib). "
            "None (default): unweighted coverage."
        ),
    )
    p.add_argument(
        "--evict_min_channels",
        type=int,
        default=0,
        help=(
            "If > 0, run a post-planning eviction pass: any routed expert assigned "
            "fewer than this many channels is fully evicted (kept count → 0), and "
            "the freed budget is redistributed to the surviving experts within each "
            "layer. Set to 0 (default) to disable. "
            "Typical value: 64 (≈4.5%% of I=1408 for Kimi-VL)."
        ),
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load scores ----
    print(f"[prune] Loading scores from {args.scores_path}")
    score_payload = torch.load(args.scores_path, weights_only=False)
    scores                = score_payload["scores"]
    layer_to_num_experts  = score_payload["layer_to_num_experts"]
    layer_to_num_channels = score_payload["layer_to_num_channels"]
    print(
        f"[prune] Scores: {len(scores)} MoE layers, "
        f"score_type={score_payload.get('score_type', 'unknown')}"
    )

    # ---- Build layerwise_weights (optional, for coverage inter-layer) ----
    layerwise_weights = None
    if args.inter_method == "coverage":
        if args.layerwise_weight_source is None:
            raise ValueError("layerwise_weight_source is required for coverage inter-layer")
        sorted_layers = sorted(scores.keys())
        if args.layerwise_weight_source == "repr_change":
            rc = score_payload.get("layerwise_repr_change", {})
            vals = [rc.get(l, None) for l in sorted_layers]
            if all(v is not None for v in vals):
                layerwise_weights = torch.tensor(vals, dtype=torch.float32)
                print(f"[prune] Using layerwise_weights from repr_change: "
                      f"min={layerwise_weights.min():.4f} max={layerwise_weights.max():.4f}")
            else:
                print("[prune] WARNING: repr_change not fully available; "
                      "falling back to unweighted coverage.")
        elif args.layerwise_weight_source == "block_loss":
            bl = score_payload.get("layerwise_loss", {})
            vals = [bl.get(l, None) for l in sorted_layers]
            if all(v is not None for v in vals):
                layerwise_weights = torch.tensor(vals, dtype=torch.float32)
                print(f"[prune] Using layerwise_weights from block_loss: "
                      f"min={layerwise_weights.min():.4f} max={layerwise_weights.max():.4f}")
            else:
                print("[prune] WARNING: layerwise_loss not available "
                      "(was --collect_contrib used?); falling back to unweighted coverage.")

    # ---- Generate masks ----
    print(
        f"[prune] Generating masks: "
        f"prune_ratio={args.prune_ratio}, "
        f"inter={args.inter_method}, intra={args.intra_method}"
        + (f", layerwise_weights={args.layerwise_weight_source}"
           if layerwise_weights is not None else "")
        + (f", evict_min={args.evict_min_channels}"
           if args.evict_min_channels > 0 else "")
    )
    masks = generate_masks(
        scores,
        args.prune_ratio,
        layer_to_num_experts,
        layer_to_num_channels,
        inter_method=args.inter_method,
        intra_method=args.intra_method,
        layerwise_weights=layerwise_weights,
        evict_min_channels=args.evict_min_channels,
    )

    # Print summary
    first_layer = sorted(masks.keys())[0]
    I_orig = layer_to_num_channels[first_layer]
    all_k = [int(masks[l][e].sum()) for l in sorted(masks.keys())
             for e in range(masks[l].shape[0])]
    print(
        f"[prune] I_orig={I_orig}, "
        f"I_prime: min={min(all_k)} max={max(all_k)} mean={sum(all_k)/len(all_k):.1f}"
    )

    # ---- Load clean model ----
    model_path = resolve_model_name_or_path(args.model_path)
    print(f"[prune] Loading model from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = _normalize_kimi_config_for_remote_code(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    text_config = model.config.text_config

    # ---- Apply structural pruning ----
    apply_structural_pruning(model, masks, text_config)
    print(
        "[prune] Leaving text_config.moe_intermediate_size unchanged so shared "
        "experts keep their original shape."
    )

    # ---- Save ----
    print(f"[prune] Saving pruned model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Copy custom modeling Python files from original snapshot
    for py_file in glob.glob(os.path.join(model_path, "*.py")):
        dst = os.path.join(args.output_dir, os.path.basename(py_file))
        shutil.copy2(py_file, dst)
        print(f"[prune] Copied {os.path.basename(py_file)}")

    patch_saved_remote_code(args.output_dir)

    print(f"[prune] Done. Set MODEL_PATH={args.output_dir} bash scripts/eval_baseline_kimi_gqa.sh to evaluate the pruned model.")


if __name__ == "__main__":
    main()
