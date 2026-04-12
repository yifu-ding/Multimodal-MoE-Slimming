"""Phase 2: Generate keep-masks and apply structural channel pruning to Kimi-VL.

Loads pre-computed channel scores (from collect_scores.py), generates per-expert
top-k keep-masks, applies structural pruning in-place (resizing gate_proj / up_proj /
down_proj), updates the model config, and saves with save_pretrained.

The pruned model is a smaller, fully valid Kimi-VL model that can be loaded with
AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True).

Usage
-----
    python src/prune.py \\
        --model_path moonshotai/Kimi-VL-A3B-Instruct \\
        --scores_path storage/prune/scores/kimi_gqa/channel_scores.pt \\
        --prune_ratio 0.30 \\
        --output_dir storage/prune/pruned_models/kimi_gqa_p30

Pruning scope
-------------
- Only routed experts (layer.mlp.experts[eid]) are pruned.
- Shared experts (layer.mlp.shared_experts) are left untouched.
- Layer 0 is dense (no MoE); only layers 1–26 are pruned.
- Hidden-dimension (H) pruning is NOT applied; only intermediate (I) is pruned.
"""

import argparse
import glob
import math
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
from src.generate_mask import generate_masks as build_masks_pipeline


_ROUTED_EXPERT_LOAD_PATCH = """

# ---------------------------------------------------------------------------
# MoDES patch: restore routed experts from checkpoint tensor shapes
# ---------------------------------------------------------------------------

import re as _modes_re
from transformers import modeling_utils as _modes_modeling_utils


_MODES_ROUTED_WEIGHT_RE = _modes_re.compile(
    r"^(.*)\\.experts\\.(\\d+)\\.(gate_proj|up_proj|down_proj)\\.weight$"
)
_MODES_ROUTER_PARAM_RE = _modes_re.compile(
    r"^(.*)\\.gate\\.(weight|e_score_correction_bias)$"
)


def _modes_resolve_moe_module(model, moe_path):
    try:
        module = model.get_submodule(moe_path)
    except AttributeError:
        module, _ = _modes_modeling_utils.get_module_from_name(model, moe_path)
    if hasattr(module, "gate") and hasattr(module, "experts"):
        return module
    if hasattr(module, "mlp") and hasattr(module.mlp, "gate") and hasattr(module.mlp, "experts"):
        return module.mlp
    raise AttributeError(
        f"Could not resolve MoE module from '{moe_path}' "
        f"(got {type(module).__name__})."
    )


def _modes_resolve_experts_container(model, moe_path):
    return _modes_resolve_moe_module(model, moe_path).experts


def _modes_fix_gate_layout(module, target_n_routed_experts):
    gate = module.gate
    target_n_routed_experts = int(target_n_routed_experts)
    gate.n_routed_experts = target_n_routed_experts
    if hasattr(module, "num_experts_per_tok"):
        module.num_experts_per_tok = min(int(module.num_experts_per_tok), target_n_routed_experts)
    if hasattr(gate, "top_k"):
        gate.top_k = min(int(gate.top_k), target_n_routed_experts)
    if hasattr(module, "experts_per_rank"):
        module.experts_per_rank = len(module.experts)
    if hasattr(gate, "experts_len"):
        gate.experts_len = len(module.experts)
    if getattr(gate, "topk_method", None) == "noaux_tc":
        n_group = int(getattr(gate, "n_group", 1))
        per_group = target_n_routed_experts // max(n_group, 1) if n_group > 0 else 0
        topk_group = int(getattr(gate, "topk_group", 1))
        if (
            n_group <= 0
            or target_n_routed_experts % n_group != 0
            or per_group < 2
            or topk_group > n_group
        ):
            gate.topk_method = "greedy"


def _modes_maybe_resize_active_experts(model, param_name, tensor):
    match = _MODES_ROUTER_PARAM_RE.match(param_name)
    if match is None:
        return

    moe_path, gate_param_name = match.groups()
    module = _modes_resolve_moe_module(model, moe_path)
    base_config = getattr(model.config, "text_config", model.config)
    target_n_routed_experts = int(tensor.shape[0])
    current_n_routed_experts = len(module.experts)

    if current_n_routed_experts == target_n_routed_experts:
        _modes_fix_gate_layout(module, target_n_routed_experts)
        return

    if current_n_routed_experts > target_n_routed_experts:
        module.experts = nn.ModuleList(
            [module.experts[i] for i in range(target_n_routed_experts)]
        )
    else:
        for _ in range(target_n_routed_experts - current_n_routed_experts):
            module.experts.append(
                DeepseekV3MLP(
                    base_config,
                    intermediate_size=base_config.moe_intermediate_size,
                ).to(device=tensor.device, dtype=tensor.dtype)
            )

    if gate_param_name == "weight":
        target_hidden = int(tensor.shape[1])
        gate_dtype = tensor.dtype
        gate_device = tensor.device
        new_weight = nn.Parameter(torch.empty((target_n_routed_experts, target_hidden), device=gate_device, dtype=gate_dtype))
        module.gate.weight = new_weight
    if gate_param_name == "e_score_correction_bias":
        module.gate.e_score_correction_bias = nn.Parameter(
            torch.empty((target_n_routed_experts,), device=tensor.device, dtype=tensor.dtype)
        )

    _modes_fix_gate_layout(module, target_n_routed_experts)


def _modes_maybe_resize_routed_expert(model, param_name, tensor):
    match = _MODES_ROUTED_WEIGHT_RE.match(param_name)
    if match is None:
        return

    moe_path, expert_idx_str, proj_name = match.groups()
    expert_idx = int(expert_idx_str)
    _modes_maybe_resize_active_experts(
        model,
        f"{moe_path}.gate.weight",
        torch.empty((expert_idx + 1, int(tensor.shape[1] if proj_name in ('gate_proj', 'up_proj') else tensor.shape[0])), device=tensor.device, dtype=tensor.dtype),
    )
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


try:
    from accelerate.utils import modeling as _modes_accel_modeling

    _modes_orig_set_module_tensor_to_device = (
        _modes_accel_modeling.set_module_tensor_to_device
    )

    def _modes_patched_set_module_tensor_to_device(
        module,
        tensor_name,
        device,
        value=None,
        dtype=None,
        fp16_statistics=None,
        tied_params_map=None,
        non_blocking=False,
        clear_cache=True,
    ):
        if value is not None:
            _modes_maybe_resize_routed_expert(module, tensor_name, value)
        return _modes_orig_set_module_tensor_to_device(
            module,
            tensor_name,
            device,
            value=value,
            dtype=dtype,
            fp16_statistics=fp16_statistics,
            tied_params_map=tied_params_map,
            non_blocking=non_blocking,
            clear_cache=clear_cache,
        )

    _modes_accel_modeling.set_module_tensor_to_device = (
        _modes_patched_set_module_tensor_to_device
    )
    if hasattr(_modes_modeling_utils, "set_module_tensor_to_device"):
        _modes_modeling_utils.set_module_tensor_to_device = (
            _modes_patched_set_module_tensor_to_device
        )
except (ImportError, AttributeError):
    pass

try:
    from transformers import core_model_loading as _modes_core_loading

    _modes_orig_set_param = _modes_core_loading.set_param_for_module

    def _modes_patched_set_param(
        model,
        target_name,
        param_value,
        loading_info,
        distributed_operation,
        hf_quantizer,
    ):
        _modes_maybe_resize_routed_expert(model, target_name, param_value)
        return _modes_orig_set_param(
            model,
            target_name,
            param_value,
            loading_info,
            distributed_operation,
            hf_quantizer,
        )

    _modes_core_loading.set_param_for_module = _modes_patched_set_param
except (ImportError, AttributeError):
    pass

if hasattr(_modes_modeling_utils, "_load_parameter_into_model"):
    _modes_orig_load_parameter_into_model = (
        _modes_modeling_utils._load_parameter_into_model
    )

    def _modes_load_parameter_into_model(model, param_name, tensor):
        _modes_maybe_resize_routed_expert(model, param_name, tensor)
        return _modes_orig_load_parameter_into_model(model, param_name, tensor)

    _modes_modeling_utils._load_parameter_into_model = (
        _modes_load_parameter_into_model
    )
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


def _shrink_kimi_router_for_active_experts(module: nn.Module, keep_mask: torch.Tensor) -> int:
    keep_mask = keep_mask.to(dtype=torch.bool)
    gate = module.gate
    old_num_experts = int(keep_mask.numel())
    n_active = int(keep_mask.sum().item())
    if n_active == 0:
        raise RuntimeError("All experts in this layer were fully pruned.")

    keep_idx = torch.nonzero(keep_mask.to(gate.weight.device), as_tuple=False).view(-1)
    gate.weight = nn.Parameter(gate.weight.data.index_select(0, keep_idx).contiguous())

    if hasattr(gate, "e_score_correction_bias") and gate.e_score_correction_bias is not None:
        gate.e_score_correction_bias = nn.Parameter(
            gate.e_score_correction_bias.data.index_select(0, keep_idx).contiguous()
        )

    gate.n_routed_experts = n_active
    if hasattr(module, "num_experts_per_tok"):
        module.num_experts_per_tok = min(int(module.num_experts_per_tok), n_active)
    if hasattr(gate, "top_k"):
        gate.top_k = min(int(gate.top_k), n_active)

    if hasattr(module, "experts_per_rank"):
        module.experts_per_rank = len(module.experts)

    if hasattr(gate, "experts_len"):
        gate.experts_len = len(module.experts)

    # noaux_tc requires a valid grouped layout; if the pruned expert count no longer
    # fits the original grouping assumptions, fall back to greedy top-k.
    if getattr(gate, "topk_method", None) == "noaux_tc":
        n_group = int(getattr(gate, "n_group", 1))
        per_group = n_active // max(n_group, 1) if n_group > 0 else 0
        topk_group = int(getattr(gate, "topk_group", 1))
        if (
            n_group <= 0
            or n_active % n_group != 0
            or per_group < 2
            or topk_group > n_group
        ):
            gate.topk_method = "greedy"

    return old_num_experts - n_active


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_structural_pruning(
    model: nn.Module,
    masks: Dict[int, torch.Tensor],
    config,
) -> None:
    """Structurally prune routed expert intermediate dimensions in-place.

    For each routed expert at every MoE layer:
        gate_proj : [I, H] -> [I', H]   (keep rows where mask=True)
        up_proj   : [I, H] -> [I', H]   (same rows)
        down_proj : [H, I] -> [H, I']   (keep cols where mask=True)

    Shared experts (layer.mlp.shared_experts) are never touched.

    Ported and simplified from:
        LLM-Distillation/src/prune/apply/slimming/expert_slim.py
        slim_moe_inter_and_optional_hidden_keepmask_inplace (lines 252-370)
    """
    layers = model.language_model.model.layers
    pbar = tqdm(total=len(layers), desc="Pruning experts", unit="layer")

    params_removed = 0
    params_kept = 0
    inactive_experts = 0
    shrink_gate_cnt = 0

    for layer_idx, layer in enumerate(layers):
        pbar.update(1)
        if not _is_moe_layer(layer_idx, config):
            continue
        if layer_idx not in masks:
            continue

        layer_mask = masks[layer_idx]  # [E, I]
        old_num_experts = len(layer.mlp.experts)
        if layer_mask.shape[0] != old_num_experts:
            raise RuntimeError(
                f"Layer {layer_idx}: mask expert dim={int(layer_mask.shape[0])} "
                f"but model has {old_num_experts} experts."
            )
        layer_active_expert = torch.ones(old_num_experts, dtype=torch.bool)

        for eid, expert in enumerate(layer.mlp.experts):
            m_inter = layer_mask[eid].to(
                device=expert.gate_proj.weight.device, dtype=torch.bool
            )  # [I]
            I_prime = int(m_inter.sum().item())
            if I_prime == 0:
                layer_active_expert[eid] = False
                I_old = expert.gate_proj.out_features
                H = expert.gate_proj.in_features
                params_removed += int(I_old * H * 3)
                continue

            dtype = expert.gate_proj.weight.dtype
            device = expert.gate_proj.weight.device
            H = expert.gate_proj.in_features

            W_gate = expert.gate_proj.weight.data[m_inter, :]  # [I', H]
            W_up   = expert.up_proj.weight.data[m_inter, :]    # [I', H]
            W_down = expert.down_proj.weight.data[:, m_inter]  # [H, I']

            I_old = expert.gate_proj.out_features
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

        n_active = int(layer_active_expert.sum().item())
        if n_active == 0:
            raise RuntimeError(
                f"All experts in layer {layer_idx} were fully pruned. "
                "Adjust masks to keep at least one expert."
            )
        inactive_experts += old_num_experts - n_active
        if n_active != old_num_experts:
            keep_eids = torch.nonzero(layer_active_expert, as_tuple=False).view(-1).tolist()
            layer.mlp.experts = nn.ModuleList([layer.mlp.experts[eid] for eid in keep_eids])
            shrink_gate_cnt += _shrink_kimi_router_for_active_experts(layer.mlp, layer_active_expert)
        
    pbar.close()
    total = params_removed + params_kept
    pct = 100.0 * params_removed / total if total > 0 else 0.0
    print(
        f"[prune] Expert params removed: {params_removed:,}  "
        f"kept: {params_kept:,}  "
        f"({pct:.1f}% removed)"
    )
    if shrink_gate_cnt > 0:
        print(
            f"[prune] Removed {inactive_experts} fully pruned experts and shrank "
            f"{shrink_gate_cnt} gate entries."
        )


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def update_config(model: nn.Module, masks: Dict[int, torch.Tensor]) -> int:
    """Keep config moe_intermediate_size unchanged and return routed expert I'."""
    # Sample I' from the first MoE layer's first expert
    first_layer = sorted(masks.keys())[0]
    I_prime = int(masks[first_layer][0].sum().item())
    print(
        "[prune] Leaving text_config.moe_intermediate_size unchanged so shared "
        f"experts keep their original shape (routed experts saved with I'={I_prime})."
    )
    return I_prime


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
        section_start = code.rfind(
            "# ---------------------------------------------------------------------------",
            0,
            marker_idx,
        )
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
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HF model id or local path to the original (un-pruned) model.",
    )
    p.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to channel_scores.pt produced by collect_scores.py.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--prune_ratio",
        type=float,
        default=0.30,
        help="Fraction of intermediate channels to remove (0.0 = no pruning, 0.5 = half).",
    )
    p.add_argument(
        "--inter_method",
        type=str,
        default="uniform",
        help="Inter-layer planner method for src/generate_mask.",
    )
    p.add_argument(
        "--intra_method",
        type=str,
        default="uniform",
        help="Intra-layer planner method for src/generate_mask.",
    )
    p.add_argument(
        "--intra_expert_metric",
        type=str,
        default="activation",
        help="Per-channel metric to use from expert_scores.pth.",
    )
    p.add_argument("--align_inter", type=int, default=0)
    p.add_argument("--min_per_expert", type=int, default=0)
    p.add_argument("--modality_aware", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Generate masks from the new score pipeline ----
    print(f"[prune] Loading scores from {args.scores_path}")
    mask_result = build_masks_pipeline(
        scores_dir=args.scores_path,
        prune_kwargs={
            "prune_ratio": args.prune_ratio,
            "mask_method_kwargs": {
                "inter_layer_method": args.inter_method,
                "intra_layer_method": args.intra_method,
                "intra_expert_metric": args.intra_expert_metric,
            },
            "adjust_masks_kwargs": {
                "align_inter": args.align_inter,
                "min_per_expert": args.min_per_expert,
            },
            "modality_aware": args.modality_aware,
            "prune_hidden": False,
            "prune_gqa": False,
        },
        device="cpu",
        verbose=True,
    )
    mask_tensor = mask_result["intermediate_masks"]
    layers = [int(layer) for layer in mask_result.get("layers", list(range(mask_tensor.shape[0])))]
    masks = {
        layer_idx: mask_tensor[pos].detach().cpu().bool()
        for pos, layer_idx in enumerate(layers)
    }
    all_k = mask_result["K_E_inter"].detach().cpu()
    first_layer = layers[0]
    I_orig = int(mask_tensor.shape[-1])
    I_prime = int(all_k[0, 0].item())
    print(
        f"[prune] Generated masks for {len(layers)} layers. "
        f"I_orig={I_orig}, I_prime min={int(all_k.min().item())} "
        f"max={int(all_k.max().item())} mean={float(all_k.float().mean().item()):.1f}"
    )

    # ---- Load clean model (no monkey-patches) ----
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
    config = model.config.text_config

    # ---- Apply structural pruning ----
    apply_structural_pruning(model, masks, config)

    # ---- Update config ----
    update_config(model, masks)

    # ---- Save ----
    print(f"[prune] Saving pruned model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    for py_file in glob.glob(os.path.join(model_path, "*.py")):
        dst = os.path.join(args.output_dir, os.path.basename(py_file))
        shutil.copy2(py_file, dst)
        print(f"[prune] Copied {os.path.basename(py_file)}")
    patch_saved_remote_code(args.output_dir)
    print(f"[prune] Done.  Pruned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
