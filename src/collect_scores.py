"""Phase 1: Per-channel importance score collection for Kimi-VL MoE experts.

Runs forward passes over VL calibration data and accumulates per-channel
activation scores (or weight-norm scores) for every routed expert in every
MoE layer.  Output is saved to ``channel_scores.pt``.

Usage
-----
    python src/collect_scores.py \\
        --model_name_or_path moonshotai/Kimi-VL-A3B-Instruct \\
        --output_dir storage/prune/scores/kimi_gqa \\
        --num_samples 128 \\
        --score_type activation

Score types
-----------
activation  (default)
    channel_rms of act_fn(gate_proj(x)) * up_proj(x) for each routed token.
    No backward pass needed.  Follows the hook pattern in observations/common.py.

weight
    weight_rms of gate_proj + up_proj weights.  No forward pass needed.
    Cheapest option; useful for a quick smoke-test.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

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
# Score accumulator
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
# Activation-scoring hook
# ---------------------------------------------------------------------------

def attach_scoring_hooks(
    model,
    config,
    accumulator: ScoreAccumulator,
    ema: float,
) -> None:
    """Monkey-patch moe_infer on every MoE layer to collect channel activations.

    Follows the same pattern as attach_kimi_observer in observations/common.py,
    but accumulates channel_rms(activation) instead of per-modality abs-sums.
    No text/visual split — all tokens are treated uniformly (naive baseline).
    """
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            continue

        # Enable the mask-population path so that moe_text_mask / moe_media_mask
        # are written by the upstream kimi model_forward (needed for the patched
        # moe_infer not to crash when it reads those attributes).
        layer.mlp.freq_save_dir = "__observation__"
        layer.mlp.gate.layer_idx = layer_idx

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
            __ema=ema,
            **kwargs,
        ):
            num_experts = len(self.experts)
            expert_mask = F.one_hot(
                topk_ids.clamp(max=num_experts - 1), num_classes=num_experts
            ).permute(2, 1, 0)  # [E, topk, T]

            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for et in expert_hit:
                eid = int(et[0].item())
                _, token_idx = torch.where(expert_mask[eid])
                if token_idx.numel() == 0:
                    continue
                with torch.no_grad():
                    act = compute_generic_expert_activation(
                        self.experts[eid], x[token_idx]
                    )  # [T', I]
                score = channel_rms(act)  # [I]
                __acc.update(__layer_idx, eid, score, __ema)

            # Temporarily disable freq_save_dir to avoid triggering the
            # frequency-logging branch in the original moe_infer body.
            saved = getattr(self, "freq_save_dir", None)
            self.freq_save_dir = None
            try:
                return __orig(x, topk_ids, topk_weight, *args, **kwargs)
            finally:
                self.freq_save_dir = saved

        layer.mlp.moe_infer = observed_moe_infer.__get__(layer.mlp)


# ---------------------------------------------------------------------------
# Weight-based scoring (no forward pass)
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
        help="EMA decay for score accumulation (only used with score_type=activation).",
    )
    p.add_argument("--force_recompute", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, "channel_scores.pt")

    if os.path.exists(out_path) and not args.force_recompute:
        print(f"[collect_scores] Found existing scores at {out_path}. "
              "Pass --force_recompute to overwrite.")
        return

    print(f"[collect_scores] Resolving model: {args.model_name_or_path}")
    bundle = load_model_bundle(args.model_name_or_path)
    model = bundle.model
    config = model.config.text_config

    layer_to_num_experts, layer_to_num_channels = discover_layer_structure(bundle)
    accumulator = ScoreAccumulator(layer_to_num_experts, layer_to_num_channels)

    print(
        f"[collect_scores] Discovered {len(layer_to_num_experts)} MoE layers, "
        f"{sum(layer_to_num_experts.values())} experts total."
    )

    if args.score_type == "weight":
        print("[collect_scores] score_type=weight: scoring from expert weight norms (no forward pass).")
        collect_weight_scores(model, config, accumulator)
    else:
        print(f"[collect_scores] score_type=activation: attaching hooks, loading dataset={args.dataset}.")
        attach_scoring_hooks(model, config, accumulator, args.ema)

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

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting scores", unit="batch"):
                inputs = prepare_inputs(bundle, batch, args.dataset)
                inputs = move_inputs_to_model_device(model, inputs)
                model(**inputs, use_cache=False, return_dict=True)

    # Summarise coverage
    total = sum(len(v) for v in accumulator.scores.values())
    seen = sum(
        1
        for layer_scores in accumulator.scores.values()
        for s in layer_scores.values()
        if s is not None
    )
    print(f"[collect_scores] Experts with scores: {seen}/{total}.")

    payload = accumulator.to_payload()
    payload["score_type"] = args.score_type
    payload["model_name_or_path"] = args.model_name_or_path
    payload["dataset"] = args.dataset
    payload["num_samples"] = args.num_samples
    torch.save(payload, out_path)
    print(f"[collect_scores] Saved scores to {out_path}.")


if __name__ == "__main__":
    main()
