import os
from typing import Any, Dict

import torch
from src.base.shared_utils import _print
from src.base.datasets.load_data import load_datasets
from src.base.shared_utils.safe_isinstance import (
    _get_num_experts,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_num_hidden_size,
)

def _maybe_move_model(model, device: str, args) -> None:
    """
    Avoid breaking quantized or device-mapped models.
    """
    # Heuristic: when using 4bit/8bit, or model has hf_device_map, do not force .to()
    if getattr(args, "load_in_4bit", False) or getattr(args, "load_in_8bit", False):
        return
    if hasattr(model, "hf_device_map"):
        return
    model.to(device)


def prepare_run_context(
    *,
    args,
    model,
    tokenizer,
    scores_dir: str,
    mask_method_kwargs: Dict[str, Any],
    HI_ratio_kwargs: Dict[str, Any],
    calib_dataset_name: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    torch.set_float32_matmul_precision("high")

    device = args.device
    dtype = args.dtype

    model.eval()
    _maybe_move_model(model, device, args)

    E = _get_num_experts(model)
    I = _get_moe_intermediate_size(model)
    L = _get_num_hidden_layers(model)
    H = _get_num_hidden_size(model)

    if verbose:
        _print(f"num_experts={E}, intermediate_size={I}, hidden_size={H}, num_layers={L}")

    calib_dataset = load_datasets(
        calib_dataset_name,
        tokenizer,
        max_samples=args.calib_batches * args.batch_size,
        max_length=args.max_seq_length,
    )
    if verbose:
        _print("✅ Loaded calib dataset")
    
    # 延迟导入以避免循环导入
    from src.generate_mask import prepare_scores
        
    intermediate_scores, hidden_scores, expertwise_scores, L, E, I, H, loss_based_kwargs = prepare_scores(
        scores_dir=scores_dir,
        mask_method_kwargs=mask_method_kwargs,
        HI_ratio_kwargs=HI_ratio_kwargs,
        prune_ratio=args.prune_kwargs['prune_ratio'],
        prune_hidden=args.prune_kwargs['prune_hidden'],
        prune_gqa=args.prune_kwargs['prune_gqa'],
        device=device,
        verbose=verbose,
    )

    return dict(
        device=device,
        dtype=dtype,
        E=E,
        I=I,
        H=H,
        L=L,
        calib_dataset=calib_dataset,
        intermediate_scores=intermediate_scores,
        hidden_scores=hidden_scores,
        expertwise_scores=expertwise_scores,
        layerwise_loss=loss_based_kwargs["layerwise_loss"],
        smooth_times=loss_based_kwargs["smooth_times"],
        layerwise_keep_plan=loss_based_kwargs["layerwise_keep_plan"],
    )

