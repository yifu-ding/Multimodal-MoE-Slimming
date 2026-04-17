import argparse
import math
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from tqdm import trange

from src.calibration.representation_distill.common import ensure_dir, seed_everything, utc_now_iso


def _sample_rows(tensor: torch.Tensor, count: int) -> torch.Tensor:
    if count >= tensor.shape[0]:
        return tensor
    indices = torch.randint(0, tensor.shape[0], (count,), device=tensor.device)
    return tensor.index_select(0, indices)


def _compute_stat_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
) -> dict[str, torch.Tensor]:
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    teacher_mean = teacher_flat.mean(dim=0)
    synth_mean = synth_flat.mean(dim=0)
    teacher_var = teacher_flat.var(dim=0, unbiased=False)
    synth_var = synth_flat.var(dim=0, unbiased=False)
    teacher_energy = teacher_flat.pow(2).mean(dim=0)
    synth_energy = synth_flat.pow(2).mean(dim=0)

    return {
        "mean": (teacher_mean - synth_mean).pow(2).mean(),
        "var": (teacher_var - synth_var).pow(2).mean(),
        "pos": (teacher_batch.mean(dim=0) - synthetic_hidden.mean(dim=0)).pow(2).mean(),
        "energy": (teacher_energy - synth_energy).pow(2).mean(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill a small synthetic hidden calibration set from teacher hidden cache."
    )
    parser.add_argument("--teacher_cache_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--synthetic_size", type=int, default=256)
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_std", type=float, default=1e-3)
    parser.add_argument("--lambda_mean", type=float, default=1.0)
    parser.add_argument("--lambda_var", type=float, default=1.0)
    parser.add_argument("--lambda_pos", type=float, default=1.0)
    parser.add_argument("--lambda_energy", type=float, default=0.5)
    parser.add_argument("--log_interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    ensure_dir(os.path.dirname(args.output_path))

    cache_payload = torch.load(args.teacher_cache_path, map_location="cpu")
    teacher_cache = cache_payload["teacher_cache"].float()
    teacher_meta = cache_payload["metadata"]

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_cache = teacher_cache.to(device)

    init_indices = torch.randint(0, teacher_cache.shape[0], (args.synthetic_size,), device=device)
    synthetic_hidden = torch.nn.Parameter(teacher_cache.index_select(0, init_indices).clone())
    if args.init_std > 0:
        synthetic_hidden.data.add_(torch.randn_like(synthetic_hidden) * args.init_std)

    optimizer = torch.optim.Adam([synthetic_hidden], lr=args.lr)
    history = []
    final_losses = None

    for step in trange(args.train_steps, desc="Distilling synthetic hidden", leave=False):
        teacher_batch = _sample_rows(teacher_cache, args.teacher_batch_size)
        stat_losses = _compute_stat_losses(teacher_batch, synthetic_hidden)
        total_loss = (
            args.lambda_mean * stat_losses["mean"]
            + args.lambda_var * stat_losses["var"]
            + args.lambda_pos * stat_losses["pos"]
            + args.lambda_energy * stat_losses["energy"]
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        final_losses = {
            "total": float(total_loss.detach().cpu().item()),
            "mean": float(stat_losses["mean"].detach().cpu().item()),
            "var": float(stat_losses["var"].detach().cpu().item()),
            "pos": float(stat_losses["pos"].detach().cpu().item()),
            "energy": float(stat_losses["energy"].detach().cpu().item()),
        }
        if step % args.log_interval == 0 or step == args.train_steps - 1:
            history.append({"step": step, **final_losses})

    synthetic_hidden_cpu = synthetic_hidden.detach().cpu().float()
    attention_mask = torch.ones(
        synthetic_hidden_cpu.shape[0],
        synthetic_hidden_cpu.shape[1],
        dtype=torch.long,
    )
    payload = {
        "synthetic_hidden": synthetic_hidden_cpu,
        "attention_mask": attention_mask,
        "metadata": {
            "method": "multimodal_representation_level_calibration_distillation",
            "teacher_cache_path": args.teacher_cache_path,
            "teacher_metadata": teacher_meta,
            "synthetic_size": args.synthetic_size,
            "compressed_length": int(synthetic_hidden_cpu.shape[1]),
            "hidden_size": int(synthetic_hidden_cpu.shape[2]),
            "dtype": "float32",
            "position_ids_strategy": "sequential_from_attention_mask",
            "train_steps": args.train_steps,
            "teacher_batch_size": args.teacher_batch_size,
            "lr": args.lr,
            "loss_weights": {
                "mean": args.lambda_mean,
                "var": args.lambda_var,
                "pos": args.lambda_pos,
                "energy": args.lambda_energy,
            },
            "final_losses": final_losses,
            "history": history,
            "created_at": utc_now_iso(),
        },
    }
    torch.save(payload, args.output_path)
    print(f"[representation_distill] Saved synthetic calibration hidden to {args.output_path}")


if __name__ == "__main__":
    main()

