import argparse
import os
from typing import Any, Dict, List, Tuple

import torch

from observations.o2.response_levels import build_response_levels_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact text/visual response tensors for channel/expert/layer levels.")
    parser.add_argument("--raw_stats_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_channel_pairs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument(
        "--include_pairs",
        type=str,
        default="1:1,12:12,24:24",
        help="Comma-separated layer:expert pairs to prefer, e.g. '1:1,12:12,24:24'.",
    )
    return parser.parse_args()


def _parse_include_pairs(raw: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not raw.strip():
        return pairs
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid pair format: {item}")
        l_str, e_str = item.split(":", 1)
        pairs.append((int(l_str), int(e_str)))
    return pairs


def main() -> None:
    args = parse_args()
    raw_stats = torch.load(args.raw_stats_path, map_location="cpu", weights_only=False)
    include_pairs = _parse_include_pairs(args.include_pairs)
    payload = build_response_levels_payload(
        raw_stats=raw_stats,
        num_channel_pairs=args.num_channel_pairs,
        seed=args.seed,
        include_pairs=include_pairs,
    )
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.output_path)
    size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"[done] saved: {args.output_path}")
    print(f"[done] size: {size_mb:.3f} MB")
    print("[done] channel-level pairs:",
          [(int(x['layer']), int(x['expert'])) for x in payload["channel_level"]])
    print("[done] expert-level shape:", tuple(payload["expert_level"]["text"].shape))
    print("[done] layer-level shape:", tuple(payload["layer_level"]["text"].shape))


if __name__ == "__main__":
    main()
