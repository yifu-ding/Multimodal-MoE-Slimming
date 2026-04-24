import argparse
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for path in (REPO_PARENT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from observations.o2.response_levels import plot_response_levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load response_levels_compact.pt and render plots directly.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input_path, map_location="cpu", weights_only=False)
    saved_paths = plot_response_levels(payload, output_dir=args.output_dir, prefix=args.prefix)
    print(f"[done] loaded: {args.input_path}")
    for path in saved_paths["channel_level"]:
        print(f"[done] channel-level plot: {path}")
    if saved_paths["expert_level"] is not None:
        print(f"[done] expert-level plot: {saved_paths['expert_level']}")
    if saved_paths["layer_level"] is not None:
        print(f"[done] layer-level plot: {saved_paths['layer_level']}")


if __name__ == "__main__":
    main()
