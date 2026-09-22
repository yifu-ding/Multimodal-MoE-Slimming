#!/usr/bin/env python3
"""Artifact state for the Kimi method1-only Router direct-mask campaign."""

import argparse
import functools
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/vllm_ours/kimi/method1-router-direct"
STATE = ROOT / ".automation/kimi_method1_router"
TASKS = (
    "gqa", "textvqa_val", "coco2017_cap_val_local", "chartqa", "mmstar",
    "mmbench_en_dev_static_local", "mmvet", "mme", "realworldqa", "videomme",
    "longvideobench_val_v", "video_mmmu_local", "egoschema_subset", "mvbench_available_3800",
)
RATIOS = ("p30", "p50")
MASK_HASHES = {
    "p30": "62daebac3db86626d3948283b8c6db7578ef1707044540bccbfbb28e75a2381e",
    "p50": "1d7600eb482254172e6bbe65200a7a68c2963517ac45981ad83ab40388aeb7bb",
}
HALF_EXPECTED = {
    "gqa": 6289, "textvqa_val": 2500, "coco2017_cap_val_local": 2500,
    "chartqa": 1250, "mmstar": 750, "mmbench_en_dev_static_local": 2165,
    "mmvet": 218, "mme": 1188, "realworldqa": 500, "videomme": 1350,
    "longvideobench_val_v": 669, "video_mmmu_local": 500,
    "egoschema_subset_local": 500, "mvbench_available_3800": 1900,
}


def mask_path(ratio):
    return ROOT / f"runtime/mask_plans/kimi-{ratio}-method1-router-direct.pt"


def run_dir(ratio):
    return RESULTS / f"{ratio}-random-half-seed42"


def actual_task(task):
    return "egoschema_subset_local" if task == "egoschema_subset" else task


@functools.lru_cache(maxsize=None)
def valid_mask(ratio):
    path = mask_path(ratio)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != MASK_HASHES[ratio]:
        return False
    try:
        plan = torch.load(path, map_location="cpu", weights_only=False)
        config = plan["pruning_config"]
        return (
            plan["model"] == "moonshotai/Kimi-VL-A3B-Instruct"
            and config["mask_method_kwargs"]["intra_layer_method"] == "router"
            and config["modality_aware"] is True
            and config["adjust_masks_kwargs"] == {"align_inter": 0, "min_per_expert": 0}
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def valid_benchmark(task, ratio):
    if not valid_mask(ratio):
        return False
    name = actual_task(task)
    directory = run_dir(ratio)
    marker = directory / "status" / f"{name}.complete"
    config = directory / "run_config.txt"
    if not marker.is_file() or not config.is_file():
        return False
    marker_text = marker.read_text(errors="replace")
    config_text = config.read_text(errors="replace")
    if (
        f"task={name}\n" not in marker_text
        or "signature=" not in marker_text
        or "limit=full\n" not in config_text
        or "random_subset_fraction=0.5\n" not in config_text
        or "random_subset_min_samples=500\n" not in config_text
        or "random_subset_seed=42\n" not in config_text
        or f"maes_mask_plan={mask_path(ratio)}\n" not in config_text
    ):
        return False
    outputs = directory / "tasks" / name
    results = list(outputs.rglob("*_results.json"))
    if name == "video_mmmu_local":
        sample_pattern = "*_samples_video_mmmu_*_local.jsonl"
    elif name == "mvbench_available_3800":
        sample_pattern = "*_samples_mvbench_*.jsonl"
    else:
        sample_pattern = f"*_samples_{name}.jsonl"
    samples = list(outputs.rglob(sample_pattern))
    if not results or not samples:
        return False
    try:
        if not any(isinstance(json.loads(path.read_text()), dict) for path in results):
            return False
        sample_count = sum(sum(1 for line in path.open() if line.strip()) for path in samples)
        return sample_count == HALF_EXPECTED[name]
    except (OSError, ValueError):
        return False
    return False


def valid_judge(ratio):
    directory = run_dir(ratio) / "local_judge"
    for name in ("mmvet", "mmbench", "video_mmmu"):
        path = directory / f"{name}_summary.json"
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        if (
            summary.get("num_failed") != 0
            or summary.get("num_scored", 0) <= 0
            or summary.get("num_scored") != summary.get("num_source_samples")
        ):
            return False
    return True


def items():
    for task in TASKS:
        for ratio in RATIOS:
            yield task, ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "completion", "progress", "validate-masks"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--ratio", choices=RATIOS)
    args = parser.parse_args()
    if args.action == "validate-masks":
        return 0 if all(valid_mask(ratio) for ratio in RATIOS) else 2
    if args.action == "check":
        if not args.task or not args.ratio:
            parser.error("check needs --task and --ratio")
        return 0 if valid_benchmark(args.task, args.ratio) else 1
    complete = sum(valid_benchmark(task, ratio) for task, ratio in items())
    judges = sum(valid_judge(ratio) for ratio in RATIOS)
    if args.action == "completion":
        if not all(valid_mask(ratio) for ratio in RATIOS):
            return 2
        return 0 if complete == len(TASKS) * len(RATIOS) and judges == 2 else 1
    current_file = STATE / "current.txt"
    current = current_file.read_text().strip() if current_file.exists() else "queued"
    elapsed = int(time.time() - current_file.stat().st_mtime) if current_file.exists() else 0
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        resources = "GPU MiB=" + "/".join(line.strip() for line in gpu.splitlines()[:4])
    except (OSError, subprocess.SubprocessError):
        resources = "GPU=unknown"
    print(
        f"{complete}/28 benchmarks ({complete / 28:.0%}), judges={judges}/2, "
        f"current={current}, stage_elapsed={elapsed}s, {resources}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
