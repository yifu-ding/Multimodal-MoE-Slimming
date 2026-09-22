#!/usr/bin/env python3
"""Artifact based state for the serial Kimi direct-mask benchmark campaign."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/vllm_ours/kimi/direct-mask"
STATE = ROOT / ".automation/kimi_direct"
TASKS = (
    "gqa", "textvqa_val", "coco2017_cap_val_local", "chartqa", "mmstar",
    "mmbench_en_dev_static_local", "mmvet", "mme", "realworldqa", "videomme",
    "longvideobench_val_v", "video_mmmu_local", "egoschema_subset", "mvbench_available_3800",
)
RATIOS = ("p30", "p50")
MASK_HASHES = {
    "p30": "13fb772fc899958f77ee6570abbe84b4a9efc966b52e9fd3d3628ed0d5f088da",
    "p50": "68d0698150eef9d32c6a0dce8d2f404c81929211f2e0752b261868592dcc86b3",
}
HALF_EXPECTED = {
    "gqa": 6289, "textvqa_val": 2500, "coco2017_cap_val_local": 2500,
    "chartqa": 1250, "mmstar": 750, "mmbench_en_dev_static_local": 2165,
    "mmvet": 218, "mme": 1188, "realworldqa": 500, "videomme": 1350,
    "longvideobench_val_v": 669, "video_mmmu_local": 500,
    "egoschema_subset_local": 500, "mvbench_available_3800": 1900,
}


def mask_path(ratio):
    return ROOT / f"runtime/mask_plans/kimi-{ratio}-direct.pt"


def run_dir(ratio, half=False):
    suffix = "random-half-seed42" if half else "full"
    return RESULTS / f"{ratio}-{suffix}"


def actual_task(task):
    return "egoschema_subset_local" if task == "egoschema_subset" else task


def valid_mask(ratio):
    path = mask_path(ratio)
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == MASK_HASHES[ratio]


def valid_benchmark(task, ratio):
    if not valid_mask(ratio):
        return False
    name = actual_task(task)
    for half in (False, True):
        directory = run_dir(ratio, half=half)
        marker = directory / "status" / f"{name}.complete"
        config = directory / "run_config.txt"
        if not marker.is_file() or not config.is_file():
            continue
        marker_text = marker.read_text(errors="replace")
        config_text = config.read_text(errors="replace")
        if (f"task={name}\n" not in marker_text or "signature=" not in marker_text
                or "limit=full\n" not in config_text
                or f"maes_mask_plan={mask_path(ratio)}\n" not in config_text):
            continue
        if half and ("random_subset_fraction=0.5\n" not in config_text
                     or "random_subset_min_samples=500\n" not in config_text
                     or "random_subset_seed=42\n" not in config_text):
            continue
        if not half and "random_subset_fraction=0.5\n" in config_text:
            continue
        outputs = directory / "tasks" / name
        results = list(outputs.rglob("*_results.json"))
        if name == "video_mmmu_local":
            sample_pattern = "*_samples_video_mmmu_*_local.jsonl"
        elif name == "mvbench_available_3800":
            sample_pattern = "*_samples_mvbench_*.jsonl"
        else:
            sample_pattern = f"*_samples_{name}.jsonl"
        samples = list(outputs.rglob(sample_pattern))
        try:
            if not results or not samples or not any(isinstance(json.loads(path.read_text()), dict) for path in results):
                continue
            sample_count = sum(sum(1 for line in path.open() if line.strip()) for path in samples)
            if half and sample_count != HALF_EXPECTED[name]:
                continue
            if sample_count > 0:
                return True
        except (OSError, ValueError):
            continue
    return False


def valid_judge(ratio):
    directory = run_dir(ratio) / "local_judge"
    for name in ("mmvet", "mmbench", "video_mmmu"):
        path = directory / f"{name}_summary.json"
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        if (summary.get("num_failed") != 0 or summary.get("num_scored", 0) <= 0
                or summary.get("num_scored") != summary.get("num_source_samples")):
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
        return 0 if all(valid_mask(r) for r in RATIOS) else 2
    if args.action == "check":
        if not args.task or not args.ratio:
            parser.error("check needs --task and --ratio")
        return 0 if valid_benchmark(args.task, args.ratio) else 1
    complete = sum(valid_benchmark(t, r) for t, r in items())
    judges = sum(valid_judge(r) for r in RATIOS)
    if args.action == "completion":
        if not all(valid_mask(r) for r in RATIOS):
            return 2
        return 0 if complete == len(TASKS) * len(RATIOS) and judges == 2 else 1
    current_file = STATE / "current.txt"
    current = current_file.read_text().strip() if current_file.exists() else "pending"
    elapsed = int(time.time() - current_file.stat().st_mtime) if current_file.exists() else 0
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        resources = "GPU MiB=" + "/".join(line.strip() for line in gpu.splitlines()[:4])
    except (OSError, subprocess.SubprocessError):
        resources = "GPU=unknown"
    print(f"{complete}/28 benchmarks ({complete / 28:.0%}), judges={judges}/2, "
          f"current={current}, stage_elapsed={elapsed}s, {resources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
