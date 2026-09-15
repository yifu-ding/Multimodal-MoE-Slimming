#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    description: str


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    description: str


DATASETS: Dict[str, DatasetSpec] = {
    "textvqa": DatasetSpec("lmms-lab/textvqa", "TextVQA (val)"),
    "chartqa": DatasetSpec("lmms-lab/ChartQA", "ChartQA"),
    "mmstar": DatasetSpec("Lin-Chen/MMStar", "MMStar"),
    "mmbench": DatasetSpec("lmms-lab/MMBench", "MMBench (dev, EN)"),
    "mmvet": DatasetSpec("lmms-lab/MMVet", "MMVet"),
    "mme": DatasetSpec("lmms-lab/MME", "MME"),
    "realworldqa": DatasetSpec("lmms-lab/RealWorldQA", "RealWorldQA"),
    "coco2017-cap": DatasetSpec(
        "lmms-lab/COCO-Caption2017",
        "COCO2017-Cap (val)",
    ),
    "mvbench": DatasetSpec("OpenGVLab/MVBench", "MVBench"),
    "egoschema": DatasetSpec("lmms-lab/egoschema", "EgoSchema"),
    "videomme": DatasetSpec("lmms-lab/Video-MME", "VideoMME"),
    "longvideobench": DatasetSpec(
        "longvideobench/LongVideoBench",
        "LongVideoBench (val)",
    ),
    "video-mmmu": DatasetSpec("lmms-lab/VideoMMMU", "Video-MMMU"),
    "gqa": DatasetSpec("lmms-lab/GQA", "GQA"),
    # "m4-instruct": DatasetSpec("lmms-lab/M4-Instruct-Data", "M4-Instruct"), # too big for calibration, using steaming
}


MODELS: Dict[str, ModelSpec] = {
    "deepseek-vl2-small": ModelSpec(
        "deepseek-ai/deepseek-vl2-small",
        "DeepSeek-VL2-Small",
    ),
    "kimi-vl-a3b-instruct": ModelSpec(
        "moonshotai/Kimi-VL-A3B-Instruct",
        "Kimi-VL-A3B-Instruct",
    ),
    "qwen3-vl-30b-a3b-instruct": ModelSpec(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen3-VL-30B-A3B-Instruct",
    ),
    "internvl3_5-30b-a3b-hf": ModelSpec(
        "OpenGVLab/InternVL3_5-30B-A3B-HF",
        "InternVL-3.5-30B-A3B-HF",
    ),
    "gemma-4-26b-a4b": ModelSpec(
        "google/gemma-4-26B-A4B",
        "Gemma 4 26B A4B",
    ),
    "qwen3.5-35b-a3b": ModelSpec(
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen3.5-35B-A3B",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download multimodal evaluation datasets and models into Hugging Face default cache."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[],
        choices=["all", *sorted(DATASETS.keys())],
        help="Benchmarks to download. Empty means do not download datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        choices=["all", *sorted(MODELS.keys())],
        help="Models to download. Empty means do not download models.",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        help="HF_HOME for default cache. Defaults to $HF_HOME or ~/.cache/huggingface.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Resume interrupted downloads.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to $HF_TOKEN.",
    )
    return parser.parse_args()


def select_items(requested: List[str], registry: Dict[str, object]) -> List[str]:
    if not requested:
        return []
    if "all" in requested:
        return list(registry.keys())
    return requested


def download_datasets(
    names: List[str],
    cache_dir: str,
    token: str | None,
    force_download: bool,
    resume_download: bool,
) -> None:
    if not names:
        return

    for name in names:
        spec = DATASETS[name]
        print(f"[dataset:{name}] {spec.description}")
        print(f"  repo: {spec.repo_id}")
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            resume_download=resume_download,
        )


def download_models(
    names: List[str],
    cache_dir: str,
    token: str | None,
    force_download: bool,
    resume_download: bool,
) -> None:
    if not names:
        return

    for name in names:
        spec = MODELS[name]
        print(f"[model:{name}] {spec.description}")
        print(f"  repo: {spec.repo_id}")
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="model",
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            resume_download=resume_download,
        )


def main() -> None:
    args = parse_args()
    args.hf_home = os.path.abspath(os.path.expanduser(args.hf_home))
    os.environ["HF_HOME"] = args.hf_home
    cache_dir = os.path.join(args.hf_home, "hub")

    print(f"HF_HOME={args.hf_home}")
    print(f"Default cache root={cache_dir}")

    selected_benchmarks = select_items(args.benchmarks, DATASETS)
    selected_models = select_items(args.models, MODELS)

    if not selected_benchmarks and not selected_models:
        raise SystemExit(
            "Nothing selected. Use --benchmarks ... and/or --models ... ."
        )

    download_datasets(
        names=selected_benchmarks,
        cache_dir=cache_dir,
        token=args.token,
        force_download=args.force_download,
        resume_download=args.resume_download,
    )

    download_models(
        names=selected_models,
        cache_dir=cache_dir,
        token=args.token,
        force_download=args.force_download,
        resume_download=args.resume_download,
    )


if __name__ == "__main__":
    main()
