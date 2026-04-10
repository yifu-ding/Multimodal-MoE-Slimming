#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    local_dir_name: str
    description: str


DATASETS: Dict[str, DatasetSpec] = {
    "textvqa": DatasetSpec("lmms-lab/textvqa", "textvqa", "TextVQA (val)"),
    "chartqa": DatasetSpec("lmms-lab/ChartQA", "ChartQA", "ChartQA"),
    "mmstar": DatasetSpec("Lin-Chen/MMStar", "MMStar", "MMStar"),
    "mmbench": DatasetSpec("lmms-lab/MMBench", "MMBench", "MMBench (dev, EN)"),
    "mmvet": DatasetSpec("lmms-lab/MMVet", "MMVet", "MMVet"),
    "mme": DatasetSpec("lmms-lab/MME", "MME", "MME"),
    "realworldqa": DatasetSpec("lmms-lab/RealWorldQA", "RealWorldQA", "RealWorldQA"),
    "coco2017-cap": DatasetSpec(
        "lmms-lab/COCO-Caption2017", "COCO-Caption2017", "COCO2017-Cap (val)"
    ),
    "mvbench": DatasetSpec("OpenGVLab/MVBench", "MVBench", "MVBench"),
    "egoschema": DatasetSpec("lmms-lab/egoschema", "egoschema", "EgoSchema"),
    "videomme": DatasetSpec("lmms-lab/Video-MME", "Video-MME", "VideoMME"),
    "longvideobench": DatasetSpec(
        "longvideobench/LongVideoBench", "LongVideoBench", "LongVideoBench (val)"
    ),
    "video-mmmu": DatasetSpec("lmms-lab/VideoMMMU", "VideoMMMU", "Video-MMMU"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download multimodal evaluation datasets into $HF_HOME/datasets."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        choices=["all", *sorted(DATASETS.keys())],
        help="Benchmarks to download.",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        help="Target HF_HOME. Defaults to $HF_HOME or ~/.cache/huggingface.",
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


def select_benchmarks(requested: List[str]) -> List[str]:
    if "all" in requested:
        return list(DATASETS.keys())
    return requested


def main() -> None:
    args = parse_args()
    hf_datasets_root = os.path.join(args.hf_home, "datasets")
    os.makedirs(hf_datasets_root, exist_ok=True)

    print(f"HF_HOME={args.hf_home}")
    print(f"Download root={hf_datasets_root}")

    for name in select_benchmarks(args.benchmarks):
        spec = DATASETS[name]
        local_dir = os.path.join(hf_datasets_root, spec.local_dir_name)
        print(f"[{name}] {spec.description}")
        print(f"  repo: {spec.repo_id}")
        print(f"  dst : {local_dir}")
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=args.token,
            force_download=args.force_download,
            resume_download=args.resume_download,
        )


if __name__ == "__main__":
    main()
