import os
from pathlib import Path


def get_hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def get_hf_datasets_root() -> str:
    return os.path.join(get_hf_home(), "datasets")


def resolve_dataset_dir(dataset_name: str, *parts: str) -> str:
    candidates = [
        os.path.join(get_hf_datasets_root(), dataset_name, *parts),
        os.path.join("storage", "datasets", dataset_name, *parts),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def require_dataset_dir(dataset_name: str, *parts: str) -> str:
    path = resolve_dataset_dir(dataset_name, *parts)
    if os.path.exists(path):
        return path
    missing = Path(path)
    raise FileNotFoundError(
        f"Dataset path does not exist: {missing}. "
        f"Expected under $HF_HOME/datasets/{dataset_name}."
    )
