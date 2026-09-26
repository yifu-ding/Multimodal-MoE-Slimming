import os
from pathlib import Path
from glob import glob


def get_hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.join(".cache", "huggingface"))


def get_hf_datasets_root() -> str:
    return os.path.join(get_hf_home(), "datasets")


def resolve_dataset_dir(dataset_name: str, *parts: str) -> str:
    candidates = [
        os.path.join(get_hf_datasets_root(), dataset_name, *parts),
        os.path.join("storage", "datasets", dataset_name, *parts),
    ]
    # Fallback: datasets downloaded via HF hub snapshots, e.g.
    # $HF_HOME/hub/datasets--lmms-lab--COCO-Caption2017/snapshots/<rev>/data
    hf_home = get_hf_home()
    hub_root = os.path.join(hf_home, "hub")
    repo_suffix = dataset_name.replace("/", "--")
    snapshot_glob = os.path.join(
        hub_root, f"datasets--*--{repo_suffix}", "snapshots", "*", *parts
    )
    snapshot_candidates = sorted(glob(snapshot_glob))
    candidates.extend(snapshot_candidates[::-1])  # prefer latest lexicographically
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
