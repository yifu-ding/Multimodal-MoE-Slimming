import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from observations.common import compute_channel_response


def sample_expert_pairs(
    raw_stats: Dict[str, Any],
    num_pairs: int,
    seed: int,
    include_pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    layers = list(raw_stats["layers"])
    rng = random.Random(seed)
    selected: List[Tuple[int, int]] = []

    for layer, expert in include_pairs:
        if layer not in raw_stats["layer_to_num_experts"]:
            continue
        n_exp = int(raw_stats["layer_to_num_experts"][layer])
        if 0 <= expert < n_exp and (layer, expert) not in selected:
            selected.append((layer, expert))
        if len(selected) >= num_pairs:
            return selected[:num_pairs]

    all_pairs: List[Tuple[int, int]] = []
    for layer in layers:
        n_exp = int(raw_stats["layer_to_num_experts"][layer])
        for expert in range(n_exp):
            all_pairs.append((layer, expert))
    rng.shuffle(all_pairs)

    for pair in all_pairs:
        if pair in selected:
            continue
        selected.append(pair)
        if len(selected) >= num_pairs:
            break
    return selected[:num_pairs]


def build_response_levels_payload(
    raw_stats: Dict[str, Any],
    num_channel_pairs: int,
    seed: int,
    include_pairs: List[Tuple[int, int]],
) -> Dict[str, Any]:
    channel_response = compute_channel_response(raw_stats)
    layers = list(raw_stats["layers"])
    selected_pairs = sample_expert_pairs(
        raw_stats=raw_stats,
        num_pairs=num_channel_pairs,
        seed=seed,
        include_pairs=include_pairs,
    )

    channel_level = []
    for layer, expert in selected_pairs:
        channel_level.append(
            {
                "layer": int(layer),
                "expert": int(expert),
                "text": channel_response[layer]["text"][expert].to(torch.float32).cpu(),
                "visual": channel_response[layer]["visual"][expert].to(torch.float32).cpu(),
                "n_text": int(raw_stats["channel_count"][layer]["text"][expert].item()),
                "n_visual": int(raw_stats["channel_count"][layer]["visual"][expert].item()),
            }
        )

    expert_text = []
    expert_visual = []
    for layer in layers:
        expert_text.append(channel_response[layer]["text"].mean(dim=-1).to(torch.float32).cpu())
        expert_visual.append(channel_response[layer]["visual"].mean(dim=-1).to(torch.float32).cpu())
    expert_level = {
        "layers": [int(x) for x in layers],
        "text": torch.stack(expert_text, dim=0),
        "visual": torch.stack(expert_visual, dim=0),
    }

    layer_level = {
        "layers": expert_level["layers"],
        "text": expert_level["text"].mean(dim=-1).to(torch.float32).cpu(),
        "visual": expert_level["visual"].mean(dim=-1).to(torch.float32).cpu(),
    }

    return {
        "meta": {
            "source_raw_stats": raw_stats.get(
                "resolved_model_name_or_path", raw_stats.get("model_name_or_path", "")
            ),
            "dataset": raw_stats.get("dataset", ""),
            "num_samples": int(raw_stats.get("num_samples", 0)),
            "seed": int(seed),
            "num_channel_pairs": int(num_channel_pairs),
        },
        "channel_level": channel_level,
        "expert_level": expert_level,
        "layer_level": layer_level,
    }


def plot_response_levels(payload: Dict[str, Any], output_dir: str, prefix: str = "") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: Dict[str, Any] = {
        "channel_level": [],
        "expert_level": None,
        "layer_level": None,
    }
    file_prefix = f"{prefix}_" if prefix else ""

    for item in payload.get("channel_level", []):
        layer = int(item["layer"])
        expert = int(item["expert"])
        text = item["text"].detach().cpu().numpy()
        visual = item["visual"].detach().cpu().numpy()
        fig, axis = plt.subplots(1, 1, figsize=(14, 3))
        axis.plot(text, label="text", alpha=0.9, color="steelblue")
        axis.plot(visual, label="visual", alpha=0.9, color="darkorange")
        axis.set_title(
            f"Layer {layer} Expert {expert}  "
            f"(n_text={int(item['n_text'])}, n_visual={int(item['n_visual'])})"
        )
        axis.set_xlabel("Channel Index")
        axis.set_ylabel("Mean |activation|")
        axis.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, f"{file_prefix}channel_resp_L{layer}_E{expert}.png")
        plt.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths["channel_level"].append(path)

    expert_level = payload.get("expert_level")
    if expert_level is not None:
        expert_text = expert_level["text"].detach().cpu()
        expert_visual = expert_level["visual"].detach().cpu()
        expert_delta = expert_visual - expert_text
        layers = expert_level["layers"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        matrices = [
            ("Text response", expert_text, "Blues"),
            ("Visual response", expert_visual, "Oranges"),
            ("Visual - Text", expert_delta, "coolwarm"),
        ]
        for axis, (title, matrix, cmap) in zip(axes, matrices):
            image = axis.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
            axis.set_title(title)
            axis.set_xlabel("Expert Index")
            axis.set_ylabel("Layer Index")
            axis.set_yticks(range(len(layers)))
            axis.set_yticklabels(layers)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        path = os.path.join(output_dir, f"{file_prefix}expert_level_overview.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths["expert_level"] = path

    layer_level = payload.get("layer_level")
    if layer_level is not None:
        layers = layer_level["layers"]
        text = layer_level["text"].detach().cpu().numpy()
        visual = layer_level["visual"].detach().cpu().numpy()
        fig, axis = plt.subplots(1, 1, figsize=(12, 4))
        axis.plot(layers, text, label="text", color="steelblue", marker="o")
        axis.plot(layers, visual, label="visual", color="darkorange", marker="o")
        axis.set_title("Layer-level mean response")
        axis.set_xlabel("Layer Index")
        axis.set_ylabel("Mean |activation|")
        axis.legend()
        axis.grid(alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{file_prefix}layer_level_overview.png")
        plt.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths["layer_level"] = path

    return saved_paths
