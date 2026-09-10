"""Build a frozen mixed-source calibration manifest with shared t-SNE + FPS."""

import argparse
import collections
import datetime as dt
import json
import os
import random
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from observations.common import load_model_bundle, resolve_model_name_or_path
from src.calibration.representation_distill.common import (
    build_compression_token_masks,
    extract_block_output,
    move_inputs_to_model_device,
    prepare_raw_batch_inputs,
)
from src.calibration.representation_distill.runtime.dump_original_data import (
    ManifestRawDataset,
    SUPPORTED_DATASETS,
    load_dataset_rows,
)


SCHEMA_VERSION = 1


def _identity_collate(batch):
    return batch


def _allocate_candidate_counts(
    total_count: int,
    capacities: Mapping[str, int],
    dataset_order: Sequence[str],
) -> Dict[str, int]:
    """Allocate a nearly balanced pool and redistribute source shortfalls."""
    if total_count <= 0:
        raise ValueError(f"candidate_pool_size must be positive, got {total_count}")
    if total_count > sum(int(capacities[name]) for name in dataset_order):
        raise ValueError(
            f"Requested {total_count} candidates, but selected datasets contain only "
            f"{sum(int(capacities[name]) for name in dataset_order)} available rows."
        )

    allocation = {name: 0 for name in dataset_order}
    remaining = total_count
    while remaining > 0:
        progressed = False
        for name in dataset_order:
            if allocation[name] >= int(capacities[name]):
                continue
            allocation[name] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Candidate allocation exhausted all source capacities.")
    return allocation


def _sample_id_from_row(dataset_name: str, row: Mapping[str, Any], dataset_index: int) -> str:
    if dataset_name == "gqa":
        value = row.get("question_id")
    elif dataset_name == "coco":
        value = row.get("id")
    elif dataset_name == "m4_instruct":
        value = row.get("sample_id")
    elif dataset_name == "video_mmmu":
        value = row.get("id")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return str(value if value is not None else f"{dataset_name}_{dataset_index}")


def _build_candidate_references(
    *,
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    datasets: Sequence[str],
    candidate_pool_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    capacities = {name: len(rows_by_dataset[name]) for name in datasets}
    allocation = _allocate_candidate_counts(candidate_pool_size, capacities, datasets)
    references: List[Dict[str, Any]] = []
    for source_offset, dataset_name in enumerate(datasets):
        rng = random.Random(seed + source_offset)
        indices = sorted(rng.sample(range(capacities[dataset_name]), allocation[dataset_name]))
        for dataset_index in indices:
            references.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_index": int(dataset_index),
                    "sample_id": _sample_id_from_row(
                        dataset_name, rows_by_dataset[dataset_name][dataset_index], dataset_index
                    ),
                }
            )
    random.Random(seed + 10_000).shuffle(references)
    return references


def _attention_mask(inputs) -> torch.Tensor:
    value = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if value is None:
        value = getattr(inputs, "attention_mask", None)
    if value is None:
        raise KeyError("Prepared model inputs do not contain attention_mask.")
    return value


def _audit_token_lengths(
    dataset: ManifestRawDataset,
    references: Sequence[Dict[str, Any]],
    bundle,
    min_sample_tokens: int,
) -> List[int]:
    qualified: List[int] = []
    progress = tqdm(range(len(dataset)), desc="Auditing candidate lengths")
    for candidate_idx in progress:
        inputs = prepare_raw_batch_inputs(bundle, [dataset[candidate_idx]])
        token_count = int(_attention_mask(inputs)[0].sum().item())
        references[candidate_idx]["model_token_count"] = token_count
        if token_count >= min_sample_tokens:
            references[candidate_idx]["token_qualified"] = True
            qualified.append(candidate_idx)
        else:
            references[candidate_idx]["token_qualified"] = False
            references[candidate_idx]["exclusion_reason"] = "too_short"
        progress.set_postfix(qualified=len(qualified), refresh=False)
    return qualified


def _extract_centroids(
    *,
    dataset: ManifestRawDataset,
    references: Sequence[Dict[str, Any]],
    qualified_indices: Sequence[int],
    bundle,
    feature_layer: int,
    batch_size: int,
    min_modality_tokens: int,
) -> tuple[List[int], np.ndarray, np.ndarray]:
    qualified_dataset = torch.utils.data.Subset(dataset, list(qualified_indices))
    loader = DataLoader(
        qualified_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_identity_collate,
    )
    kept_indices: List[int] = []
    text_features: List[np.ndarray] = []
    visual_features: List[np.ndarray] = []
    offset = 0

    for raw_batch in tqdm(loader, desc="Extracting calibration centroids"):
        batch_candidate_indices = list(qualified_indices[offset : offset + len(raw_batch)])
        offset += len(raw_batch)
        inputs = prepare_raw_batch_inputs(bundle, raw_batch)
        inputs = move_inputs_to_model_device(bundle.model, inputs)
        masks = build_compression_token_masks(bundle, inputs)
        extraction = extract_block_output(bundle, inputs, layer_idx=feature_layer)
        hidden = extraction.hidden_states.detach().float()
        attention_mask = _attention_mask(inputs).to(hidden.device).bool()

        if hidden.shape[:2] != attention_mask.shape:
            raise ValueError(
                f"Hidden sequence shape {tuple(hidden.shape[:2])} does not match attention mask "
                f"shape {tuple(attention_mask.shape)}."
            )

        for batch_idx, candidate_idx in enumerate(batch_candidate_indices):
            text_mask = masks["text"][batch_idx].to(hidden.device) & attention_mask[batch_idx]
            visual_mask = masks["visual"][batch_idx].to(hidden.device) & attention_mask[batch_idx]
            text_count = int(text_mask.sum().item())
            visual_count = int(visual_mask.sum().item())
            references[candidate_idx]["text_token_count"] = text_count
            references[candidate_idx]["visual_token_count"] = visual_count
            if text_count < min_modality_tokens or visual_count < min_modality_tokens:
                references[candidate_idx]["feature_qualified"] = False
                references[candidate_idx]["exclusion_reason"] = "insufficient_modality_tokens"
                continue

            text_centroid = hidden[batch_idx][text_mask].mean(dim=0).cpu().numpy()
            visual_centroid = hidden[batch_idx][visual_mask].mean(dim=0).cpu().numpy()
            references[candidate_idx]["feature_qualified"] = True
            kept_indices.append(candidate_idx)
            text_features.append(text_centroid)
            visual_features.append(visual_centroid)

        del inputs, extraction, hidden, attention_mask, masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not kept_indices:
        raise RuntimeError("No candidate contains enough text and visual tokens for t-SNE.")
    return kept_indices, np.stack(text_features), np.stack(visual_features)


def _fit_shared_tsne(
    text_features: np.ndarray,
    visual_features: np.ndarray,
    *,
    pca_dim: int,
    perplexity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    features = np.concatenate([text_features, visual_features], axis=0).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = np.clip(features.std(axis=0, keepdims=True), 1e-6, None)
    standardized = (features - mean) / std
    effective_dim = min(int(pca_dim), standardized.shape[0] - 1, standardized.shape[1])
    if effective_dim < 1:
        raise ValueError("At least two paired samples are required for PCA + t-SNE.")
    pca = PCA(n_components=effective_dim, random_state=seed)
    reduced = pca.fit_transform(standardized)
    effective_perplexity = min(float(perplexity), max(1.0, (reduced.shape[0] - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)
    sample_count = text_features.shape[0]
    return (
        embedding[:sample_count],
        embedding[sample_count:],
        {
            "pca_dim": int(effective_dim),
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "perplexity": float(effective_perplexity),
        },
    )


def farthest_point_sample(
    points: np.ndarray,
    count: int,
) -> tuple[List[int], List[float]]:
    """FPS seeded by the sample nearest the global centroid."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape [N, 2], got {points.shape}")
    if count <= 0 or count > points.shape[0]:
        raise ValueError(f"count must be in [1, {points.shape[0]}], got {count}")

    global_centroid = points.mean(axis=0, keepdims=True)
    start_idx = int(np.argmin(np.linalg.norm(points - global_centroid, axis=1)))
    selected = [start_idx]
    selection_distances = [0.0]
    min_distances = np.linalg.norm(points - points[start_idx], axis=1)
    min_distances[start_idx] = -1.0
    while len(selected) < count:
        next_idx = int(np.argmax(min_distances))
        selection_distances.append(float(min_distances[next_idx]))
        selected.append(next_idx)
        next_distances = np.linalg.norm(points - points[next_idx], axis=1)
        min_distances = np.minimum(min_distances, next_distances)
        min_distances[selected] = -1.0
    return selected, selection_distances


def _write_json_atomic(path: str, payload: Mapping[str, Any]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(SUPPORTED_DATASETS),
        default=list(SUPPORTED_DATASETS),
    )
    parser.add_argument("--candidate_pool_size", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--score_tokens_per_sample", type=int, default=2048)
    parser.add_argument("--min_sample_tokens", type=int, default=None)
    parser.add_argument("--feature_layer", type=int, default=0)
    parser.add_argument("--feature_batch_size", type=int, default=2)
    parser.add_argument("--min_modality_tokens", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--video_max_long_side", type=int, default=480)
    parser.add_argument("--device_map", default=None)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--allow_missing_sources",
        action="store_true",
        help="Allow global FPS to return no sample from one of the requested sources.",
    )
    return parser


def run(args) -> Dict[str, Any]:
    datasets = list(dict.fromkeys(args.datasets))
    min_sample_tokens = (
        args.score_tokens_per_sample
        if args.min_sample_tokens is None
        else int(args.min_sample_tokens)
    )
    if min_sample_tokens < args.score_tokens_per_sample:
        raise ValueError(
            "min_sample_tokens must be >= score_tokens_per_sample so every selected sample "
            "can contribute the same number of score tokens."
        )
    if args.num_samples > args.candidate_pool_size:
        raise ValueError("num_samples cannot exceed candidate_pool_size.")

    rows_by_dataset = {
        name: load_dataset_rows(name, minimum_rows=args.candidate_pool_size)
        for name in datasets
    }
    references = _build_candidate_references(
        rows_by_dataset=rows_by_dataset,
        datasets=datasets,
        candidate_pool_size=args.candidate_pool_size,
        seed=args.seed,
    )
    raw_dataset = ManifestRawDataset(
        references,
        num_video_frames=args.num_video_frames,
        video_max_long_side=args.video_max_long_side,
        rows_by_dataset=rows_by_dataset,
    )

    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    bundle = load_model_bundle(
        args.model_name_or_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        max_decoder_layer=args.feature_layer,
    )

    qualified_indices = _audit_token_lengths(
        raw_dataset, references, bundle, min_sample_tokens
    )
    if len(qualified_indices) < args.num_samples:
        raise RuntimeError(
            f"Only {len(qualified_indices)} candidates have at least {min_sample_tokens} tokens; "
            f"cannot select {args.num_samples}. Increase candidate_pool_size or lower the threshold."
        )
    kept_indices, text_features, visual_features = _extract_centroids(
        dataset=raw_dataset,
        references=references,
        qualified_indices=qualified_indices,
        bundle=bundle,
        feature_layer=args.feature_layer,
        batch_size=args.feature_batch_size,
        min_modality_tokens=args.min_modality_tokens,
    )
    if len(kept_indices) < args.num_samples:
        raise RuntimeError(
            f"Only {len(kept_indices)} candidates have paired text/visual centroids; "
            f"cannot select {args.num_samples}."
        )

    text_tsne, visual_tsne, reduction_metadata = _fit_shared_tsne(
        text_features,
        visual_features,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        seed=args.seed,
    )
    sample_centers = 0.5 * (text_tsne + visual_tsne)
    for local_idx, candidate_idx in enumerate(kept_indices):
        references[candidate_idx]["text_tsne"] = [
            float(value) for value in text_tsne[local_idx]
        ]
        references[candidate_idx]["visual_tsne"] = [
            float(value) for value in visual_tsne[local_idx]
        ]
        references[candidate_idx]["sample_center"] = [
            float(value) for value in sample_centers[local_idx]
        ]
    selected_local_indices, selection_distances = farthest_point_sample(
        sample_centers, args.num_samples
    )

    selected_samples: List[Dict[str, Any]] = []
    for rank, (local_idx, min_distance) in enumerate(
        zip(selected_local_indices, selection_distances)
    ):
        candidate_idx = kept_indices[local_idx]
        reference = references[candidate_idx]
        selected_samples.append(
            {
                "selection_rank": rank,
                "dataset_name": reference["dataset_name"],
                "dataset_index": reference["dataset_index"],
                "sample_id": reference["sample_id"],
                "model_token_count": reference["model_token_count"],
                "score_token_count": args.score_tokens_per_sample,
                "text_token_count": reference["text_token_count"],
                "visual_token_count": reference["visual_token_count"],
                "text_tsne": [float(value) for value in text_tsne[local_idx]],
                "visual_tsne": [float(value) for value in visual_tsne[local_idx]],
                "sample_center": [float(value) for value in sample_centers[local_idx]],
                "fps_min_distance": float(min_distance),
            }
        )

    selected_counts = collections.Counter(item["dataset_name"] for item in selected_samples)
    missing_sources = [name for name in datasets if selected_counts[name] == 0]
    if missing_sources and not args.allow_missing_sources:
        raise RuntimeError(
            f"Global FPS selected no samples from {missing_sources}. This is not a valid "
            "multi-source calibration set; change the candidate pool/seed or explicitly pass "
            "--allow_missing_sources."
        )

    source_summary = {}
    for dataset_name in datasets:
        source_summary[dataset_name] = {
            "candidate_count": sum(
                item["dataset_name"] == dataset_name for item in references
            ),
            "token_qualified_count": sum(
                item["dataset_name"] == dataset_name and item.get("token_qualified", False)
                for item in references
            ),
            "feature_qualified_count": sum(
                item["dataset_name"] == dataset_name and item.get("feature_qualified", False)
                for item in references
            ),
            "selected_count": int(selected_counts[dataset_name]),
        }

    candidate_records = []
    for candidate_idx, reference in enumerate(references):
        record = dict(reference)
        record["candidate_index"] = candidate_idx
        candidate_records.append(record)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": "shared_standardized_pca_tsne_global_fps",
        "model_name_or_path": args.model_name_or_path,
        "resolved_model_name_or_path": resolve_model_name_or_path(args.model_name_or_path),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "datasets": datasets,
        "candidate_pool_size": len(references),
        "num_samples": len(selected_samples),
        "min_sample_tokens": min_sample_tokens,
        "score_tokens_per_sample": args.score_tokens_per_sample,
        "score_token_sampling": "per_sample_proportional_modality_uniform_positions",
        "feature_token_scope": "full_valid_sequence",
        "feature_layer": args.feature_layer,
        "feature_batch_size": args.feature_batch_size,
        "min_modality_tokens": args.min_modality_tokens,
        "seed": args.seed,
        "num_video_frames": args.num_video_frames,
        "video_max_long_side": args.video_max_long_side,
        "reduction": reduction_metadata,
        "source_summary": source_summary,
        "candidates": candidate_records,
        "samples": selected_samples,
    }
    _write_json_atomic(args.output_manifest, payload)
    print(f"[mixed-calibration] Saved {len(selected_samples)} samples: {args.output_manifest}")
    print(f"[mixed-calibration] Selected source counts: {dict(selected_counts)}")
    return payload


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
