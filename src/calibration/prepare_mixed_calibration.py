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
from decord._ffi.base import DECORDError
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


SCHEMA_VERSION = 2


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
    video_decode_errors = 0
    progress = tqdm(range(len(dataset)), desc="Auditing candidate lengths")
    for candidate_idx in progress:
        reference = references[candidate_idx]
        try:
            sample = dataset[candidate_idx]
        except (DECORDError, OSError, EOFError) as error:
            if reference["dataset_name"] != "video_mmmu":
                raise
            video_decode_errors += 1
            reference["token_qualified"] = False
            reference["exclusion_reason"] = "video_decode_error"
            reference["exclusion_error_type"] = type(error).__name__
            reference["exclusion_error"] = str(error)[:1000]
            progress.write(
                "[mixed-calibration] Skipping undecodable VideoMMMU candidate "
                f"{reference.get('sample_id', candidate_idx)!r}: "
                f"{type(error).__name__}: {str(error)[:240]}"
            )
            progress.set_postfix(
                qualified=len(qualified),
                video_decode_errors=video_decode_errors,
                refresh=False,
            )
            continue

        inputs = prepare_raw_batch_inputs(bundle, [sample])
        token_count = int(_attention_mask(inputs)[0].sum().item())
        reference["model_token_count"] = token_count
        if token_count >= min_sample_tokens:
            reference["token_qualified"] = True
            qualified.append(candidate_idx)
        else:
            reference["token_qualified"] = False
            reference["exclusion_reason"] = "too_short"
        progress.set_postfix(
            qualified=len(qualified),
            video_decode_errors=video_decode_errors,
            refresh=False,
        )
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


def balanced_farthest_point_sample(
    points: np.ndarray,
    sources: Sequence[str],
    source_quotas: Mapping[str, int],
) -> tuple[List[int], List[float]]:
    """Run global FPS while respecting an exact per-source sample quota."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape [N, 2], got {points.shape}")
    if len(sources) != points.shape[0]:
        raise ValueError("sources length must match points.")
    source_counts = collections.Counter(str(source) for source in sources)
    for source, quota in source_quotas.items():
        if int(quota) < 0 or int(quota) > source_counts[source]:
            raise ValueError(
                f"Invalid quota {quota} for source {source!r} with "
                f"{source_counts[source]} candidates."
            )
    total_count = sum(int(value) for value in source_quotas.values())
    if total_count <= 0:
        raise ValueError("At least one sample must be selected.")

    global_centroid = points.mean(axis=0, keepdims=True)
    centroid_distances = np.linalg.norm(points - global_centroid, axis=1)
    remaining = {str(source): int(quota) for source, quota in source_quotas.items()}
    selected: List[int] = []
    selection_distances: List[float] = []
    min_distances = np.full(points.shape[0], np.inf, dtype=np.float64)

    while len(selected) < total_count:
        allowed = np.asarray(
            [remaining.get(str(source), 0) > 0 for source in sources], dtype=bool
        )
        if selected:
            allowed[np.asarray(selected, dtype=np.int64)] = False
            candidate_scores = np.where(allowed, min_distances, -1.0)
            next_idx = int(np.argmax(candidate_scores))
            next_distance = float(candidate_scores[next_idx])
        else:
            candidate_scores = np.where(allowed, centroid_distances, np.inf)
            next_idx = int(np.argmin(candidate_scores))
            next_distance = 0.0
        if not allowed[next_idx]:
            raise RuntimeError("Balanced FPS exhausted eligible source candidates.")
        selected.append(next_idx)
        selection_distances.append(next_distance)
        remaining[str(sources[next_idx])] -= 1
        distances = np.linalg.norm(points - points[next_idx], axis=1)
        min_distances = np.minimum(min_distances, distances)
    return selected, selection_distances


def ensure_token_budget_capacity(
    selected_indices: Sequence[int],
    *,
    candidate_indices: Sequence[int],
    candidate_sources: Sequence[str],
    candidate_capacities: Sequence[int],
    target_tokens: int,
) -> tuple[List[int], int]:
    """Replace short selections within each source until capacity reaches target."""
    selected = [int(index) for index in selected_indices]
    capacities = np.asarray(candidate_capacities, dtype=np.int64)
    if capacities.shape != (len(candidate_indices),):
        raise ValueError("candidate_capacities shape does not match candidates.")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")

    selected_set = set(selected)
    current_capacity = int(capacities[selected].sum())
    replacements = 0
    swaps = []
    source_names = sorted({str(source) for source in candidate_sources})
    for source in source_names:
        source_selected = sorted(
            (
                (int(capacities[index]), position, index)
                for position, index in enumerate(selected)
                if str(candidate_sources[index]) == source
            ),
            key=lambda item: (item[0], item[2]),
        )
        source_unselected = sorted(
            (
                (int(capacities[index]), index)
                for index, candidate_source in enumerate(candidate_sources)
                if index not in selected_set and str(candidate_source) == source
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for old, new in zip(source_selected, source_unselected):
            gain = new[0] - old[0]
            if gain > 0:
                swaps.append((gain, old[1], old[2], new[1]))

    for gain, selected_position, old_idx, new_idx in sorted(
        swaps, key=lambda item: (-item[0], item[1], item[3])
    ):
        if current_capacity >= target_tokens:
            break
        selected[selected_position] = new_idx
        selected_set.remove(old_idx)
        selected_set.add(new_idx)
        current_capacity += gain
        replacements += 1

    if current_capacity < target_tokens:
        raise RuntimeError(
            f"Balanced selection can provide only {current_capacity} score tokens, "
            f"below requested total budget {target_tokens}. Increase candidate_pool_size, "
            "reduce the total budget, or reduce num_samples per constrained source."
        )
    return selected, replacements


def allocate_total_score_tokens(
    capacities: Sequence[int], total_tokens: int
) -> List[int]:
    """Deterministically water-fill per-sample quotas to an exact total."""
    capacity = [int(value) for value in capacities]
    if not capacity or min(capacity) <= 0:
        raise ValueError("All selected samples must have positive token capacity.")
    if total_tokens < len(capacity):
        raise ValueError("total_tokens must permit at least one token per sample.")
    if sum(capacity) < total_tokens:
        raise ValueError(
            f"Selected capacity {sum(capacity)} is below total token budget {total_tokens}."
        )

    allocation = [0] * len(capacity)
    remaining = int(total_tokens)
    active = list(range(len(capacity)))
    while remaining > 0:
        if not active:
            raise RuntimeError("Token allocation exhausted capacity before reaching budget.")
        share = max(remaining // len(active), 1)
        next_active = []
        for sample_idx in active:
            available = capacity[sample_idx] - allocation[sample_idx]
            give = min(available, share, remaining)
            allocation[sample_idx] += give
            remaining -= give
            if allocation[sample_idx] < capacity[sample_idx]:
                next_active.append(sample_idx)
            if remaining == 0:
                break
        active = next_active
    return allocation


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
    parser.add_argument(
        "--score_token_budget",
        type=int,
        default=None,
        help="Exact total score-token budget. Enables variable per-sample quotas with no per-sample cap.",
    )
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
    variable_score_quota = args.score_token_budget is not None
    min_sample_tokens = (
        (1 if variable_score_quota else args.score_tokens_per_sample)
        if args.min_sample_tokens is None
        else int(args.min_sample_tokens)
    )
    if not variable_score_quota and min_sample_tokens < args.score_tokens_per_sample:
        raise ValueError(
            "min_sample_tokens must be >= score_tokens_per_sample so every selected sample "
            "can contribute the same number of score tokens."
        )
    if variable_score_quota and int(args.score_token_budget) < args.num_samples:
        raise ValueError("score_token_budget must provide at least one token per sample.")
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
    kept_sources = [references[index]["dataset_name"] for index in kept_indices]
    feature_capacities = collections.Counter(kept_sources)
    selected_source_quotas = _allocate_candidate_counts(
        args.num_samples, feature_capacities, datasets
    )
    selected_local_indices, selection_distances = balanced_farthest_point_sample(
        sample_centers, kept_sources, selected_source_quotas
    )

    token_budget_replacements = 0
    if variable_score_quota:
        candidate_capacities = [
            int(references[index]["model_token_count"]) for index in kept_indices
        ]
        selected_local_indices, token_budget_replacements = ensure_token_budget_capacity(
            selected_local_indices,
            candidate_indices=kept_indices,
            candidate_sources=kept_sources,
            candidate_capacities=candidate_capacities,
            target_tokens=int(args.score_token_budget),
        )
        selection_distances = []
        for rank, local_idx in enumerate(selected_local_indices):
            if rank == 0:
                selection_distances.append(0.0)
            else:
                prior = sample_centers[np.asarray(selected_local_indices[:rank])]
                selection_distances.append(
                    float(np.linalg.norm(prior - sample_centers[local_idx], axis=1).min())
                )

    selected_capacities = [
        int(references[kept_indices[local_idx]]["model_token_count"])
        for local_idx in selected_local_indices
    ]
    if variable_score_quota:
        selected_score_counts = allocate_total_score_tokens(
            selected_capacities, int(args.score_token_budget)
        )
    else:
        selected_score_counts = [int(args.score_tokens_per_sample)] * args.num_samples

    selected_samples: List[Dict[str, Any]] = []
    for rank, (local_idx, min_distance, score_token_count) in enumerate(
        zip(selected_local_indices, selection_distances, selected_score_counts)
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
                "score_token_count": int(score_token_count),
                "text_token_count": reference["text_token_count"],
                "visual_token_count": reference["visual_token_count"],
                "text_tsne": [float(value) for value in text_tsne[local_idx]],
                "visual_tsne": [float(value) for value in visual_tsne[local_idx]],
                "sample_center": [float(value) for value in sample_centers[local_idx]],
                "fps_min_distance": float(min_distance),
            }
        )

    selected_counts = collections.Counter(item["dataset_name"] for item in selected_samples)
    selected_score_tokens = collections.Counter()
    for item in selected_samples:
        selected_score_tokens[item["dataset_name"]] += int(item["score_token_count"])
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
            "selected_score_token_count": int(selected_score_tokens[dataset_name]),
        }

    candidate_records = []
    for candidate_idx, reference in enumerate(references):
        record = dict(reference)
        record["candidate_index"] = candidate_idx
        candidate_records.append(record)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": "shared_standardized_pca_tsne_balanced_fps",
        "model_name_or_path": args.model_name_or_path,
        "resolved_model_name_or_path": resolve_model_name_or_path(args.model_name_or_path),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "datasets": datasets,
        "candidate_pool_size": len(references),
        "num_samples": len(selected_samples),
        "min_sample_tokens": min_sample_tokens,
        "score_tokens_per_sample": None
        if variable_score_quota
        else args.score_tokens_per_sample,
        "score_token_budget": int(sum(selected_score_counts)),
        "score_token_counts_variable": bool(variable_score_quota),
        "token_budget_replacements": int(token_budget_replacements),
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
    quotas = np.asarray(selected_score_counts, dtype=np.int64)
    print(
        "[mixed-calibration] Score-token quotas: "
        f"total={int(quotas.sum())}, min={int(quotas.min())}, "
        f"median={float(np.median(quotas)):.1f}, max={int(quotas.max())}; "
        f"by_source={dict(selected_score_tokens)}"
    )
    return payload


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
