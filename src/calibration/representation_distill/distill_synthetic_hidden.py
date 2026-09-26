"""从教师模型 hidden 缓存中优化出一小份合成 hidden, 用于表征级校准.

训练目标: 用 MMD, 协方差, 多样性等分布损失, 辅以均值与方差矩匹配, 使合成样本在统计上逼近教师缓存.
"""
import argparse
import bisect
import os
import sys

# 将仓库根与父目录加入 sys.path, 以便以包形式导入 src.*
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
for _p in (REPO_PARENT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from src.calibration.representation_distill.common import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    ensure_dir,
    resolve_hidden_start_layer,
    seed_everything,
    sort_sequence_by_position_ids,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Helpers (行采样与下采样等工具函数)
# ---------------------------------------------------------------------------

def _sample_rows(tensor: torch.Tensor, count: int) -> torch.Tensor:
    # 从首维随机抽 count 行, 若不足则全取
    if count >= tensor.shape[0]:
        return tensor
    indices = torch.randint(0, tensor.shape[0], (count,), device=tensor.device)
    return tensor.index_select(0, indices)


def _log_stage(message: str) -> None:
    print(f"[representation_distill] {message}", flush=True)


def _subsample_flat(tensor: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Subsample rows for efficient kernel computation."""
    # 对行做随机下采样, 控制核矩阵规模, 降低 MMD 等计算开销
    if tensor.shape[0] <= max_tokens:
        return tensor
    indices = torch.randperm(tensor.shape[0], device=tensor.device)[:max_tokens]
    return tensor[indices]


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _torch_load_cpu(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_teacher_cache_payload(cache_path: str) -> dict:
    return _torch_load_cpu(cache_path)


class _DenseTeacherCacheStore:
    def __init__(self, payload: dict):
        self.payload = payload
        self.teacher_cache = payload["teacher_cache"]
        self.next_block_cache = payload.get("next_block_cache", None)
        self.teacher_labels = payload.get("modality_labels", None)
        self.teacher_position_ids = payload.get("position_ids", None)
        if self.teacher_position_ids is not None:
            self.teacher_position_ids = self.teacher_position_ids.to(dtype=torch.long)
            self.teacher_cache, self.teacher_position_ids, self.teacher_labels = sort_sequence_by_position_ids(
                self.teacher_cache,
                self.teacher_position_ids,
                self.teacher_labels,
            )
        else:
            self.teacher_position_ids = torch.arange(
                self.teacher_cache.shape[1], dtype=torch.long
            ).unsqueeze(0).expand(self.teacher_cache.shape[0], -1)

    @property
    def num_samples(self) -> int:
        return int(self.teacher_cache.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.teacher_cache.shape)

    def fetch_indices(
        self,
        indices: torch.Tensor,
        *,
        target_device: str | torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        indices = indices.to(dtype=torch.long, device="cpu")
        hidden = self.teacher_cache.index_select(0, indices).to(device=target_device, dtype=target_dtype)
        labels = (
            self.teacher_labels.index_select(0, indices).to(target_device)
            if self.teacher_labels is not None
            else None
        )
        position_ids = self.teacher_position_ids.index_select(0, indices).to(target_device)
        return hidden, labels, position_ids

    def sample_random(
        self,
        count: int,
        *,
        target_device: str | torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.num_samples, (count,), device="cpu")
        hidden, labels, position_ids = self.fetch_indices(
            indices,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        return hidden, labels, position_ids, indices

    def fetch_next_block_indices(
        self,
        indices: torch.Tensor,
        *,
        target_device: str | torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.next_block_cache is None:
            raise KeyError("Teacher cache payload does not contain `next_block_cache`.")
        indices = indices.to(dtype=torch.long, device="cpu")
        return self.next_block_cache.index_select(0, indices).to(device=target_device, dtype=target_dtype)


class _ShardedTeacherCacheStore:
    def __init__(self, cache_path: str, payload: dict, *, pool_size: int):
        self.payload = payload
        self.cache_root = os.path.dirname(cache_path)
        self.shards = payload["shards"]
        self.num_samples = int(payload.get("metadata", {}).get("total_samples", sum(int(s["num_samples"]) for s in self.shards)))
        self.pool_size = max(1, min(int(pool_size), self.num_samples))
        self._shard_offsets = []
        offset = 0
        for shard in self.shards:
            self._shard_offsets.append(offset)
            offset += int(shard["num_samples"])
        self._shard_cache: dict[int, dict] = {}
        self._buffer_hidden: torch.Tensor | None = None
        self._buffer_labels: torch.Tensor | None = None
        self._buffer_position_ids: torch.Tensor | None = None
        self._buffer_global_indices: torch.Tensor | None = None
        self._shard_order = torch.randperm(len(self.shards)).tolist() if self.shards else []
        self._shard_cursor = 0
        self._sample_calls = 0
        self._refresh_every_calls = 16

        first = self._load_shard(0)
        self.shape = (
            self.num_samples,
            int(first["teacher_cache"].shape[1]),
            int(first["teacher_cache"].shape[2]),
        )

    def _load_shard(self, shard_idx: int) -> dict:
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            return cached
        shard_path = os.path.join(self.cache_root, self.shards[shard_idx]["path"])
        shard_payload = _torch_load_cpu(shard_path)
        if "position_ids" not in shard_payload:
            shard_payload["position_ids"] = torch.arange(
                shard_payload["teacher_cache"].shape[1], dtype=torch.long
            ).unsqueeze(0).expand(shard_payload["teacher_cache"].shape[0], -1)
        else:
            shard_payload["position_ids"] = shard_payload["position_ids"].to(dtype=torch.long)
        self._shard_cache = {shard_idx: shard_payload}
        return shard_payload

    def _append_to_buffer(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        position_ids: torch.Tensor,
        global_indices: torch.Tensor,
    ) -> None:
        if self._buffer_hidden is None:
            self._buffer_hidden = hidden
            self._buffer_labels = labels
            self._buffer_position_ids = position_ids
            self._buffer_global_indices = global_indices
            return
        self._buffer_hidden = torch.cat([self._buffer_hidden, hidden], dim=0)
        if self._buffer_labels is not None and labels is not None:
            self._buffer_labels = torch.cat([self._buffer_labels, labels], dim=0)
        elif labels is None:
            self._buffer_labels = None
        self._buffer_position_ids = torch.cat([self._buffer_position_ids, position_ids], dim=0)
        self._buffer_global_indices = torch.cat([self._buffer_global_indices, global_indices], dim=0)

    def _buffer_size(self) -> int:
        return 0 if self._buffer_hidden is None else int(self._buffer_hidden.shape[0])

    def _trim_buffer(self, max_samples: int) -> None:
        if self._buffer_hidden is None:
            return
        if self._buffer_hidden.shape[0] <= max_samples:
            return
        keep_start = self._buffer_hidden.shape[0] - max_samples
        keep = torch.arange(keep_start, self._buffer_hidden.shape[0], dtype=torch.long)
        self._buffer_hidden = self._buffer_hidden.index_select(0, keep)
        if self._buffer_labels is not None:
            self._buffer_labels = self._buffer_labels.index_select(0, keep)
        self._buffer_position_ids = self._buffer_position_ids.index_select(0, keep)
        self._buffer_global_indices = self._buffer_global_indices.index_select(0, keep)

    def _fill_buffer(self, min_samples: int) -> None:
        current = 0 if self._buffer_hidden is None else int(self._buffer_hidden.shape[0])
        while current < min_samples:
            if self._shard_cursor >= len(self._shard_order):
                self._shard_order = torch.randperm(len(self.shards)).tolist()
                self._shard_cursor = 0
            shard_idx = self._shard_order[self._shard_cursor]
            self._shard_cursor += 1
            shard_payload = self._load_shard(shard_idx)
            offset = self._shard_offsets[shard_idx]
            global_indices = offset + torch.arange(shard_payload["teacher_cache"].shape[0], dtype=torch.long)
            self._append_to_buffer(
                shard_payload["teacher_cache"],
                shard_payload.get("modality_labels", None),
                shard_payload["position_ids"],
                global_indices,
            )
            current = int(self._buffer_hidden.shape[0])

    def _refresh_buffer(self, add_samples: int) -> None:
        if add_samples <= 0:
            return
        old_target = self.pool_size
        self._fill_buffer(min(old_target + add_samples, self.num_samples))
        self._trim_buffer(old_target)

    def sample_random(
        self,
        count: int,
        *,
        target_device: str | torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        self._fill_buffer(min(self.pool_size, self.num_samples))
        assert self._buffer_hidden is not None and self._buffer_position_ids is not None and self._buffer_global_indices is not None
        pool_count = self._buffer_hidden.shape[0]
        if count <= pool_count:
            take = torch.randperm(pool_count, device="cpu")[:count]
        else:
            take = torch.randint(0, pool_count, (count,), device="cpu")
        hidden = self._buffer_hidden.index_select(0, take).to(device=target_device, dtype=target_dtype)
        labels = (
            self._buffer_labels.index_select(0, take).to(target_device)
            if self._buffer_labels is not None
            else None
        )
        position_ids = self._buffer_position_ids.index_select(0, take).to(target_device)
        global_indices = self._buffer_global_indices.index_select(0, take)
        self._sample_calls += 1
        shard_samples = int(self.shards[0]["num_samples"]) if self.shards else 0
        if (
            shard_samples > 0
            and self._buffer_size() < self.num_samples
            and self._sample_calls % self._refresh_every_calls == 0
        ):
            self._refresh_buffer(shard_samples)
        return hidden, labels, position_ids, global_indices

    def fetch_indices(
        self,
        indices: torch.Tensor,
        *,
        target_device: str | torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        indices = indices.to(dtype=torch.long, device="cpu")
        order = torch.argsort(indices)
        sorted_indices = indices.index_select(0, order)
        hidden_parts = []
        label_parts = []
        position_parts = []
        start = 0
        while start < sorted_indices.numel():
            global_idx = int(sorted_indices[start].item())
            shard_idx = bisect.bisect_right(self._shard_offsets, global_idx) - 1
            shard_start = self._shard_offsets[shard_idx]
            shard_end = shard_start + int(self.shards[shard_idx]["num_samples"])
            end = start
            while end < sorted_indices.numel() and int(sorted_indices[end].item()) < shard_end:
                end += 1
            shard_payload = self._load_shard(shard_idx)
            local_indices = (sorted_indices[start:end] - shard_start).to(dtype=torch.long)
            hidden_parts.append(shard_payload["teacher_cache"].index_select(0, local_indices))
            if "modality_labels" in shard_payload:
                label_parts.append(shard_payload["modality_labels"].index_select(0, local_indices))
            position_parts.append(shard_payload["position_ids"].index_select(0, local_indices))
            start = end

        hidden = torch.cat(hidden_parts, dim=0)
        labels = torch.cat(label_parts, dim=0) if label_parts else None
        position_ids = torch.cat(position_parts, dim=0)
        inverse = torch.argsort(order)
        hidden = hidden.index_select(0, inverse).to(device=target_device, dtype=target_dtype)
        labels = labels.index_select(0, inverse).to(target_device) if labels is not None else None
        position_ids = position_ids.index_select(0, inverse).to(target_device)
        return hidden, labels, position_ids

    def fetch_next_block_indices(
        self,
        indices: torch.Tensor,
        *,
        target_device: str | torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        indices = indices.to(dtype=torch.long, device="cpu")
        order = torch.argsort(indices)
        sorted_indices = indices.index_select(0, order)
        parts = []
        start = 0
        while start < sorted_indices.numel():
            global_idx = int(sorted_indices[start].item())
            shard_idx = bisect.bisect_right(self._shard_offsets, global_idx) - 1
            shard_start = self._shard_offsets[shard_idx]
            shard_end = shard_start + int(self.shards[shard_idx]["num_samples"])
            end = start
            while end < sorted_indices.numel() and int(sorted_indices[end].item()) < shard_end:
                end += 1
            shard_payload = self._load_shard(shard_idx)
            if "next_block_cache" not in shard_payload:
                raise KeyError("Teacher cache shard does not contain `next_block_cache`.")
            local_indices = (sorted_indices[start:end] - shard_start).to(dtype=torch.long)
            parts.append(shard_payload["next_block_cache"].index_select(0, local_indices))
            start = end
        out = torch.cat(parts, dim=0)
        inverse = torch.argsort(order)
        return out.index_select(0, inverse).to(device=target_device, dtype=target_dtype)


def _build_teacher_cache_store(cache_path: str, payload: dict, *, pool_size: int):
    if "teacher_cache" in payload:
        return _DenseTeacherCacheStore(payload)
    if "shards" in payload:
        return _ShardedTeacherCacheStore(cache_path, payload, pool_size=pool_size)
    raise KeyError(
        f"Teacher cache payload at {cache_path} contains neither `teacher_cache` nor `shards`."
    )


def _build_synthetic_banks(
    init_hidden: torch.Tensor,
    init_labels: torch.Tensor | None,
    init_std: float,
) -> tuple[torch.nn.ParameterDict, torch.Tensor, torch.Tensor]:
    flat_hidden = init_hidden.reshape(-1, init_hidden.shape[-1])
    if init_labels is None:
        flat_labels = torch.full(
            (flat_hidden.shape[0],),
            MODALITY_TEXT,
            dtype=torch.int8,
            device=flat_hidden.device,
        )
    else:
        flat_labels = init_labels.reshape(-1).to(torch.int8)

    bank_params: dict[str, torch.nn.Parameter] = {}
    flat_bank_indices = torch.empty(flat_labels.shape[0], dtype=torch.long, device=flat_labels.device)
    for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
        mod_mask = flat_labels == mod_id
        if int(mod_mask.sum()) == 0:
            continue
        bank_value = flat_hidden[mod_mask].clone()
        if init_std > 0:
            bank_value.add_(torch.randn_like(bank_value) * init_std)
        bank_params[str(int(mod_id))] = torch.nn.Parameter(bank_value)
        flat_bank_indices[mod_mask] = torch.arange(
            int(mod_mask.sum()),
            device=flat_labels.device,
            dtype=torch.long,
        )
    return torch.nn.ParameterDict(bank_params), flat_labels, flat_bank_indices


def _assemble_synthetic_hidden(
    bank_params: torch.nn.ParameterDict,
    template_labels: torch.Tensor,
    template_bank_indices: torch.Tensor,
    synthetic_size: int,
    compressed_length: int,
    hidden_size: int,
) -> torch.Tensor:
    flat_labels = template_labels.reshape(-1)
    flat_bank_indices = template_bank_indices.reshape(-1)
    flat_hidden = torch.empty(
        flat_labels.shape[0],
        hidden_size,
        device=flat_labels.device,
        dtype=next(iter(bank_params.values())).dtype,
    )
    for mod_id_str, bank in bank_params.items():
        mod_mask = flat_labels == int(mod_id_str)
        flat_hidden[mod_mask] = bank.index_select(0, flat_bank_indices[mod_mask])
    return flat_hidden.view(synthetic_size, compressed_length, hidden_size)


def _sample_synthetic_batch_indices(
    synthetic_size: int,
    synthetic_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if synthetic_batch_size <= 0 or synthetic_batch_size >= synthetic_size:
        return torch.arange(synthetic_size, device=device, dtype=torch.long)
    return torch.randperm(synthetic_size, device=device)[:synthetic_batch_size]


def _compute_cached_next_block_rel_l2(
    *,
    cached_teacher_target: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    synthetic_attention_mask: torch.Tensor,
) -> torch.Tensor:
    diff = (synthetic_hidden.float() - cached_teacher_target.float()).pow(2).sum(dim=-1).sqrt()
    ref = cached_teacher_target.float().pow(2).sum(dim=-1).sqrt().clamp_min(1e-6)
    token_loss = diff / ref
    masked = token_loss * synthetic_attention_mask.float()
    return masked.sum() / synthetic_attention_mask.float().sum().clamp_min(1.0)


def _sample_synthetic_batch_indices(
    synthetic_size: int,
    synthetic_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if synthetic_batch_size <= 0 or synthetic_batch_size >= synthetic_size:
        return torch.arange(synthetic_size, device=device, dtype=torch.long)
    return torch.randperm(synthetic_size, device=device)[:synthetic_batch_size]


# ---------------------------------------------------------------------------
# Loss functions (MMD, 协方差, 多样性, 矩匹配)
# ---------------------------------------------------------------------------

def _mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidth_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> torch.Tensor:
    """Unbiased MMD^2 estimate with multi-bandwidth RBF kernel.

    Parameters
    ----------
    X, Y : (N, D) and (M, D) token-level feature matrices.
    bandwidth_multipliers : scales applied to the median pairwise distance.
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.tensor(0.0, device=X.device)
    X = X.float()
    Y = Y.float()

    # 成对欧氏距离平方, 用于 RBF 核
    XX = torch.cdist(X, X).pow(2)
    YY = torch.cdist(Y, Y).pow(2)
    XY = torch.cdist(X, Y).pow(2)

    # 带宽: 用全体距离的中位数启发式, 再乘多组 multiplier
    with torch.no_grad():
        all_dists = torch.cat([XX.view(-1), YY.view(-1), XY.view(-1)])
        median_dist = all_dists.median().clamp(min=1e-6)

    # 多带宽 RBF 下无偏 MMD^2 估计, 各带宽结果相加
    mmd = torch.tensor(0.0, device=X.device)
    for mult in bandwidth_multipliers:
        bw_sq = 2.0 * (mult * median_dist)
        k_XX = (-XX / bw_sq).exp().mean()
        k_YY = (-YY / bw_sq).exp().mean()
        k_XY = (-XY / bw_sq).exp().mean()
        mmd = mmd + k_XX + k_YY - 2.0 * k_XY
    return mmd


def _cov_loss(teacher_flat: torch.Tensor, synth_flat: torch.Tensor) -> torch.Tensor:
    """MSE between cross-channel covariance matrices."""
    teacher_flat = teacher_flat.float()
    synth_flat = synth_flat.float()
    # 通道维协方差矩阵的逐元素 MSE, 对齐二阶结构
    t_centered = teacher_flat - teacher_flat.mean(dim=0, keepdim=True)
    s_centered = synth_flat - synth_flat.mean(dim=0, keepdim=True)
    t_cov = (t_centered.T @ t_centered) / max(teacher_flat.shape[0] - 1, 1)
    s_cov = (s_centered.T @ s_centered) / max(synth_flat.shape[0] - 1, 1)
    return (t_cov - s_cov).pow(2).mean()


def _mean_var_loss(
    teacher_flat: torch.Tensor,
    synth_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel 1st/2nd moment MSE, returned as (mean_loss, var_loss)."""
    t_mean = teacher_flat.mean(dim=0)
    s_mean = synth_flat.mean(dim=0)
    t_var = teacher_flat.var(dim=0, unbiased=False)
    s_var = synth_flat.var(dim=0, unbiased=False)
    return (t_mean - s_mean).pow(2).mean(), (t_var - s_var).pow(2).mean()


def _diversity_loss(synthetic_flat: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity (lower = more diverse)."""
    if synthetic_flat.shape[0] <= 1:
        return torch.tensor(0.0, device=synthetic_flat.device)
    normed = F.normalize(synthetic_flat, dim=-1)
    sim = normed @ normed.T
    n = sim.shape[0]
    # 非对角线余弦相似度均值, 越小表示样本间越分散
    off_diag = sim.masked_select(~torch.eye(n, dtype=torch.bool, device=sim.device))
    return off_diag.mean()


def _cosine_offdiag_stats(flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Diagnostics for sample diversity based on pairwise cosine similarities."""
    if flat.shape[0] <= 1:
        zero = torch.tensor(0.0, device=flat.device)
        return {
            "cosine_mean": zero,
            "cosine_p50": zero,
            "cosine_p90": zero,
            "cosine_max": zero,
        }

    flat = flat.float()
    normed = F.normalize(flat, dim=-1)
    sim = normed @ normed.T
    off_diag = sim.masked_select(~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device))
    return {
        "cosine_mean": off_diag.mean(),
        "cosine_p50": off_diag.quantile(0.5),
        "cosine_p90": off_diag.quantile(0.9),
        "cosine_max": off_diag.max(),
    }


def _compute_diagnostics(
    *,
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    teacher_labels: torch.Tensor | None,
    synth_labels: torch.Tensor | None,
    mmd_subsample: int,
) -> dict[str, torch.Tensor]:
    """Interpretable diagnostics for plotting and failure analysis."""
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    teacher_norm = teacher_flat.norm(dim=-1)
    synth_norm = synth_flat.norm(dim=-1)
    cosine_stats = _cosine_offdiag_stats(_subsample_flat(synth_flat, mmd_subsample))
    mean_gap = (teacher_flat.mean(dim=0) - synth_flat.mean(dim=0)).norm()

    diagnostics = {
        "diag/token_norm_mean_teacher": teacher_norm.mean().detach(),
        "diag/token_norm_mean_synth": synth_norm.mean().detach(),
        "diag/token_norm_std_teacher": teacher_norm.std(unbiased=False).detach(),
        "diag/token_norm_std_synth": synth_norm.std(unbiased=False).detach(),
        "diag/token_norm_mean_gap": (teacher_norm.mean() - synth_norm.mean()).abs().detach(),
        "diag/centroid_l2": mean_gap.detach(),
        "diag/div_cosine_mean": cosine_stats["cosine_mean"].detach(),
        "diag/div_cosine_p50": cosine_stats["cosine_p50"].detach(),
        "diag/div_cosine_p90": cosine_stats["cosine_p90"].detach(),
        "diag/div_cosine_max": cosine_stats["cosine_max"].detach(),
    }

    teacher_labels_flat = teacher_labels.reshape(-1) if teacher_labels is not None else None
    synth_labels_flat = synth_labels.reshape(-1) if synth_labels is not None else None
    for mod_id, t_group, s_group in _iter_modality_groups(
        teacher_flat, synth_flat, teacher_labels_flat, synth_labels_flat
    ):
        if mod_id is None:
            continue
        diagnostics[f"diag/centroid_l2/mod{int(mod_id)}"] = (
            t_group.mean(dim=0) - s_group.mean(dim=0)
        ).norm().detach()
        diagnostics[f"diag/token_norm_mean_gap/mod{int(mod_id)}"] = (
            t_group.norm(dim=-1).mean() - s_group.norm(dim=-1).mean()
        ).abs().detach()
    return diagnostics


def _compute_teacher_baseline_losses(
    *,
    teacher_store,
    teacher_batch_size: int,
    mmd_subsample: int,
    target_device: str | torch.device,
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Teacher-vs-teacher baseline used to contextualize non-zero distribution losses."""
    if teacher_store.num_samples == 0:
        return {}
    ref_batch_size = min(teacher_batch_size, teacher_store.num_samples)
    ref_batch_a, ref_labels_a, _, _ = teacher_store.sample_random(
        ref_batch_size,
        target_device=target_device,
        target_dtype=target_dtype,
    )
    ref_batch_b, ref_labels_b, _, _ = teacher_store.sample_random(
        ref_batch_size,
        target_device=target_device,
        target_dtype=target_dtype,
    )
    ref_losses = _compute_losses(
        ref_batch_a,
        ref_batch_b,
        ref_labels_a,
        ref_labels_b,
        mmd_subsample=mmd_subsample,
    )
    return {
        "baseline/mmd_teacher_teacher": ref_losses["mmd"].detach(),
        "baseline/cov_teacher_teacher": ref_losses["cov"].detach(),
        "baseline/mean_teacher_teacher": ref_losses["mean"].detach(),
        "baseline/var_teacher_teacher": ref_losses["var"].detach(),
    }


def _iter_modality_groups(
    teacher_flat: torch.Tensor,
    synth_flat: torch.Tensor,
    teacher_labels_flat: torch.Tensor | None,
    synth_labels_flat: torch.Tensor | None,
    min_tokens: int = 2,
):
    """Yield (modality_id, teacher_sub, synth_sub) token groups.

    When labels are missing, yields a single ``(None, teacher_flat, synth_flat)`` pair
    so that the caller can fall back to a global (pooled) computation.
    """
    if teacher_labels_flat is None or synth_labels_flat is None:
        yield None, teacher_flat, synth_flat
        return

    for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
        t_mask = teacher_labels_flat == mod_id
        s_mask = synth_labels_flat == mod_id
        if int(t_mask.sum()) < min_tokens or int(s_mask.sum()) < min_tokens:
            continue
        yield mod_id, teacher_flat[t_mask], synth_flat[s_mask]


def _compute_losses(
    teacher_batch: torch.Tensor,
    synthetic_hidden: torch.Tensor,
    teacher_labels: torch.Tensor | None,
    synth_labels: torch.Tensor | None,
    mmd_subsample: int,
) -> dict[str, torch.Tensor]:
    """All distillation losses, computed per-modality and averaged across modalities.

    Groups covered (MMD / cov / mean / var): text, image, video — whichever are
    present in *both* teacher and synthetic batches. When no modality labels are
    available the computation gracefully degrades to a single pooled pair.
    ``div`` stays pooled on the synthetic tokens (regardless of modality) because
    diversity should be measured on the whole synthetic set.
    """
    teacher_flat = teacher_batch.reshape(-1, teacher_batch.shape[-1])
    synth_flat = synthetic_hidden.reshape(-1, synthetic_hidden.shape[-1])

    teacher_labels_flat = teacher_labels.reshape(-1) if teacher_labels is not None else None
    synth_labels_flat = synth_labels.reshape(-1) if synth_labels is not None else None

    device = teacher_flat.device
    zero = torch.tensor(0.0, device=device)

    mmd_sum = zero.clone()
    cov_sum = zero.clone()
    mean_sum = zero.clone()
    var_sum = zero.clone()
    n_groups = 0
    per_group: dict[str, torch.Tensor] = {}

    for mod_id, t_group, s_group in _iter_modality_groups(
        teacher_flat, synth_flat, teacher_labels_flat, synth_labels_flat
    ):
        t_sub = _subsample_flat(t_group, mmd_subsample)
        s_sub = _subsample_flat(s_group, mmd_subsample)

        mmd_g = _mmd_rbf(t_sub, s_sub)
        cov_g = _cov_loss(t_sub, s_sub)
        mean_g, var_g = _mean_var_loss(t_group, s_group)

        mmd_sum = mmd_sum + mmd_g
        cov_sum = cov_sum + cov_g
        mean_sum = mean_sum + mean_g
        var_sum = var_sum + var_g
        n_groups += 1

        if mod_id is not None:
            per_group[f"mmd/mod{int(mod_id)}"] = mmd_g.detach()
            per_group[f"cov/mod{int(mod_id)}"] = cov_g.detach()
            per_group[f"mean/mod{int(mod_id)}"] = mean_g.detach()
            per_group[f"var/mod{int(mod_id)}"] = var_g.detach()

    if n_groups == 0:
        # Nothing matched (e.g. degenerate batch); fall back to pooled.
        t_sub = _subsample_flat(teacher_flat, mmd_subsample)
        s_sub = _subsample_flat(synth_flat, mmd_subsample)
        mmd_sum = _mmd_rbf(t_sub, s_sub)
        cov_sum = _cov_loss(t_sub, s_sub)
        mean_sum, var_sum = _mean_var_loss(teacher_flat, synth_flat)
        n_groups = 1

    div_loss = _diversity_loss(_subsample_flat(synth_flat, mmd_subsample))

    losses: dict[str, torch.Tensor] = {
        "mmd": mmd_sum / n_groups,
        "cov": cov_sum / n_groups,
        "mean": mean_sum / n_groups,
        "var": var_sum / n_groups,
        "div": div_loss,
    }
    losses.update(per_group)
    return losses


def _compute_weighted_total_loss(
    *,
    args,
    losses: dict[str, torch.Tensor],
    div_scale: float,
    ema_state: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base_device = losses["mmd"].device
    raw_terms = {
        "mmd": losses["mmd"],
        "cov": losses["cov"],
        "mean": losses["mean"],
        "var": losses["var"],
        "div": losses["div"],
    }
    if "block_rel_l2" in losses:
        raw_terms["block_rel_l2"] = losses["block_rel_l2"]

    weights = {
        "mmd": torch.tensor(args.lambda_mmd, device=base_device),
        "cov": torch.tensor(args.lambda_cov, device=base_device),
        "mean": torch.tensor(args.lambda_mean, device=base_device),
        "var": torch.tensor(args.lambda_var, device=base_device),
        "div": torch.tensor(div_scale * args.lambda_div, device=base_device),
    }
    if "block_rel_l2" in raw_terms:
        weights["block_rel_l2"] = torch.tensor(args.lambda_block, device=base_device)

    total_loss = torch.tensor(0.0, device=base_device)
    weighted_terms: dict[str, torch.Tensor] = {}
    normalized_terms: dict[str, torch.Tensor] = {}
    for name, raw in raw_terms.items():
        term = raw
        if args.use_ema_normalized_losses:
            if ema_state is None:
                raise ValueError("EMA state is required when EMA-normalized losses are enabled.")
            ema_value = ema_state.get(name)
            if ema_value is None:
                ema_value = raw.detach().float()
            else:
                ema_value = args.loss_ema_decay * ema_value + (1.0 - args.loss_ema_decay) * raw.detach().float()
            ema_state[name] = ema_value
            term = raw / ema_value.clamp_min(1e-8).to(device=raw.device, dtype=raw.dtype)
            normalized_terms[name] = term.detach()
        weighted = weights[name] * term
        weighted_terms[name] = weighted.detach()
        total_loss = total_loss + weighted
    return total_loss, weighted_terms, normalized_terms


def _move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj


def _build_training_state(
    *,
    step: int,
    optimizer: torch.optim.Optimizer | None,
    loss_ema_state: dict[str, torch.Tensor] | None,
    teacher_anchor_indices: torch.Tensor,
) -> dict:
    state = {
        "global_step": int(step),
        "teacher_anchor_indices": teacher_anchor_indices.detach().cpu().long(),
    }
    if optimizer is not None:
        state["optimizer_state"] = _move_to_cpu(optimizer.state_dict())
    if loss_ema_state is not None:
        state["loss_ema_state"] = {
            key: value.detach().cpu().float()
            for key, value in loss_ema_state.items()
        }
    return state


def _infer_resume_step(resume_payload: dict) -> int:
    training_state = resume_payload.get("training_state")
    if isinstance(training_state, dict) and "global_step" in training_state:
        return int(training_state["global_step"])

    metadata = resume_payload.get("metadata", {})
    history = metadata.get("history")
    if isinstance(history, list) and history:
        last_entry = history[-1]
        if isinstance(last_entry, dict) and "step" in last_entry:
            return int(last_entry["step"]) + 1
    return 0


def _build_output_payload(
    *,
    synthetic_hidden_cpu: torch.Tensor,
    synth_position_ids_cpu: torch.Tensor,
    synth_labels_cpu: torch.Tensor | None,
    teacher_meta: dict,
    args,
    final_losses: dict[str, float] | None,
    history: list[dict],
    block_constraint_layer: int | None,
    ablation_notes: dict[str, str],
    training_state: dict | None,
) -> dict:
    attention_mask = torch.ones(
        synthetic_hidden_cpu.shape[0],
        synthetic_hidden_cpu.shape[1],
        dtype=torch.long,
    )
    payload = {
        "synthetic_hidden": synthetic_hidden_cpu,
        "attention_mask": attention_mask,
        "position_ids": synth_position_ids_cpu,
            "metadata": {
                "method": "multimodal_representation_level_calibration_distillation",
                "teacher_cache_path": args.teacher_cache_path,
                "teacher_metadata": teacher_meta,
            "synthetic_size": args.synthetic_size,
            "synthetic_batch_size": args.synthetic_batch_size,
            "compressed_length": int(synthetic_hidden_cpu.shape[1]),
            "hidden_size": int(synthetic_hidden_cpu.shape[2]),
            "dtype": "float32",
            "position_ids_strategy": "frozen_from_init_teacher_samples",
            "train_steps": args.train_steps,
            "teacher_batch_size": args.teacher_batch_size,
            "teacher_pool_size": args.teacher_pool_size,
            "lr": args.lr,
            "resume_from": args.resume_from,
            "wandb_every_n_steps": args.wandb_every_n_steps,
            "loss_weights": {
                "mmd": args.lambda_mmd,
                "cov": args.lambda_cov,
                "div": args.lambda_div,
                "mean": args.lambda_mean,
                "var": args.lambda_var,
                "block_rel_l2": args.lambda_block,
            },
                "modality_grouped_losses": ["mmd", "cov", "mean", "var"],
                "div_warmup_steps": args.div_warmup_steps,
                "mmd_subsample": args.mmd_subsample,
                "use_ema_normalized_losses": args.use_ema_normalized_losses,
            "loss_ema_decay": args.loss_ema_decay,
            "reset_optimizer_on_resume": args.reset_optimizer_on_resume,
            "diversity_ablation": args.diversity_ablation,
            "distribution_ablation": args.distribution_ablation,
            "ablation_notes": ablation_notes,
                "synthetic_bank_mode": "independent_per_modality",
                "block_constraint_layer": block_constraint_layer,
                "final_losses": final_losses,
                "history": history,
            "created_at": utc_now_iso(),
        },
    }
    if training_state is not None:
        payload["training_state"] = training_state
    if synth_labels_cpu is not None:
        payload["modality_labels"] = synth_labels_cpu
    return payload


def _save_payload(
    *,
    output_path: str,
    synthetic_hidden: torch.Tensor,
    synth_position_ids: torch.Tensor,
    synth_labels: torch.Tensor | None,
    teacher_meta: dict,
    args,
    final_losses: dict[str, float] | None,
    history: list[dict],
    block_constraint_layer: int | None,
    ablation_notes: dict[str, str],
    training_state: dict | None,
) -> dict:
    synthetic_hidden_cpu = synthetic_hidden.detach().cpu().float()
    synth_position_ids_cpu = synth_position_ids.detach().cpu().long()
    synth_labels_cpu = synth_labels.cpu() if synth_labels is not None else None
    payload = _build_output_payload(
        synthetic_hidden_cpu=synthetic_hidden_cpu,
        synth_position_ids_cpu=synth_position_ids_cpu,
        synth_labels_cpu=synth_labels_cpu,
        teacher_meta=teacher_meta,
        args=args,
        final_losses=final_losses,
        history=history,
        block_constraint_layer=block_constraint_layer,
        ablation_notes=ablation_notes,
        training_state=training_state,
    )
    torch.save(payload, output_path)
    return payload


def _prune_saved_pt_paths(saved_paths: list[str], *, max_keep: int) -> list[str]:
    if max_keep <= 0:
        max_keep = 1
    retained: list[str] = []
    for path in saved_paths:
        if path not in retained:
            retained.append(path)
    while len(retained) > max_keep:
        stale_path = retained.pop(0)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"[representation_distill] Removed old checkpoint: {stale_path}", flush=True)
    return retained


def _assemble_full_synthetic_hidden(
    *,
    bank_params,
    synth_template_labels: torch.Tensor,
    synth_template_bank_indices: torch.Tensor,
    synthetic_size: int,
    compressed_length: int,
    hidden_size: int,
) -> torch.Tensor:
    return _assemble_synthetic_hidden(
        bank_params=bank_params,
        template_labels=synth_template_labels,
        template_bank_indices=synth_template_bank_indices,
        synthetic_size=synthetic_size,
        compressed_length=compressed_length,
        hidden_size=hidden_size,
    )


def _apply_ablation_presets(args) -> dict[str, str]:
    """Apply named ablations by rewriting effective loss weights in-place."""
    notes: dict[str, str] = {}

    if args.diversity_ablation == "no_div":
        args.lambda_div = 0.0
        args.div_warmup_steps = 0
        notes["diversity_ablation"] = "Disabled diversity supervision (lambda_div=0, div_warmup_steps=0)."
    else:
        notes["diversity_ablation"] = "Full diversity supervision."

    if args.distribution_ablation == "moment_only":
        args.lambda_mmd = 0.0
        notes["distribution_ablation"] = "Moment-only distribution matching (lambda_mmd=0)."
    elif args.distribution_ablation == "mmd_only":
        args.lambda_cov = 0.0
        args.lambda_mean = 0.0
        args.lambda_var = 0.0
        notes["distribution_ablation"] = (
            "MMD-only distribution matching (lambda_cov=0, lambda_mean=0, lambda_var=0)."
        )
    else:
        notes["distribution_ablation"] = "Full distribution matching."
    return notes


# ---------------------------------------------------------------------------
# CLI (命令行参数)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill a small synthetic hidden calibration set from teacher hidden cache."
    )
    parser.add_argument("--teacher_cache_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a saved distilled hidden checkpoint/payload.")
    parser.add_argument("--synthetic_size", type=int, default=256)
    parser.add_argument("--synthetic_batch_size", type=int, default=0,
                        help="How many synthetic samples to use per optimization step. "
                             "0 means using all synthetic samples.")
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument(
        "--teacher_pool_size",
        type=int,
        default=128,
        help="For sharded teacher caches, keep only this many samples in the in-memory sampling pool.",
    )
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--init_std", type=float, default=0.0)
    # Distribution-aware loss weights
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_cov", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    # 辅助矩匹配: 均值与方差
    parser.add_argument("--lambda_mean", type=float, default=0.5)
    parser.add_argument("--lambda_var", type=float, default=0.5)
    parser.add_argument("--lambda_block", type=float, default=0.0)
    parser.add_argument(
        "--diversity_ablation",
        type=str,
        default="full",
        choices=["full", "no_div"],
        help="Named ablation for diversity supervision.",
    )
    parser.add_argument(
        "--distribution_ablation",
        type=str,
        default="full",
        choices=["full", "moment_only", "mmd_only"],
        help="Named ablation for distribution-matching supervision.",
    )
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    # Diversity warmup: avoids ``div`` dominating the first few hundred steps
    # when the other losses are still on the order of 1e-5.
    parser.add_argument("--div_warmup_steps", type=int, default=200,
                        help="Linearly ramp ``lambda_div`` from 0 to its full value over "
                             "this many steps. Set to 0 to disable.")
    # MMD efficiency
    parser.add_argument("--mmd_subsample", type=int, default=2048,
                        help="Max tokens for kernel matrix computation.")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--wandb_every_n_steps",
        type=int,
        default=1,
        help="Upload W&B metrics every N training steps. Set <=0 to log only the final step.",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--use_ema_normalized_losses", action="store_true")
    parser.add_argument("--loss_ema_decay", type=float, default=0.99)
    parser.add_argument("--checkpoint_interval", type=int, default=0)
    parser.add_argument("--reset_optimizer_on_resume", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ablation_notes = _apply_ablation_presets(args)
    train_dtype = _parse_dtype(args.train_dtype)
    seed_everything(args.seed)
    ensure_dir(os.path.dirname(args.output_path))
    _log_stage(
        "Starting synthetic hidden distillation "
        f"(train_steps={args.train_steps}, synthetic_size={args.synthetic_size}, "
        f"synthetic_batch_size={args.synthetic_batch_size}, teacher_batch_size={args.teacher_batch_size}, "
        f"train_dtype={args.train_dtype})."
    )

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required when --wandb_project is set. "
                "Install it with `pip install wandb` or disable logging with "
                "`--wandb_mode disabled`."
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "teacher_cache_path": args.teacher_cache_path,
                "output_path": args.output_path,
                "resume_from": args.resume_from,
                "synthetic_size": args.synthetic_size,
                "synthetic_batch_size": args.synthetic_batch_size,
                "teacher_batch_size": args.teacher_batch_size,
                "teacher_pool_size": args.teacher_pool_size,
                "train_steps": args.train_steps,
                "lr": args.lr,
                "seed": args.seed,
                "train_dtype": args.train_dtype,
                "init_std": args.init_std,
                "lambda_mmd": args.lambda_mmd,
                "lambda_cov": args.lambda_cov,
                "lambda_div": args.lambda_div,
                "lambda_mean": args.lambda_mean,
                "lambda_var": args.lambda_var,
                "lambda_block": args.lambda_block,
                "diversity_ablation": args.diversity_ablation,
                "distribution_ablation": args.distribution_ablation,
                "use_ema_normalized_losses": args.use_ema_normalized_losses,
                "loss_ema_decay": args.loss_ema_decay,
                "checkpoint_interval": args.checkpoint_interval,
                "div_warmup_steps": args.div_warmup_steps,
                "mmd_subsample": args.mmd_subsample,
                "log_interval": args.log_interval,
                "wandb_every_n_steps": args.wandb_every_n_steps,
                "model_name_or_path": args.model_name_or_path,
                "device": args.device,
                "ablation_notes": ablation_notes,
                "reset_optimizer_on_resume": args.reset_optimizer_on_resume,
            },
        )
        wandb_run.define_metric("step")
        wandb_run.define_metric("loss/*", step_metric="step")
        wandb_run.define_metric("diag/*", step_metric="step")
        wandb_run.define_metric("baseline/*", step_metric="step")
        wandb_run.define_metric("ratio/*", step_metric="step")
        wandb_run.define_metric("schedule/*", step_metric="step")
        wandb_run.define_metric("meta/*", step_metric="step")
        _log_stage(
            f"W&B initialized (project={args.wandb_project}, run={args.wandb_run_name}, mode={args.wandb_mode})."
        )

    _log_stage(f"Loading teacher cache from {args.teacher_cache_path}.")
    cache_payload = _load_teacher_cache_payload(args.teacher_cache_path)
    teacher_store = _build_teacher_cache_store(
        args.teacher_cache_path,
        cache_payload,
        pool_size=args.teacher_pool_size,
    )
    teacher_meta = cache_payload["metadata"]
    teacher_labels = getattr(teacher_store, "teacher_labels", None)
    _log_stage(
        "Loaded teacher cache "
        f"shape={teacher_store.shape}."
    )
    resume_payload = None
    if args.resume_from:
        _log_stage(f"Loading resume payload from {args.resume_from}.")
        resume_payload = torch.load(args.resume_from, map_location="cpu")

    # 2) 设备选择
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if "shards" in cache_payload:
        _log_stage(
            "Using shard lazy sampling for teacher cache; shards will be loaded on demand "
            f"(teacher_pool_size={args.teacher_pool_size})."
        )
    else:
        _log_stage("Keeping teacher cache on CPU; batches will be moved to the training device on demand.")

    resume_training_state = {}
    resume_start_step = 0

    if resume_payload is not None:
        init_hidden = resume_payload["synthetic_hidden"].to(device=device, dtype=train_dtype)
        init_position_ids = resume_payload["position_ids"].to(device=device, dtype=torch.long)
        synth_labels = resume_payload.get("modality_labels", None)
        if synth_labels is not None:
            synth_labels = synth_labels.to(device=device)
        resume_training_state = resume_payload.get("training_state", {}) or {}
        resume_start_step = _infer_resume_step(resume_payload)

        if args.synthetic_size != init_hidden.shape[0]:
            print(
                f"[representation_distill] Overriding synthetic_size from {args.synthetic_size} "
                f"to resumed value {init_hidden.shape[0]}."
            )
            args.synthetic_size = int(init_hidden.shape[0])
        if args.synthetic_batch_size > args.synthetic_size:
            args.synthetic_batch_size = args.synthetic_size
    else:
        # 3) 初始化: 从教师缓存随机抽 synthetic_size 条序列作为可学习参数, 可选加高斯噪声
        init_hidden, synth_labels, init_position_ids, init_indices = teacher_store.sample_random(
            args.synthetic_size,
            target_device=device,
            target_dtype=train_dtype,
        )

    bank_params, synth_template_labels_flat, synth_template_bank_indices_flat = _build_synthetic_banks(
        init_hidden=init_hidden,
        init_labels=synth_labels,
        init_std=0.0 if resume_payload is not None else args.init_std,
    )
    synth_template_labels = (
        synth_labels
        if synth_labels is not None
        else synth_template_labels_flat.view(init_hidden.shape[0], init_hidden.shape[1])
    )
    synth_template_bank_indices = synth_template_bank_indices_flat.view(
        init_hidden.shape[0], init_hidden.shape[1]
    )
    optimizer = torch.optim.Adam(list(bank_params.parameters()), lr=args.lr)
    history = []
    saved_pt_paths: list[str] = []
    final_losses = None
    loss_ema_state: dict[str, torch.Tensor] | None = {} if args.use_ema_normalized_losses else None

    block_constraint_layer = None
    teacher_anchor_indices = resume_training_state.get("teacher_anchor_indices")
    if teacher_anchor_indices is None:
        if resume_payload is None:
            teacher_anchor_indices = init_indices.clone()
        elif args.lambda_block > 0:
            raise ValueError(
                "Resume payload is missing teacher_anchor_indices, so block loss cannot be resumed safely. "
                "Resume from a newer checkpoint or set --lambda_block 0."
            )
        else:
            teacher_anchor_indices = torch.zeros(args.synthetic_size, dtype=torch.long)
    teacher_anchor_indices = teacher_anchor_indices.to(device=device, dtype=torch.long)
    synth_attention_mask = torch.ones(init_hidden.shape[:2], dtype=torch.long, device=device)
    synth_position_ids = init_position_ids.clone()
    effective_synth_batch_size = (
        args.synthetic_size if args.synthetic_batch_size <= 0
        else min(args.synthetic_batch_size, args.synthetic_size)
    )
    if resume_payload is not None:
        if "history" in resume_payload.get("metadata", {}):
            history = list(resume_payload["metadata"]["history"])
        if args.use_ema_normalized_losses:
            resume_ema = resume_training_state.get("loss_ema_state")
            if isinstance(resume_ema, dict):
                loss_ema_state = {
                    key: value.to(device=device, dtype=torch.float32)
                    for key, value in resume_ema.items()
                }
        if (
            not args.reset_optimizer_on_resume
            and isinstance(resume_training_state, dict)
            and "optimizer_state" in resume_training_state
        ):
            optimizer.load_state_dict(resume_training_state["optimizer_state"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        print(
            f"[representation_distill] Resumed from {args.resume_from} at global_step={resume_start_step} "
            f"(optimizer_reset={int(args.reset_optimizer_on_resume)})."
        )

    if args.lambda_block > 0:
        block_constraint_layer = resolve_hidden_start_layer(teacher_meta)
        try:
            teacher_store.fetch_next_block_indices(
                teacher_anchor_indices[:1],
                target_device=device,
                target_dtype=train_dtype,
            )
        except KeyError as exc:
            raise ValueError(
                "Teacher cache does not contain cached `next_block_cache`, so "
                "--lambda_block must be 0 or you need to rebuild the cache with "
                "--cache_next_block_targets."
            ) from exc
        _log_stage(
            f"Using cached next-block targets for block supervision (layer={block_constraint_layer})."
        )

    if wandb_run is not None and teacher_labels is not None and synth_labels is not None:
        teacher_labels_flat = teacher_labels.reshape(-1)
        synth_labels_flat = synth_labels.reshape(-1)
        modality_payload = {}
        for mod_id in (MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO):
            modality_payload[f"modality_share/teacher/mod{int(mod_id)}"] = float(
                (teacher_labels_flat == mod_id).float().mean().cpu().item()
            )
            modality_payload[f"modality_share/synth/mod{int(mod_id)}"] = float(
                (synth_labels_flat == mod_id).float().mean().cpu().item()
            )
        wandb_run.summary.update(modality_payload)

    total_target_steps = resume_start_step + args.train_steps
    _log_stage(
        f"Entering optimization loop at step={resume_start_step}, target_step={total_target_steps}."
    )
    progress = tqdm(
        range(resume_start_step, total_target_steps),
        desc="Distilling compact hidden",
        initial=resume_start_step,
        total=total_target_steps,
        dynamic_ncols=True,
        leave=True,
    )
    for step in progress:
        synth_batch_indices = _sample_synthetic_batch_indices(
            synthetic_size=args.synthetic_size,
            synthetic_batch_size=effective_synth_batch_size,
            device=synth_template_labels.device,
        )
        synth_batch_labels = synth_template_labels.index_select(0, synth_batch_indices)
        synth_batch_bank_indices = synth_template_bank_indices.index_select(0, synth_batch_indices)
        synth_batch_position_ids = synth_position_ids.index_select(0, synth_batch_indices)
        synth_batch_attention_mask = synth_attention_mask.index_select(0, synth_batch_indices)
        synthetic_hidden = _assemble_synthetic_hidden(
            bank_params=bank_params,
            template_labels=synth_batch_labels,
            template_bank_indices=synth_batch_bank_indices,
            synthetic_size=synth_batch_indices.shape[0],
            compressed_length=init_hidden.shape[1],
            hidden_size=init_hidden.shape[2],
        )
        # Sample a teacher batch
        teacher_batch, teacher_batch_labels, _, _ = teacher_store.sample_random(
            args.teacher_batch_size,
            target_device=device,
            target_dtype=train_dtype,
        )

        # Per-modality grouped losses (primary). ``div`` stays pooled.
        losses = _compute_losses(
            teacher_batch, synthetic_hidden,
            teacher_batch_labels, synth_batch_labels,
            mmd_subsample=args.mmd_subsample,
        )

        # Warmup: ramp ``div`` in linearly so it doesn't dominate at step 0 when
        # the other losses are tiny (and would be destabilized by a large cosine
        # regularizer). After ``div_warmup_steps`` the full weight is applied.
        div_scale = 1.0
        if args.div_warmup_steps > 0:
            div_scale = min(1.0, (step + 1) / float(args.div_warmup_steps))

        if args.lambda_block > 0:
            cached_teacher_target = teacher_store.fetch_next_block_indices(
                teacher_anchor_indices.index_select(0, synth_batch_indices),
                target_device=device,
                target_dtype=train_dtype,
            )
            block_rel_l2 = _compute_cached_next_block_rel_l2(
                cached_teacher_target=cached_teacher_target,
                synthetic_hidden=synthetic_hidden,
                synthetic_attention_mask=synth_batch_attention_mask,
            )
            losses["block_rel_l2"] = block_rel_l2

        total_loss, weighted_terms, normalized_terms = _compute_weighted_total_loss(
            args=args,
            losses=losses,
            div_scale=div_scale,
            ema_state=loss_ema_state,
        )

        # 反传并更新合成 hidden
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_losses = {
            "total": float(total_loss.detach().cpu().item()),
            "mmd": float(losses["mmd"].detach().cpu().item()),
            "cov": float(losses["cov"].detach().cpu().item()),
            "div": float(losses["div"].detach().cpu().item()),
            "mean": float(losses["mean"].detach().cpu().item()),
            "var": float(losses["var"].detach().cpu().item()),
            "div_scale": div_scale,
        }
        if "block_rel_l2" in losses:
            final_losses["block_rel_l2"] = float(losses["block_rel_l2"].detach().cpu().item())
        for k, v in weighted_terms.items():
            final_losses[f"weighted/{k}"] = float(v.detach().cpu().item())
        for k, v in normalized_terms.items():
            final_losses[f"normalized/{k}"] = float(v.detach().cpu().item())
        for k, v in losses.items():
            if "/" in k:
                final_losses[k] = float(v.detach().cpu().item())

        should_log_diagnostics = (
            step % args.log_interval == 0
            or step == total_target_steps - 1
        )
        if should_log_diagnostics:
            diagnostics = _compute_diagnostics(
                teacher_batch=teacher_batch,
                synthetic_hidden=synthetic_hidden,
                teacher_labels=teacher_batch_labels,
                synth_labels=synth_batch_labels,
                mmd_subsample=args.mmd_subsample,
            )
            for k, v in diagnostics.items():
                final_losses[k] = float(v.detach().cpu().item())

            teacher_baseline = _compute_teacher_baseline_losses(
                teacher_store=teacher_store,
                teacher_batch_size=teacher_batch.shape[0],
                mmd_subsample=args.mmd_subsample,
                target_device=device,
                target_dtype=train_dtype,
            )
            for k, v in teacher_baseline.items():
                final_losses[k] = float(v.detach().cpu().item())

            for key in ("mmd", "cov", "mean", "var"):
                baseline_key = f"baseline/{key}_teacher_teacher"
                if baseline_key in final_losses:
                    final_losses[f"ratio/{key}_vs_teacher_teacher"] = (
                        final_losses[key] / max(final_losses[baseline_key], 1e-8)
                    )

        should_log_wandb = (
            wandb_run is not None
            and (
                step == total_target_steps - 1
                or (
                    args.wandb_every_n_steps > 0
                    and step % args.wandb_every_n_steps == 0
                )
            )
        )
        if should_log_wandb:
            wandb_payload = {
                "step": step,
                "loss/total": final_losses["total"],
                "loss/raw/mmd": final_losses["mmd"],
                "loss/raw/cov": final_losses["cov"],
                "loss/raw/mean": final_losses["mean"],
                "loss/raw/var": final_losses["var"],
                "loss/raw/div": final_losses["div"],
                "loss/weighted/mmd": final_losses["weighted/mmd"],
                "loss/weighted/cov": final_losses["weighted/cov"],
                "loss/weighted/mean": final_losses["weighted/mean"],
                "loss/weighted/var": final_losses["weighted/var"],
                "loss/weighted/div": final_losses["weighted/div"],
                "schedule/div_scale": div_scale,
                "meta/lr": args.lr,
                "meta/use_ema_normalized_losses": int(args.use_ema_normalized_losses),
            }
            if "block_rel_l2" in final_losses:
                wandb_payload["loss/raw/block_rel_l2"] = final_losses["block_rel_l2"]
                wandb_payload["loss/weighted/block_rel_l2"] = final_losses["weighted/block_rel_l2"]
            for k, v in final_losses.items():
                if "/" in k:
                    if k.startswith(("diag/", "baseline/", "ratio/", "modality_share/")):
                        wandb_payload[k] = v
                    else:
                        wandb_payload[f"loss/{k}"] = v
            wandb_run.log(wandb_payload, step=step)
        progress.set_postfix(
            loss=f"{final_losses['total']:.4f}",
            mmd=f"{final_losses['mmd']:.4f}",
            div=f"{final_losses['div']:.4f}",
        )
        if step % args.log_interval == 0 or step == total_target_steps - 1:
            history.append({"step": step, **final_losses})
        if (
            args.checkpoint_interval > 0
            and (step + 1) % args.checkpoint_interval == 0
            and step != total_target_steps - 1
        ):
            checkpoint_hidden = _assemble_full_synthetic_hidden(
                bank_params=bank_params,
                synth_template_labels=synth_template_labels,
                synth_template_bank_indices=synth_template_bank_indices,
                synthetic_size=args.synthetic_size,
                compressed_length=init_hidden.shape[1],
                hidden_size=init_hidden.shape[2],
            )
            checkpoint_path = os.path.join(
                os.path.dirname(args.output_path),
                f"{os.path.splitext(os.path.basename(args.output_path))[0]}-step{step + 1}.pt",
            )
            _save_payload(
                output_path=checkpoint_path,
                synthetic_hidden=checkpoint_hidden,
                synth_position_ids=synth_position_ids,
                synth_labels=synth_labels,
                teacher_meta=teacher_meta,
                args=args,
                final_losses=final_losses,
                history=history,
                block_constraint_layer=block_constraint_layer,
                ablation_notes=ablation_notes,
                training_state=_build_training_state(
                    step=step + 1,
                    optimizer=optimizer,
                    loss_ema_state=loss_ema_state,
                    teacher_anchor_indices=teacher_anchor_indices,
                ),
            )
            saved_pt_paths.append(checkpoint_path)
            saved_pt_paths = _prune_saved_pt_paths(saved_pt_paths, max_keep=2)
            print(f"[representation_distill] Saved checkpoint: {checkpoint_path}")

    synthetic_hidden = _assemble_full_synthetic_hidden(
        bank_params=bank_params,
        synth_template_labels=synth_template_labels,
        synth_template_bank_indices=synth_template_bank_indices,
        synthetic_size=args.synthetic_size,
        compressed_length=init_hidden.shape[1],
        hidden_size=init_hidden.shape[2],
    )
    payload = _save_payload(
        output_path=args.output_path,
        synthetic_hidden=synthetic_hidden,
        synth_position_ids=synth_position_ids,
        synth_labels=synth_labels,
        teacher_meta=teacher_meta,
        args=args,
        final_losses=final_losses,
        history=history,
        block_constraint_layer=block_constraint_layer,
        ablation_notes=ablation_notes,
        training_state=_build_training_state(
            step=total_target_steps,
            optimizer=optimizer,
            loss_ema_state=loss_ema_state,
            teacher_anchor_indices=teacher_anchor_indices,
        ),
    )
    saved_pt_paths.append(args.output_path)
    saved_pt_paths = _prune_saved_pt_paths(saved_pt_paths, max_keep=2)
    print(f"[representation_distill] Saved synthetic calibration hidden to {args.output_path}")
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "output_path": args.output_path,
                "created_at": payload["metadata"]["created_at"],
                **{f"final/{k}": v for k, v in (final_losses or {}).items()},
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
