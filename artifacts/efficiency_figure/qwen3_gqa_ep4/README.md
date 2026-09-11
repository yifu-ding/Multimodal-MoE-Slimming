# Qwen3-VL EP4 balanced prefill efficiency data

This directory contains the final plot-ready four-GPU measurements for
`Qwen/Qwen3-VL-30B-A3B-Instruct` after 30% and 50% structured MoE channel
pruning.

Use only these result directories:

```text
batch_sweep_balanced_prefill/prune_30/
batch_sweep_balanced_prefill/prune_50/
```

Each directory contains `throughput_memory_curve.csv` (21 points),
`best_by_strategy.csv`, `sweep_status.tsv`, and `sweep_summary.json`. The curve
CSV is the primary plotting input.

## Protocol

- Hardware: 4 x NVIDIA H20, verified idle before every point
- Backend: vLLM 0.11.2, CUDA, BF16, eager mode, TP4 + EP4
- MoE execution: vLLM fused-MoE kernels
- Task: GQA with real images, prefill only (`max_new_tokens=1`)
- Pruning ratios: 30% and 50%
- Strategies: padded, multi-kernel, and per-layer greedy cross-layer
- Batch sizes: 8, 16, 32, 64, 128, 256, and 512
- Repetitions: 1 warmup batch + 4 measured batches per point
- Memory: `gpu_memory_utilization=0.90`; vLLM automatically sizes KV cache
- Width tiers: 0, 384, 512, 640, and 768

The performance-only plans balance the four active width tiers. At 30%
pruning, every layer has 29-30 experts in each active tier and 8-9 zero-width
experts. At 50%, each active tier has 21-22 experts per layer and the zero tier
has 42-43. Actual pruning is 29.9995% and 50.0000%, respectively.

The real checkpoint's gate/up/down mean-absolute weight magnitude determines
expert identities and channel order. These balanced plans are intended for
efficiency measurements and should not be used for accuracy claims.

## Plotting

For each pruning ratio, plot:

- `requests_per_second` against `batch_size`
- `max_non_kv_peak_memory_mib / 1024` against `batch_size`
- optionally, `input_tokens_per_second` against `batch_size`

Use the maximum memory across ranks because the heaviest rank determines the
capacity limit. Non-KV peak includes weights, CUDA context, communication and
MoE workspaces, allocator cache, and activations. It is not a pure
weights-plus-activations measurement.

See `docs/qwen3_ep4_efficiency_plotting.md` for exact commands, the batch-512
summary, metric interpretation, and figure recommendations.
