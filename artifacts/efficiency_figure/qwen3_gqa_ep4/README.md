# Qwen3-VL EP4 efficiency data

This directory contains four-GPU measurements for
`Qwen/Qwen3-VL-30B-A3B-Instruct` after 30% and 50% structured MoE channel
pruning. The model has 48 MoE layers, 128 experts per layer, and an original
expert intermediate width of 768.

## Protocol

- Hardware: 4 x NVIDIA H20, verified idle before every strategy
- Backend: vLLM 0.11.2, CUDA, BF16, eager mode, TP=4, EP=4
- MoE execution: vLLM fused-MoE kernels
- Task: GQA with real images
- Batch size: 8
- Samples: 8 warmup followed by 64 measured samples
- KV cache: 4 GiB per rank
- Width tiers: 0, 384, 512, 640, and 768

The plans were derived from the real checkpoint's gate/up/down weights using
mean absolute weight magnitude as a performance-experiment proxy. They should
not be treated as substitutes for calibration-based scores in accuracy claims.

## Adjusted width counts

| Pruning | Width 0 | Width 384 | Width 512 | Width 640 | Width 768 |
|---|---:|---:|---:|---:|---:|
| 30% | 0 | 70 | 4823 | 1203 | 48 |
| 50% | 96 | 5904 | 48 | 48 | 48 |

The actual pruning ratios are 29.9995% and 50.0000%. The 30% greedy placement
has 0.0194% maximum relative rank-load deviation. The 50% placement has 1.3672%
deviation, which does not satisfy the configured 1% placement tolerance.

## Summary

| Pruning | Strategy | Requests/s | Total tokens/s | Mean peak MiB | Max peak MiB |
|---|---|---:|---:|---:|---:|
| 30% | padded | 5.2702 | 1538.88 | 29671.0 | 29671.0 |
| 30% | multi_kernel | 4.2098 | 1230.13 | 35709.5 | 35969.0 |
| 30% | cross_layer | 4.8337 | 1412.05 | 26261.0 | 26527.0 |
| 50% | padded | 4.5129 | 1321.21 | 29671.0 | 29671.0 |
| 50% | multi_kernel | 3.2293 | 945.09 | 31092.0 | 31617.0 |
| 50% | cross_layer | 4.5771 | 1340.79 | 23254.0 | 23775.0 |

Use `prune_*/batch_metrics.csv` for a sample-index plot combining memory and
throughput. It contains cumulative samples, elapsed time, per-batch and
cumulative throughput, four-rank memory statistics, utilization, and power.
Use `prune_*/gpu_timeseries.csv` for the 200 ms per-GPU memory trace. Negative
elapsed times in the GPU trace are model loading and warmup observations before
the measured interval.

Requests/s is the preferred primary throughput metric because the generated
output-token count differs slightly across strategies.
