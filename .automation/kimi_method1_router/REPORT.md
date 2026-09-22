# Kimi-VL 方法1-only Router Direct Mask Campaign

> [!IMPORTANT]
> **Prepared 2026-09-22**
> 已排队等待当前 `kimi-direct` campaign 完整结束。配置为 `intra_layer_method=router`、`modality_aware=true`，只使用方法1，不使用二阶 attribution 方法2；Direct Mask 不做圆整、上下取整规整或 Rearrange。

## Events

- 2026-09-22 00:03 CST: QUEUED | waiting for the current kimi-direct campaign to validate 28/28 benchmarks and 2/2 Judge stages.
- 2026-09-22 00:03 CST: VERIFIED | Kimi scores contain nonzero `router`, `usage`, `token_count_text`, and `token_count_visual` values for all 26 × 64 experts; no `router_fillzero` or `usage_fillzero` key exists, so the native supported `router` method is used.
- 2026-09-22 00:03 CST: VERIFIED | p30/p50 Router Direct Masks have actual prune ratios 0.2999998/0.5000000 and 734/816 unique expert widths. `align_inter=0`, `min_per_expert=0`; no rounding or Rearrange is applied.
- 2026-09-22 00:03 CST: VERIFIED | with the same p30 Router configuration, toggling `modality_aware` changes 228,779 mask entries, confirming that the text/visual path remains active.
- 2026-09-22 00:03 CST: WAITING | queue session=`kimi-method1-router-queue`; future worker=`kimi-method1-router-worker`; future supervisor=`kimi-method1-router-supervisor`.
- 2026-09-22 15:44 CST: UPDATED | campaign changed to deterministic random 1/2 subsets (`seed=42`, minimum 500, or full dataset below 500). Results use separate `*-random-half-seed42` directories and are not mixed with full-dataset metrics.
- 2026-09-22 15:51 CST: QUEUED | waiting for the current kimi-direct campaign to validate 28/28 benchmarks and 2/2 Judge stages.
