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
- 2026-09-22 17:24 CST: READY | current kimi-direct campaign is complete; waiting for all four GPUs to become free.
- 2026-09-22 17:24 CST: STARTED | worker=kimi-method1-router-worker, supervisor=kimi-method1-router-supervisor.
- 2026-09-22 17:24 CST: RUNNING | 0/28 benchmarks (0%), judges=0/2, current=waiting-for-resources, stage_elapsed=1s, GPU MiB=0/0/0/0 | session=kimi-method1-router-worker.
- 2026-09-22 17:24 CST: WARNING | gqa p30 stopped before inference on both configured attempts because `response_cache.py` raised `SyntaxError`; no sample was evaluated and no response cache was lost.
- 2026-09-22 18:18 CST: RECOVERED | fixed the response-cache identity patch's ambiguous first hunk and repaired the local checkout. Python compilation, patch reverse-check, response-cache tests (3/3), and random-subset tests (3/3) passed.
- 2026-09-22 18:18 CST: RUNNING | worker restarted at gqa p30; lmms-eval passed imports and entered vLLM model initialization. Progress remains 0/28 until the first validated result is complete.
- 2026-09-22 18:21 CST: HEALTHY | gqa p30 is performing inference; response cache advanced to 1760/6289 and all four GPUs hold about 89.9 GiB. No current stall or error is present.
