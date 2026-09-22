# Kimi-VL direct-mask campaign

> [!IMPORTANT]
> **Prepared 2026-09-21**
> 14 benchmarks × p30/p50, followed by two local Judge stages. Existing valid full runs are preserved; unfinished work uses deterministic random 1/2 subsets (`seed=42`, minimum 500).

Status is checked from completion markers, full-run configuration, result JSON and sample JSONL. Logs and artifacts: `results/vllm_ours/kimi/direct-mask/p30-full` and `p50-full`.

## Events
- 2026-09-21 08:53 CST: RUNNING | 0/28 benchmarks (0%), judges=0/2, current=gqa p30 | session=kimi-direct-worker.
- 2026-09-21 08:56 CST: GQA p30 is generating full predictions; response cache has persisted 1,056/12,578 requests. Both worker and supervisor sessions are alive; all four H20s are at about 89,853 MiB and 99–100% utilization.
- 2026-09-21 10:23 CST: RUNNING | 6/28 benchmarks (21%), judges=0/2, current=chartqa p30, stage_elapsed=111s, GPU MiB=89841/89841/89841/89841 | session=kimi-direct-worker.
- 2026-09-21 11:53 CST: RUNNING | 13/28 benchmarks (46%), judges=0/2, current=mmvet p50, stage_elapsed=364s, GPU MiB=89843/89843/89843/89843 | session=kimi-direct-worker.
- 2026-09-21 13:23 CST: RUNNING | 18/28 benchmarks (64%), judges=0/2, current=videomme p30, stage_elapsed=3303s, GPU MiB=89863/89863/89863/89863 | session=kimi-direct-worker.
- 2026-09-21 13:56 CST: HEALTHY | 18/28 benchmarks (64%), judges=0/2, current=videomme p30. Response cache advanced from 1075/2700 to 1107/2700 in 162s; pipeline log and cache audit continued updating; all four H20 workers were runnable at 99-100% CPU and GPUs were at 100% utilization. This is a long video stage, not a stall.

> [!WARNING]
> Completion validation has a configuration-consistency gap: the shared runner's task signature does not include the direct mask plan, while `campaign.py` only checks that a `signature=` line exists. Reusing a ratio run directory with a different mask could therefore accept stale artifacts. Do not hot-edit the active scripts; fix the signature and validation after this campaign stops.
- 2026-09-21 14:53 CST: RUNNING | 18/28 benchmarks (64%), judges=0/2, current=videomme p30, stage_elapsed=8703s, GPU MiB=89865/89865/89865/89865 | session=kimi-direct-worker.
- 2026-09-21 15:12 CST: HEALTHY | 18/28 benchmarks (64%), judges=0/2, current=videomme p30, response_cache=2013/2700. Cache advanced from 1162 to 2013 in about 72 minutes and continued advancing during a 10-second resample; estimated current-stage ETA is about 55-60 minutes. Worker and supervisor tmux sessions remain alive. GitHub `origin/ep4_intplan` was fetched, and an isolated clean worktree was fast-forwarded to `ca1bc7d`; the dirty live workspace was not overwritten.
- 2026-09-21 17:18 CST: HEALTHY | 19/28 benchmarks (68%), judges=0/2, current=videomme p50, response_cache=819/2700. VideoMME p30 completed at 16:13 and the dispatcher advanced automatically. The p50 cache advanced from 808 to 819 during inspection, log and watchdog timestamps remained current, and the four H20s were observed at 74-100% utilization with about 89.9 GiB allocated per card. No current-stage OOM, stall timeout, killed process, or evaluation error was found. Estimated p50 VideoMME ETA is about 2.5 hours at the observed average rate.
- 2026-09-21 16:23 CST: RUNNING | 19/28 benchmarks (68%), judges=0/2, current=videomme p50, stage_elapsed=602s, GPU MiB=89861/89861/89861/89861 | session=kimi-direct-worker.
- 2026-09-21 17:53 CST: RUNNING | 19/28 benchmarks (68%), judges=0/2, current=videomme p50, stage_elapsed=6002s, GPU MiB=89863/89863/89863/89863 | session=kimi-direct-worker.
- 2026-09-21 19:23 CST: RUNNING | 19/28 benchmarks (68%), judges=0/2, current=videomme p50, stage_elapsed=11403s, GPU MiB=89865/89865/89865/89865 | session=kimi-direct-worker.
- 2026-09-21 20:53 CST: RUNNING | 20/28 benchmarks (71%), judges=0/2, current=longvideobench_val_v p30, stage_elapsed=3163s, GPU MiB=89863/89863/89863/89863 | session=kimi-direct-worker.
- 2026-09-21 21:48 CST: HEALTHY | 20/28 benchmarks (71%), judges=0/2, current=longvideobench_val_v p30, response_cache=1062/1337 (79.4%). Cache advanced from 1044 to 1062 during inspection; logs and watchdog remained current and all four H20s were observed at 100% utilization. No current-stage OOM, timeout, killed process, or evaluation error was found. New validated results: VideoMME p30=63.4815 and p50=60.0370, both with 2700/2700 samples; VideoMME p50 completed at 20:00. Estimated LongVideoBench p30 ETA is about 25-30 minutes.
- 2026-09-21 22:23 CST: RUNNING | 21/28 benchmarks (75%), judges=0/2, current=longvideobench_val_v p50, stage_elapsed=357s, GPU MiB=89861/89861/89861/89861 | session=kimi-direct-worker.
- 2026-09-21 23:53 CST: RUNNING | 21/28 benchmarks (75%), judges=0/2, current=longvideobench_val_v p50, stage_elapsed=5757s, GPU MiB=89865/89865/89865/89865 | session=kimi-direct-worker.
- 2026-09-22 01:23 CST: RUNNING | 22/28 benchmarks (79%), judges=0/2, current=video_mmmu_local p30, stage_elapsed=2799s, GPU MiB=89881/89881/89881/89881 | session=kimi-direct-worker.
- 2026-09-22 02:53 CST: RUNNING | 23/28 benchmarks (82%), judges=0/2, current=video_mmmu_local p50, stage_elapsed=1508s, GPU MiB=89877/89877/89877/89877 | session=kimi-direct-worker.
- 2026-09-22 04:23 CST: RUNNING | 23/28 benchmarks (82%), judges=0/2, current=video_mmmu_local p50, stage_elapsed=6908s, GPU MiB=89877/89877/89877/89877 | session=kimi-direct-worker.
- 2026-09-22 05:53 CST: RUNNING | 23/28 benchmarks (82%), judges=0/2, current=video_mmmu_local p50, stage_elapsed=12308s, GPU MiB=89877/89877/89877/89877 | session=kimi-direct-worker.
- 2026-09-22 07:23 CST: RUNNING | 23/28 benchmarks (82%), judges=0/2, current=video_mmmu_local p50, stage_elapsed=17708s, GPU MiB=89877/89877/89877/89877 | session=kimi-direct-worker.
- 2026-09-22 08:52 CST: RECOVERY | p50 VideoMMMU had stopped at response cache 134/900 after decord failed on `validation_Electronics_13.mp4`; the parent evaluator was killed and four orphan vLLM workers retained about 89.9 GiB per GPU at 0% utilization. The stale worker session and owned orphan processes were terminated. Added a selective OpenCV qwen-vl-utils reader, enabled only for `video_mmmu_local`, with reload-safe installation through the actual Kimi `sitecustomize`/`maes_ep4_bootstrap` path. The failing video decoded 32 frames in 4.66s, cache advanced from 134 to 140, and all four GPUs returned to 100% utilization. Existing cached responses were preserved.
- 2026-09-22 08:53 CST: RUNNING | 23/28 benchmarks (82%), judges=0/2, current=video_mmmu_local p50, stage_elapsed=239s, GPU MiB=89875/89875/89875/89875 | session=kimi-direct-worker.
- 2026-09-22 10:23 CST: RUNNING | 24/28 benchmarks (86%), judges=0/2, current=egoschema_subset p30, stage_elapsed=319s, GPU MiB=89847/89847/89847/89847 | session=kimi-direct-worker.
- 2026-09-22 11:53 CST: RUNNING | 26/28 benchmarks (93%), judges=0/2, current=mvbench_available_3800 p30, stage_elapsed=3073s, GPU MiB=89873/89873/89873/89873 | session=kimi-direct-worker.
- 2026-09-22 13:23 CST: RUNNING | 26/28 benchmarks (93%), judges=0/2, current=mvbench_available_3800 p30, stage_elapsed=8473s, GPU MiB=89875/89875/89875/89875 | session=kimi-direct-worker.
- 2026-09-22 14:53 CST: RUNNING | 27/28 benchmarks (96%), judges=0/2, current=mvbench_available_3800 p50, stage_elapsed=1114s, GPU MiB=89855/89855/89855/89855 | session=kimi-direct-worker.
- 2026-09-22 15:18 CST: PAUSED | p50 MVBench was intentionally interrupted after 1111/3800 cached responses to adopt the new random-half policy. SQLite integrity is `ok`; this was not a crash or stall.
- 2026-09-22 15:44 CST: UPDATED | unfinished inference uses random 1/2 (`seed=42`, minimum 500), with separate `*-random-half-seed42` outputs. p50 MVBench will reuse its existing full-run response cache and only infer random-subset misses. Pending Judge uses half subsets for existing full MMBench/VideoMMMU predictions and full MMVet because it has only 218 samples.
- 2026-09-22 15:51 CST: RUNNING | 27/28 benchmarks (96%), judges=0/2, current=mvbench_available_3800 p50, stage_elapsed=4607s, GPU MiB=0/0/0/0 | session=kimi-direct-worker.
