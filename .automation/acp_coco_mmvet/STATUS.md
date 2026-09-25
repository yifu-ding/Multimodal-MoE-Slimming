# ACP COCO and MMVet status

> [!IMPORTANT]
> **PREPARED (2026-09-26 CST)**
> The old align=128 result directories and plans are no longer present locally.
> Per the current instruction, the retained mixed-512 scores will be used to
> rebuild the original align=128 ACP setting for the missing COCO/MMVet results.

Report: `docs/acp_coco_mmvet_results.md`. Raw results: `results/vllm_acp_align128_completion/`.
- 2026-09-26 01:11 CST: RUNNING | acp-align128 plans=1/6 coco=0/6 mmvet_predict=0/6 mmvet_judge=0/6 current=internvl30_p50/mmvet-judge | session=maes-acp-coco-mmvet-worker.

> [!IMPORTANT]
> **BATCH INCREASED (2026-09-26 CST)**
> Kimi p=0.3 COCO completed normally with batch 32. Its result and cache were
> preserved. Remaining inference uses COCO batch 64 and MMVet batch 32 after
> observing empty scheduler wait queues and underused KV cache at batch 32.

- 2026-09-26 01:21 CST: The first resume briefly invalidated the COCO marker
  because its task signature included batch size. It was stopped before model
  inference, the original marker/result was restored, and Kimi p=0.3 now resumes
  with `TASKS=mmvet` only. Other configurations use both tasks at the larger batches.

> [!IMPORTANT]
> **RECOVERED (2026-09-26 01:50 CST)**
> Kimi p=0.3 MMVet prediction completed, but the report updater selected the
> COCO submission JSON instead of the aggregate result and stopped the worker.
> The updater now validates the JSON structure. Its standalone check passed,
> and the resumable worker restarted without rerunning either completed Kimi task.

> [!IMPORTANT]
> **JUDGE SMOKE PASSED (2026-09-26 01:56 CST)**
> Per request, Qwen p=0.3 COCO was paused after 832/2500 cached responses.
> The local Qwen2.5-32B Judge scored all 109 Kimi p=0.3 MMVet samples:
> score 64.8624, failed 0. The local server shut down cleanly; inference then resumed
> from the persisted response cache.
- 2026-09-26 02:41 CST: RUNNING | acp-align128 plans=6/6 coco=5/6 mmvet_predict=4/6 mmvet_judge=1/6 current=qwen30_p30/mmvet-judge | session=maes-acp-coco-mmvet-worker.

> [!DONE]
> **COMPLETE (2026-09-26 03:08 CST)**
> All 12 ACP COCO/MMVet results passed validation.
