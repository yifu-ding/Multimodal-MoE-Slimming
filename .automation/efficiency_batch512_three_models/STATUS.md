# Three-model batch=512 efficiency status

> [!IMPORTANT]
> **LAUNCHED (2026-09-26 CST)**
> The mistaken batch=112 campaign was stopped and all of its artifacts were deleted. Existing p=0 batch=512 results for all three models passed validation; this queue runs only the 24 missing p=0.3/p=0.5 cases.

Raw artifacts: `artifacts/efficiency_batch512_three_models/`. Generated report: `docs/efficiency_batch512_results.md`.
- 2026-09-26 00:07 CST: RUNNING | batch512 p0=3/3 new=0/24 failed=0 current=p30/kimi/padded/bs_512 | session=maes-efficiency-bs512-worker.

> [!DONE]
> **COMPLETE (2026-09-26 01:02 CST)**
> All three p=0 batch=512 baselines and all 24 p=0.3/p=0.5 cases passed artifact validation. No failed cases remain.

- 2026-09-26 01:02 CST: COMPLETE | batch512 p0=3/3 new=24/24 failed=0 | full completion check passed.
