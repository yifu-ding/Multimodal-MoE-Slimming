# Kimi-VL direct-mask benchmark campaign

- Run each of 14 Kimi-VL default benchmarks with p30 followed immediately by p50.
- Use the unadjusted 26-layer channel masks and the existing `vllm-maes` EP4 environment.
- Preserve every already-complete full-dataset result. For unfinished work, use deterministic random half subsets with seed 42 and at least 500 samples (or all samples when the dataset has fewer than 500).
- Store subset outputs under `p30-random-half-seed42` / `p50-random-half-seed42`; reuse compatible response caches without restarting inference from zero.
- After inference, score MMVet, MMBench and VideoMMMU with the local judge for both ratios. For existing full prediction files, Judge uses the same half/minimum-500 policy; MMBench keeps circular groups intact.
- Preserve completed artifacts and resume from the first unvalidated task.
