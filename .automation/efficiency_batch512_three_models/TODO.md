# Three-model p=0.3/p=0.5 batch=512 efficiency campaign

- [ ] Measure p=0.3 Padding, Multi-kernel, Single-width, and Ours for Kimi, Qwen3-VL-30B, and InternVL3.5-30B.
- [ ] Measure p=0.5 Padding, Multi-kernel, Single-width, and Ours for the same three models.
- [ ] Combine the 24 new cases with the validated p=0 batch=512 results and generate a throughput/memory report.

Fixed protocol: 4 x H20, TP=4, EP enabled, batch=512, prefill 512+1,
decode 32+128, warmup=1, measured=3, prefix cache disabled, 2 GiB/GPU
KV cache, max_model_len=2048, max_num_seqs=512,
max_num_batched_tokens=262144, seed=2603.

Mistral-Small-4-119B-2603 and Qwen3-VL-235B-A22B-Instruct-FP8 are out of scope.
