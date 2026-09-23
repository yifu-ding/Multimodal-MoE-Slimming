#!/usr/bin/env python3
"""Measure prefill and decode throughput for a loaded EP4 model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--seed", type=int, default=2603)
    return parser.parse_args()


def requests(batch_size: int, prompt_tokens: int) -> list[dict[str, list[int]]]:
    return [
        {"prompt_token_ids": [1] + [42] * (prompt_tokens - 1)}
        for _ in range(batch_size)
    ]


def measure(
    llm: LLM,
    *,
    name: str,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
    warmup_runs: int,
    measured_runs: int,
) -> dict:
    prompts = requests(batch_size, prompt_tokens)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
    )
    for _ in range(warmup_runs):
        llm.generate(prompts, sampling, use_tqdm=False)
    runs = []
    for repeat in range(measured_runs):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        generated = sum(len(item.outputs[0].token_ids) for item in outputs)
        runs.append(
            {
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "requests_per_second": batch_size / elapsed,
                "input_tokens_per_second": batch_size * prompt_tokens / elapsed,
                "output_tokens_per_second": generated / elapsed,
                "generated_tokens": generated,
            }
        )
    return {
        "workload": name,
        "batch_size": batch_size,
        "prompt_tokens_per_request": prompt_tokens,
        "requested_output_tokens_per_request": output_tokens,
        "warmup_runs": warmup_runs,
        "measured_runs": runs,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.measured_runs <= 0 or args.warmup_runs < 0:
        raise SystemExit("batch/measured runs must be positive and warmup non-negative")
    required_length = max(
        args.prefill_tokens + 1, args.decode_prompt_tokens + args.decode_tokens
    )
    if args.max_model_len < required_length:
        raise SystemExit(
            f"max model length {args.max_model_len} is below workload length {required_length}"
        )

    is_mistral = Path(args.model_path).name.lower().startswith("mistral")
    tokenizer_mode = "mistral" if is_mistral else "auto"
    modality_limits = {"image": 0} if is_mistral else {"image": 0, "video": 0}
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tokenizer_mode=tokenizer_mode,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        dtype="auto",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=False,
        max_num_batched_tokens=(
            args.max_num_batched_tokens
            if args.max_num_batched_tokens is not None
            else max(16384, args.batch_size * args.prefill_tokens)
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        seed=args.seed,
        limit_mm_per_prompt=modality_limits,
        disable_log_stats=False,
    )
    result = {
        "model_path": str(Path(args.model_path).resolve()),
        "seed": args.seed,
        "tensor_parallel_size": 4,
        "expert_parallel": True,
        "kv_cache_memory_bytes_per_gpu": args.kv_cache_memory_bytes,
        "prefill": measure(
            llm,
            name="prefill",
            batch_size=args.batch_size,
            prompt_tokens=args.prefill_tokens,
            output_tokens=1,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        ),
        "decode": measure(
            llm,
            name="decode",
            batch_size=args.batch_size,
            prompt_tokens=args.decode_prompt_tokens,
            output_tokens=args.decode_tokens,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
