#!/usr/bin/env python3
import csv, json, math, sys
from pathlib import Path

def main():
    path = Path(sys.argv[1]); model, ratio, strategy = sys.argv[2:5]
    try:
        identity = json.loads((path / "run_identity.json").read_text())
        result = json.loads((path / "result.json").read_text())
        trace = list(csv.DictReader((path / "gpu_trace.csv").open()))
        runs = result["prefill"]["measured_runs"] + result["decode"]["measured_runs"]
        valid = (
            (path / "COMPLETE").is_file()
            and identity["model_key"] == model
            and float(identity["prune_ratio"]) == {"p30": 0.3, "p50": 0.5}[ratio]
            and identity["implementation"] == strategy
            and identity["batch_size"] == 512
            and identity["prefill_tokens"] == 512
            and identity["decode_prompt_tokens"] == 32
            and identity["decode_tokens"] == 128
            and identity["warmup_runs"] == 1
            and identity["measured_runs"] == 3
            and identity["prefix_cache"] is False
            and identity["kv_cache_memory_bytes_per_gpu"] == 2 * 1024**3
            and len(identity["plan_sha256"]) == 64
            and result["tensor_parallel_size"] == 4
            and result["expert_parallel"] is True
            and result["prefill"]["batch_size"] == 512
            and result["decode"]["batch_size"] == 512
            and len(result["prefill"]["measured_runs"]) == 3
            and len(result["decode"]["measured_runs"]) == 3
            and trace and {int(row["index"]) for row in trace} == {0, 1, 2, 3}
            and all(math.isfinite(float(run[key])) and float(run[key]) > 0
                    for run in runs for key in ("elapsed_seconds", "input_tokens_per_second", "output_tokens_per_second"))
        )
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return 1
    return 0 if valid else 1

if __name__ == "__main__": raise SystemExit(main())
