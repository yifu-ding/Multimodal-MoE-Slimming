#!/usr/bin/env python3
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "artifacts/efficiency_batch512_three_models/runs"
P0 = ROOT / "artifacts/efficiency_p0/runs"
OUT = ROOT / "docs/efficiency_batch512_results.md"
MODELS = (("kimi", "Kimi-VL-A3B-Instruct"), ("qwen30", "Qwen3-VL-30B-A3B-Instruct"), ("internvl30", "InternVL3.5-30B-A3B-HF"))
STRATEGIES = (("padded", "Padding"), ("multi_kernel", "Multi-kernel"), ("single_width", "Single-width"), ("cross_layer", "Ours (cross-layer)"))

def mean(rows, key): return sum(float(x[key]) for x in rows) / len(rows)

def metrics(path):
    if not (path / "COMPLETE").exists(): return ("-", "-", "-", "-", "失败" if (path / "FAILED").exists() else "待运行")
    result = json.loads((path / "result.json").read_text())
    with (path / "gpu_trace.csv").open() as handle:
        peak = max(float(row["memory_used_mib"]) for row in csv.DictReader(handle))
    kv = result["kv_cache_memory_bytes_per_gpu"] / 1024**2
    return (f'{mean(result["prefill"]["measured_runs"], "input_tokens_per_second"):.2f}', f'{mean(result["decode"]["measured_runs"], "output_tokens_per_second"):.2f}', f"{peak/1024:.2f}", f"{(peak-kv)/1024:.2f}", "完成")

def main():
    lines = ["# Three-model batch=512 efficiency results", "", "统一协议：4 x H20、TP=4、EP、batch=512、prefill 512+1、decode 32+128、warmup=1、measured=3、关闭 prefix cache、固定 2 GiB/GPU KV cache。吞吐为三次均值；显存为四卡最大 peak，non-KV 为该峰值减 2 GiB。", ""]
    for ratio, title in (("p0", "p = 0"), ("p30", "p = 0.3"), ("p50", "p = 0.5")):
        lines += [f"## {title}", "", "| Model | Implementation | Prefill input tok/s | Decode output tok/s | Peak memory (GiB) | non-KV peak (GiB) | Status |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for model, label in MODELS:
            entries = (("default", "Default"),) if ratio == "p0" else STRATEGIES
            for strategy, strategy_label in entries:
                path = P0/model/"default"/"bs_512" if ratio == "p0" else NEW/ratio/model/strategy/"bs_512"
                lines.append(f"| {label} | {strategy_label} | " + " | ".join(metrics(path)) + " |")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__": main()
