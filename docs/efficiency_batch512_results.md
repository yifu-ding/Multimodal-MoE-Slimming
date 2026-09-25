# Three-model batch=512 efficiency results

统一协议：4 x H20、TP=4、EP、batch=512、prefill 512+1、decode 32+128、warmup=1、measured=3、关闭 prefix cache、固定 2 GiB/GPU KV cache。吞吐为三次均值；显存为四卡最大 peak，non-KV 为该峰值减 2 GiB。

## p = 0

| Model | Implementation | Prefill input tok/s | Decode output tok/s | Peak memory (GiB) | non-KV peak (GiB) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Kimi-VL-A3B-Instruct | Default | 56024.94 | 8208.45 | 25.11 | 23.11 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Default | 44662.39 | 8431.61 | 26.06 | 24.06 | 完成 |
| InternVL3.5-30B-A3B-HF | Default | 44585.55 | 9401.26 | 26.13 | 24.13 | 完成 |

## p = 0.3

| Model | Implementation | Prefill input tok/s | Decode output tok/s | Peak memory (GiB) | non-KV peak (GiB) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Kimi-VL-A3B-Instruct | Padding | 54701.36 | 8369.37 | 25.11 | 23.11 | 完成 |
| Kimi-VL-A3B-Instruct | Multi-kernel | 42827.64 | 4381.41 | 28.33 | 26.33 | 完成 |
| Kimi-VL-A3B-Instruct | Single-width | 63049.16 | 8419.70 | 25.11 | 23.11 | 完成 |
| Kimi-VL-A3B-Instruct | Ours (cross-layer) | 60488.14 | 8428.58 | 23.21 | 21.21 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Padding | 45391.23 | 8983.19 | 26.05 | 24.05 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Multi-kernel | 28224.65 | 4524.90 | 34.08 | 32.08 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Single-width | 53106.88 | 9002.97 | 26.05 | 24.05 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Ours (cross-layer) | 49458.98 | 8948.54 | 22.04 | 20.04 | 完成 |
| InternVL3.5-30B-A3B-HF | Padding | 45734.67 | 9914.64 | 26.12 | 24.12 | 完成 |
| InternVL3.5-30B-A3B-HF | Multi-kernel | 28563.78 | 4715.81 | 34.16 | 32.16 | 完成 |
| InternVL3.5-30B-A3B-HF | Single-width | 52106.88 | 10036.29 | 26.12 | 24.12 | 完成 |
| InternVL3.5-30B-A3B-HF | Ours (cross-layer) | 48929.48 | 9976.40 | 22.11 | 20.11 | 完成 |

## p = 0.5

| Model | Implementation | Prefill input tok/s | Decode output tok/s | Peak memory (GiB) | non-KV peak (GiB) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Kimi-VL-A3B-Instruct | Padding | 53800.26 | 8262.84 | 25.11 | 23.11 | 完成 |
| Kimi-VL-A3B-Instruct | Multi-kernel | 46167.32 | 4434.06 | 25.75 | 23.75 | 完成 |
| Kimi-VL-A3B-Instruct | Single-width | 62598.92 | 8346.21 | 25.11 | 23.11 | 完成 |
| Kimi-VL-A3B-Instruct | Ours (cross-layer) | 68503.11 | 8365.11 | 21.85 | 19.85 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Padding | 44594.87 | 8952.61 | 26.05 | 24.05 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Multi-kernel | 30126.15 | 4584.08 | 28.68 | 26.68 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Single-width | 53323.79 | 8923.37 | 26.05 | 24.05 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | Ours (cross-layer) | 55504.81 | 9006.34 | 19.34 | 17.34 | 完成 |
| InternVL3.5-30B-A3B-HF | Padding | 46464.30 | 9757.42 | 26.13 | 24.13 | 完成 |
| InternVL3.5-30B-A3B-HF | Multi-kernel | 30477.85 | 4809.83 | 28.76 | 26.76 | 完成 |
| InternVL3.5-30B-A3B-HF | Single-width | 52175.01 | 10038.14 | 26.12 | 24.12 | 完成 |
| InternVL3.5-30B-A3B-HF | Ours (cross-layer) | 56287.42 | 9989.59 | 19.41 | 17.41 | 完成 |
