# 五模型 Prefill / Decode 部署效率结果

统一协议：4 x H20、TP=4、EP、prefill 512+1、decode 32+128、warmup=1、measured=3、关闭 prefix cache、固定 2 GiB/GPU KV cache。p=0 时 Kimi/Qwen3-VL-30B/InternVL3.5 使用 batch=512，Mistral-119B/Qwen3-VL-235B 使用 batch=64；p=0.3/0.5 的当前正式结果均为 batch=64。吞吐为三次均值；显存为四卡最大 peak，non-KV 为该峰值减 2 GiB。

## p = 0 (Default, Unpruned)

| 模型 | Batch size | Implementation | Prefill input tok/s | Decode output tok/s | 最大峰值显存 (GiB) | non-KV 峰值 (GiB) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Kimi-VL-A3B-Instruct | 512 | Default | 56024.94 | 8208.45 | 25.11 | 23.11 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 512 | Default | 44662.39 | 8431.61 | 26.06 | 24.06 | 完成 |
| InternVL3.5-30B-A3B-HF | 512 | Default | 44585.55 | 9401.26 | 26.13 | 24.13 | 完成 |
| Mistral-Small-4-119B-2603 | 64 | Default | 33484.58 | 880.44 | 36.48 | 34.48 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 64 | Default | 10667.34 | 584.49 | 65.23 | 63.23 | 完成 |

## p = 0.3

| 模型 | 实际剪枝率 | Implementation | Prefill input tok/s | Decode output tok/s | 最大峰值显存 (GiB) | non-KV 峰值 (GiB) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Kimi-VL-A3B-Instruct | 0.299716 | Padding | 52009.99 | 1486.65 | 16.35 | 14.35 | 完成 |
| Kimi-VL-A3B-Instruct | 0.299716 | Multi-kernel | 39420.21 | 761.78 | 19.59 | 17.59 | 完成 |
| Kimi-VL-A3B-Instruct | 0.299716 | Single-width | 59321.00 | 1509.05 | 16.35 | 14.35 | 完成 |
| Kimi-VL-A3B-Instruct | 0.299716 | Ours (cross-layer) | 57324.33 | 1492.37 | 14.85 | 12.85 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.299479 | Padding | 43342.38 | 1365.23 | 21.61 | 19.61 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.299479 | Multi-kernel | 26661.72 | 638.65 | 27.52 | 25.52 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.299479 | Single-width | 49898.97 | 1371.07 | 21.61 | 19.61 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.299479 | Ours (cross-layer) | 47098.06 | 1392.06 | 17.55 | 15.55 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.299479 | Padding | 43827.28 | 1520.65 | 21.68 | 19.68 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.299479 | Multi-kernel | 27023.64 | 671.21 | 27.73 | 25.73 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.299479 | Single-width | 49646.75 | 1485.11 | 21.69 | 19.69 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.299479 | Ours (cross-layer) | 46648.03 | 1565.16 | 17.62 | 15.62 | 完成 |
| Mistral-Small-4-119B-2603 | 0.299805 | Padding | 32867.21 | 901.23 | 36.48 | 34.48 | 完成 |
| Mistral-Small-4-119B-2603 | 0.299805 | Multi-kernel | 24026.20 | 485.10 | 48.35 | 46.35 | 完成 |
| Mistral-Small-4-119B-2603 | 0.299805 | Single-width | 36587.35 | 901.87 | 36.48 | 34.48 | 完成 |
| Mistral-Small-4-119B-2603 | 0.299805 | Ours (cross-layer) | 35132.99 | 886.23 | 28.70 | 26.70 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.299479 | Padding | 10608.02 | 596.00 | 65.23 | 63.23 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.299479 | Multi-kernel | 6108.61 | 270.36 | 88.54 | 86.54 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.299479 | Single-width | 11580.73 | 589.09 | 65.23 | 63.23 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.299479 | Ours (cross-layer) | 10961.03 | 599.26 | 49.95 | 47.95 | 完成 |

## p = 0.5

| 模型 | 实际剪枝率 | Implementation | Prefill input tok/s | Decode output tok/s | 最大峰值显存 (GiB) | non-KV 峰值 (GiB) | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Kimi-VL-A3B-Instruct | 0.500000 | Padding | 50949.64 | 1497.28 | 16.35 | 14.35 | 完成 |
| Kimi-VL-A3B-Instruct | 0.500000 | Multi-kernel | 42163.91 | 773.66 | 17.01 | 15.01 | 完成 |
| Kimi-VL-A3B-Instruct | 0.500000 | Single-width | 59320.34 | 1491.36 | 16.35 | 14.35 | 完成 |
| Kimi-VL-A3B-Instruct | 0.500000 | Ours (cross-layer) | 63934.90 | 1522.61 | 13.53 | 11.53 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.500000 | Padding | 42519.90 | 1384.70 | 21.61 | 19.61 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.500000 | Multi-kernel | 28200.39 | 628.85 | 22.13 | 20.13 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.500000 | Single-width | 50249.92 | 1384.04 | 21.61 | 19.61 | 完成 |
| Qwen3-VL-30B-A3B-Instruct | 0.500000 | Ours (cross-layer) | 52261.01 | 1389.22 | 15.14 | 13.14 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.500000 | Padding | 44413.05 | 1547.11 | 21.68 | 19.68 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.500000 | Multi-kernel | 28590.02 | 660.65 | 22.33 | 20.33 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.500000 | Single-width | 49618.65 | 1545.19 | 21.68 | 19.68 | 完成 |
| InternVL3.5-30B-A3B-HF | 0.500000 | Ours (cross-layer) | 53289.76 | 1564.00 | 15.21 | 13.21 | 完成 |
| Mistral-Small-4-119B-2603 | 0.499973 | Padding | 32514.69 | 856.36 | 36.11 | 34.11 | 完成 |
| Mistral-Small-4-119B-2603 | 0.499973 | Multi-kernel | 24253.74 | 477.44 | 37.60 | 35.60 | 完成 |
| Mistral-Small-4-119B-2603 | 0.499973 | Single-width | 35698.33 | 890.34 | 36.11 | 34.11 | 完成 |
| Mistral-Small-4-119B-2603 | 0.499973 | Ours (cross-layer) | 36900.95 | 897.22 | 22.94 | 20.94 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.500000 | Padding | 10297.75 | 586.91 | 65.23 | 63.23 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.500000 | Multi-kernel | 6334.12 | 270.64 | 67.37 | 65.37 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.500000 | Single-width | 11581.89 | 586.26 | 65.23 | 63.23 | 完成 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 0.500000 | Ours (cross-layer) | 12351.05 | 583.11 | 39.27 | 37.27 | 完成 |
