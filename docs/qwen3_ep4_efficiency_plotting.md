# Qwen3-VL EP4 效率实验画图说明

## 最终数据

画图只使用本轮 balanced、prefill-only 四卡 sweep：

```text
artifacts/efficiency_figure/qwen3_gqa_ep4/batch_sweep_balanced_prefill/prune_30/
artifacts/efficiency_figure/qwen3_gqa_ep4/batch_sweep_balanced_prefill/prune_50/
```

每个剪枝比例目录包含：

- `throughput_memory_curve.csv`：三种策略的完整 batch-size 曲线，共 21 个画图点
- `best_by_strategy.csv`：每种策略在已测试范围内的最佳吞吐点
- `sweep_status.tsv`：每个点的完成状态和搜索终止状态
- `sweep_summary.json`：上述结果的结构化汇总

旧的 `batch_sweep`、单点测试、unbalanced 分配和 block12 ablation 已删除，不应再用于画图。

## 实验协议

- 模型：`Qwen/Qwen3-VL-30B-A3B-Instruct`
- 模型结构：48 个 MoE 层，每层 128 个专家，原始专家宽度 768
- 任务：GQA，使用真实图像
- 硬件：4 x NVIDIA H20；每个点开始前确认四卡无其他 compute process 且显存占用不超过 64 MiB
- 后端：vLLM 0.11.2、CUDA、BF16、TP=4、EP=4、eager mode
- MoE 执行：vLLM fused-MoE kernel
- 剪枝比例：30%、50%
- 宽度档位：0、384、512、640、768
- 策略：`padded`、`multi_kernel`、逐层 greedy `cross_layer`
- batch size：8、16、32、64、128、256、512
- 每个点：1 个 warmup batch + 4 个 measured batch
- 生成设置：`max_new_tokens=1`，只执行 prefill 和产生首 token，不包含迭代 decode
- 显存设置：`gpu_memory_utilization=0.90`，未固定 KV cache，交由 vLLM 自动分配

balanced plan 使用真实 Qwen 权重的 gate/up/down mean-absolute magnitude 确定专家身份和通道顺序，但为纯性能实验强制平衡四个活跃宽度档位。它不用于精度结论。

| Pruning | 每层 Width 384 | 每层 Width 512 | 每层 Width 640 | 每层 Width 768 | 每层 Width 0 | 实际剪枝率 |
|---|---:|---:|---:|---:|---:|---:|
| 30% | 29-30 | 29-30 | 29-30 | 29-30 | 8-9 | 29.9995% |
| 50% | 21-22 | 21-22 | 21-22 | 21-22 | 42-43 | 50.0000% |

## 主结果

以下是 batch size 512 的结果。所有策略在两个剪枝率下均成功运行到 512；`capped` 表示达到本轮预设上限，不表示 512 是 OOM 前的真实上限。

| Pruning | Strategy | Requests/s | Input tokens/s | p95 batch latency (s) | Peak (GiB/GPU) | KV (GiB/GPU) | Non-KV peak (GiB/GPU) |
|---|---|---:|---:|---:|---:|---:|---:|
| 30% | padded | 77.03 | 22,966 | 7.230 | 92.38 | 58.13 | 34.25 |
| 30% | multi-kernel | 74.44 | 22,195 | 7.636 | 91.15 | 50.38 | 40.77 |
| 30% | cross-layer | 77.83 | 23,206 | 7.108 | 92.40 | 62.15 | 30.25 |
| 50% | padded | 75.96 | 22,648 | 7.222 | 92.37 | 58.13 | 34.24 |
| 50% | multi-kernel | 74.17 | 22,113 | 7.641 | 91.19 | 55.55 | 35.64 |
| 50% | cross-layer | 77.06 | 22,975 | 7.113 | 92.43 | 64.84 | 27.59 |

batch size 512 时：

- 30%：cross-layer 相比 padded 吞吐提高 1.04%，Non-KV 峰值减少 4.00 GiB/GPU；相比 multi-kernel 吞吐提高 4.56%，Non-KV 峰值减少 10.52 GiB/GPU。
- 50%：cross-layer 相比 padded 吞吐提高 1.44%，Non-KV 峰值减少 6.65 GiB/GPU；相比 multi-kernel 吞吐提高 3.90%，Non-KV 峰值减少 8.05 GiB/GPU。

## 推荐画法

推荐使用 2 x 2 图，两列分别对应 30% 和 50% 剪枝率：

- 上排：X 轴 `batch_size`，Y 轴 `requests_per_second`
- 下排：X 轴 `batch_size`，Y 轴 `max_non_kv_peak_memory_mib / 1024`（GiB/GPU）
- 三条曲线：`padded`、`multi_kernel`、`cross_layer`
- 吞吐补充图可将 Y 轴替换为 `input_tokens_per_second`

主内存指标使用四卡最大 Non-KV 峰值，因为最重 rank 决定可运行上限。CSV 已提供 `max_non_kv_peak_memory_mib`，其计算为：

```text
Non-KV peak = max_peak_memory_mib - gpu_kv_cache_gib * 1024
```

Non-KV 包含模型权重、CUDA context、视觉 encoder cache、通信与 MoE workspace、allocator cache 和激活，不应标为纯“权重+激活”。vLLM 会把 90% 显存预算中的剩余空间分配给 KV cache，因此三种策略的总峰值接近是正常现象；省下的 Non-KV 显存体现为更大的 KV capacity。

## 复现命令

先生成 balanced plan（50% 时把两个 `30` 和 `0.30` 分别改为 `50` 和 `0.50`）：

```bash
MODEL_PATH=/path/to/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
python scripts/build_qwen_weight_proxy_ep4_plan.py \
  --model-path "$MODEL_PATH" \
  --output artifacts/efficiency_figure/qwen3_gqa_ep4/weight_proxy_ep4_30_balanced.pt \
  --prune-ratio 0.30 \
  --balance-tier-counts
```

再运行四卡 sweep：

```bash
EP4_PLAN=artifacts/efficiency_figure/qwen3_gqa_ep4/weight_proxy_ep4_30_balanced.pt \
OUTPUT_ROOT=artifacts/efficiency_figure/qwen3_gqa_ep4/batch_sweep_balanced_prefill/prune_30 \
START_BATCH_SIZE=8 MAX_BATCH_SIZE=512 \
MEASURED_BATCHES=4 WARMUP_BATCHES=1 \
STRATEGIES=padded,multi_kernel,cross_layer \
PREFILL_TASK=gqa_prefill PREFILL_MAX_NEW_TOKENS=1 \
GPU_MEMORY_UTILIZATION=0.90 \
bash scripts/run_qwen_ep4_batch_sweep.sh
```

## 限制

- 当前每个点只有 4 个 measured batch。p95 可用于趋势图；若作为高精度尾延迟结论，应增加重复次数。
- `activation_workspace_delta_*` 从 warmup 后基线计算。allocator 可能已在 warmup 缓存 workspace，因此不能将该列视为完整激活显存。
- `rank0_model_loading_gib` 只用于诊断，不应替代四卡最大 Non-KV 峰值。
