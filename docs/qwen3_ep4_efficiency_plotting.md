# Qwen3-VL EP4 效率实验画图说明

## 实验范围

当前已完成并可用于画图的是 30% 和 50% 剪枝实验：

- 模型：`Qwen/Qwen3-VL-30B-A3B-Instruct`
- 任务：GQA
- 硬件：4 x NVIDIA H20
- 后端：vLLM 0.11.2，CUDA fused-MoE，EP=4
- 剪枝比例：30%、50%
- 三种策略：`padded`、`multi_kernel`、`cross_layer`
- batch size：8、16、32、64、128、256、512
- 每个点：1 个 warmup batch，4 个 measured batch
- `gpu_memory_utilization=0.90`，KV cache 由 vLLM 自动分配

数据目录：

```text
artifacts/efficiency_figure/qwen3_gqa_ep4/batch_sweep/prune_30/
artifacts/efficiency_figure/qwen3_gqa_ep4/batch_sweep/prune_50/
```

每个剪枝比例目录都包含以下主要文件：

- `throughput_memory_curve.csv`：三种策略的完整 batch-size 曲线，共 21 个点
- `best_by_strategy.csv`：每种策略在已测试范围内的最佳吞吐点
- `sweep_status.tsv`：每个点的完成状态和搜索终止状态
- `sweep_summary.json`：CSV 内容的结构化汇总

## 推荐主图

建议使用两面板图，而不是将所有信息压在同一个双 Y 轴中。

### (a) Saturation throughput vs. batch size

- X 轴：`batch_size`，使用以 2 为底的离散刻度
- Y 轴：`requests_per_second`
- 三条曲线：`padded`、`multi_kernel`、`cross_layer`
- 可在附图中将 Y 轴替换成 `total_tokens_per_second`

GQA 的答案很短，`output_tokens_per_second` 容易受少量生成 token 波动影响；主图使用 req/s 更能反映端到端多模态请求处理能力。`total_tokens_per_second` 主要由输入 token 决定，适合作为补充指标。

### (b) Non-KV peak memory vs. batch size

- X 轴：`batch_size`
- Y 轴：`max_non_kv_peak_memory_mib / 1024`，单位 GiB/GPU
- 三条曲线与 (a) 使用相同颜色和 marker
- 使用四卡中的最大值，而不是四卡均值，因为最重 rank 决定可运行上限

该指标按每个实验点实际分配的 KV cache 单独扣除：

```text
Non-KV peak = max_peak_memory_mib / 1024 - gpu_kv_cache_gib
```

CSV 已直接提供 `max_non_kv_peak_memory_mib`，画图时不需要再次计算。这里的 Non-KV 包含模型权重、CUDA context、视觉 encoder cache、通信与 MoE workspace、allocator cache 和激活；不要将它标成纯“权重+激活”。

## 推荐内存分解图

为了说明三种策略的总峰值为何接近，建议再画一个 batch 512 的堆叠柱状图：

- 下段：`max_non_kv_peak_memory_mib / 1024`
- 上段：`gpu_kv_cache_gib`
- 柱顶：`max_peak_memory_mib / 1024`
- 柱旁标注：`gpu_kv_cache_tokens`

vLLM 会将 90% 显存预算中剩余的空间自动分配给 KV cache。因此，总显存接近不代表三种策略的模型与 workspace 开销接近；更低的 Non-KV 占用会体现为更大的 KV capacity。

batch 512 的 50% 剪枝结果如下：

| Strategy | req/s | p95 batch latency (s) | Peak (GiB/GPU) | KV (GiB/GPU) | Non-KV peak (GiB/GPU) | KV capacity (tokens) |
|---|---:|---:|---:|---:|---:|---:|
| padded | 65.92 | 8.42 | 92.38 | 58.13 | 34.25 | 2,539,520 |
| multi-kernel | 62.83 | 8.90 | 91.09 | 55.49 | 35.60 | 2,424,592 |
| cross-layer | 68.44 | 8.09 | 92.40 | 64.83 | 27.57 | 2,832,160 |

在该点，cross-layer 相比 padded 的 req/s 提高约 3.8%，Non-KV 峰值降低约 6.68 GiB/GPU；相比 multi-kernel 的 req/s 提高约 8.9%，Non-KV 峰值降低约 8.03 GiB/GPU。

batch 512 的 30% 剪枝结果如下：

| Strategy | req/s | p95 batch latency (s) | Peak (GiB/GPU) | KV (GiB/GPU) | Non-KV peak (GiB/GPU) | KV capacity (tokens) |
|---|---:|---:|---:|---:|---:|---:|
| padded | 69.54 | 7.80 | 92.38 | 58.13 | 34.25 | 2,539,520 |
| multi-kernel | 64.04 | 8.56 | 91.08 | 50.30 | 40.78 | 2,197,552 |
| cross-layer | 68.62 | 7.86 | 92.46 | 62.16 | 30.30 | 2,715,680 |

30% 剪枝下，batch 512 的 padded 吞吐比 cross-layer 高约 1.3%，但 cross-layer 的 Non-KV 峰值低约 3.95 GiB/GPU。画图时应保留这一结果，不把 50% 剪枝下的速度结论外推到 30%。

## 可选单图方案

如果版面只能容纳一张图，可使用双 Y 轴：

- X 轴：`batch_size`
- 左 Y 轴：`requests_per_second`，实线
- 右 Y 轴：`max_non_kv_peak_memory_mib / 1024`，虚线
- 颜色区分策略，线型区分指标

不要用时间作为当前主图 X 轴。时间序列适合展示单次运行的瞬时显存与吞吐，但当前论文问题是不同并发下的饱和效率，batch size 曲线更直接，也更容易公平比较。

## 状态与限制

- 两个剪枝比例下的三种策略都在 batch 512 成功，`search_terminal_status=capped`。
- `capped` 表示到达本轮预设上限和 GQA 样本预算，并不表示 batch 512 是 OOM 前的真实最大稳定 batch。
- `activation_workspace_delta_*` 是 warmup 后基线到正式测量峰值的增量。CUDA allocator 已可能在 warmup 中缓存 workspace，因此该列可能只有数 MiB，不能作为完整激活显存使用。
- `rank0_model_loading_gib` 是 vLLM rank 0 日志中的加载阶段指标，可作为诊断项，不应替代四卡最大 Non-KV 峰值作为主内存指标。
- 当前每个点只有 4 个 measured batch。p95 适合展示趋势；若作为论文中的高精度尾延迟结论，应增加重复次数。

## 样式建议

- `padded`：灰色圆点
- `multi_kernel`：橙色三角
- `cross_layer`：蓝绿色方块
- batch size 使用明确的离散刻度，不使用连续插值
- 吞吐图从 0 起始；内存图可从 0 起始，或在图注中明确截断范围
- 图注中写清 `4 x H20`、`vLLM fused-MoE`、`GQA`、剪枝比例和 `gpu_memory_utilization=0.90`
