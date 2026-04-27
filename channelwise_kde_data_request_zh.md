# Channel-wise KDE 数据采集需求说明

## 背景

当前的 `raw_stats` 文件中保存的是聚合后的统计量，主要包括：

- `channel_abs_sum[layer][modality]`：形状为 `(#experts, #channels)`
- `channel_count[layer][modality]`：形状为 `(#experts,)`

利用这两个量，可以恢复每个 `(layer, expert, channel)` 的平均绝对激活值：

```python
mean_abs = channel_abs_sum / channel_count.unsqueeze(-1)
```

因此，当前数据已经足够支持：

- channel response 的折线图 / 柱状图
- expert-level 的整体统计图
- layer-level 的整体统计图

但是，如果左图希望绘制**真正的 channel-wise KDE**，那么当前数据还不够，因为 KDE 需要的是**样本级观测值**，而不是每个 channel 最后只保留一个全局均值。

## 当前缺少什么

如果要对左图中的某个 `(layer, expert, channel)` 做真实的 KDE，需要拿到该 channel 在不同样本上的一组绝对激活值，例如：

```python
[sample_1_abs, sample_2_abs, ..., sample_N_abs]
```

而不是只有：

```python
mean_abs
```

也就是说，当前缺少的是：

- 每个 sample 上的 channel 级别绝对激活值

而不是：

- 所有 sample 聚合之后的总和
- 所有 sample 聚合之后的均值

## 建议额外保存的数据

### 方案 A：保存完整的 sample-level channel response

对于每个 `layer` 和 `modality`，建议保存：

```python
channel_abs_samples[layer][modality]
# shape: (num_samples, num_experts, num_channels)
```

含义是：

- 对每个 sample
- 对每个 expert
- 对每个 channel
- 保存该 sample 上该 modality 对应的平均绝对激活值

这个格式最完整，后续可视化和分析最灵活。

### 方案 B：只保存左图需要的少量 `(layer, expert)`，减少存储

如果担心存储太大，可以只保存左图中会用到的几个 `(layer, expert)` 对。

当前建议保留的 pair：

- Kimi: `(5, 5)`、`(10, 57)`、`(25, 60)`
- Qwen3: `(5, 48)`、`(25, 115)`、`(45, 102)`

此时建议保存：

```python
selected_channel_abs_samples[(layer, expert)][modality]
# shape: (num_samples, num_channels)
```

这个版本已经足够用于左图的 channel-wise KDE，而且数据量会小很多。

## 采集逻辑建议

1. 加载模型和数据集。
2. 按 sample 做 forward。
3. 在目标 MoE 层拿到 expert 输出激活。
4. 区分 `text` token 和 `visual` token。
5. 对每个 modality：
   - 找到被路由到某个 expert 的 token
   - 取该 expert 输出激活的绝对值 `abs`
   - 在当前 sample 内，对这些 token 沿 token 维做平均
   - 得到一个 `(num_experts, num_channels)` 的 sample-level 张量
6. 将所有 sample 的结果累积保存下来。

## 这里需要保存的值是什么

需要保存的是：

- **sample-level mean absolute activation**

也就是：

- 每个 sample
- 每个 layer
- 每个 expert
- 每个 channel
- 在指定 modality 下的平均绝对激活值

不需要的是：

- 原始 signed activation
- 只保留聚合后的 sum
- 只保留跨样本平均后的最终均值

因为 KDE 需要的是一组样本观测值，而不是最终汇总值。

## 建议输出格式

### 完整版

```python
{
    "layers": [...],
    "channel_abs_samples": {
        layer_id: {
            "text": torch.Tensor[num_samples, num_experts, num_channels],
            "visual": torch.Tensor[num_samples, num_experts, num_channels],
        }
    },
    "model_name_or_path": ...,
    "dataset": ...,
    "num_samples": ...,
}
```

### 精简版

```python
{
    "selected_pairs": [(layer, expert), ...],
    "selected_channel_abs_samples": {
        (layer, expert): {
            "text": torch.Tensor[num_samples, num_channels],
            "visual": torch.Tensor[num_samples, num_channels],
        }
    },
    "model_name_or_path": ...,
    "dataset": ...,
    "num_samples": ...,
}
```

## 和当前三张图的关系

- 中图 `expert-level`：当前 `raw_stats` 已经足够
- 右图 `layer-level`：当前 `raw_stats` 已经足够
- 左图：
  - 如果只是画 channel response 曲线，当前 `raw_stats` 已经足够
  - 如果要画真正的 channel-wise KDE，还需要额外保存 sample-level 的绝对激活值

## 一句话请求

请额外保存**每个 sample 上的 per-layer / per-expert / per-channel mean absolute activation**；如果全量保存开销过大，至少请保存左图选中的若干 `(layer, expert)` 对应的 `shape = (num_samples, num_channels)` 的 text / visual 张量，这样后续才能对每个 channel 在样本维上做真实的 KDE。
