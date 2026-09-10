# 混合数据校准与双掩码计分

这套流程把“选哪些样本”和“在固定样本上收集分数”分成两个阶段。最终样本及其来源、原始索引、样本 ID、token 数和降维坐标都冻结在 JSON manifest 中，单数据源与混合数据实验只要使用相同的最终样本数和每样本计分 token 数，就可以公平比较。

## 第一阶段：生成混合校准集

默认候选源为 GQA、COCO2017Cap、M4-Instruct 和 Video-MMMU。候选池在数据源之间近似均分；某个源容量不足时，剩余额度会自动分给其他源。

每个候选样本先经过模型 processor，过滤掉不足 `SCORE_TOKENS_PER_SAMPLE` 的短样本。随后在指定模型层上分别对完整有效序列的文本 token 和视觉 token 求 hidden-state centroid。所有文本和视觉 centroid 共同经过标准化、PCA 和同一个 t-SNE，样本坐标定义为：

```text
p_i = (z_i_text + z_i_visual) / 2
```

最后从全局中心附近的样本开始做 greedy farthest-point sampling（FPS）。这会同时覆盖中心区域、边界区域和不同数据源的特征空间，而不是只按数据集配额抽样。默认要求最终结果至少包含每个请求的数据源。

```bash
MODEL_PATH=moonshotai/Kimi-VL-A3B-Instruct \
OUTPUT_MANIFEST=storage/calibration_manifests/kimi-mixed-512.json \
CANDIDATE_POOL_SIZE=4096 \
NUM_SAMPLES=512 \
SCORE_TOKENS_PER_SAMPLE=2048 \
FEATURE_BATCH_SIZE=2 \
bash scripts/prepare_mixed_calibration.sh
```

## 第二阶段：收集分数

```bash
SELECTION_MANIFEST=storage/calibration_manifests/kimi-mixed-512.json \
BATCH_SIZE=16 \
SECOND_ORDER_CHUNK_SIZE=auto \
AGGREGATION=mean \
bash scripts/run_collect_scores.sh
```

manifest 模式会严格按 `selection_rank` 重建相同样本；样本数和每样本计分 token 数以 JSON 为准。`scores.pt` metadata 会保存 manifest 路径、SHA256、来源统计和计分配置。

## 为什么需要两个 token mask

固定 2048 个 token 不能直接理解为“取序列前 2048 个”。如果视觉 token 在前、文本 token 在后，前缀截断会人为增加视觉路由事件，进而把专家误判为视觉偏好。

实现中使用两个不同用途的 mask：

- `affinity_mask` 是 processor 输出中的完整有效序列，用于累计每个专家的文本/视觉路由次数。只要序列没有超过模型本身的最大上下文，这里不做 2048-token 截断。
- `score_mask` 对每个样本恰好选择 2048 个 token，用于 layer loss、activation、gradient、saliency、channel score 和 expert Hessian。视觉/文本配额按该样本原始模态 token 数成比例分配，再分别在各模态的完整位置范围内等距抽取。

因此，固定 token 预算只控制不同实验的计分计算量，不改变专家模态偏好的曝光范围。模型前向仍使用完整输入，以免破坏视觉 placeholder 与视觉特征的对齐关系。

## 模态偏好归一化

每层先分别计算专家在两种模态中的路由率：

```text
r_text[e]   = routed_text[e]   / sum_e routed_text[e]
r_visual[e] = routed_visual[e] / sum_e routed_visual[e]
affinity[e] = (r_visual[e] - r_text[e]) / (r_visual[e] + r_text[e] + eps)
```

这样即使校准集中视觉 token 总数远多于文本 token，只要某个专家在两种模态中接收的路由份额相同，其 affinity 仍为 0。`+1` 表示偏视觉，`-1` 表示偏文本。
