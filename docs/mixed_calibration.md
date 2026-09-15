# 混合数据校准与双掩码计分

这套流程把“选哪些样本”和“在固定样本上收集分数”分成两个阶段。最终样本及其来源、原始索引、样本 ID、token 数、每样本计分配额和降维坐标都冻结在 JSON manifest 中。后续 score、Hessian probe 和 beta sweep 都读取同一个 manifest，因此使用完全相同的样本和 token mask。

## 第一阶段：生成混合校准集

默认候选源为 GQA、COCO2017Cap、M4-Instruct 和 Video-MMMU。候选池和最终 512 条样本都在数据源之间近似均分；某个源的合格样本不足时，剩余额度才会自动分给其他源。

每个候选样本先经过模型 processor，只过滤不足 `MIN_SAMPLE_TOKENS` 的特别短样本。无法解码的 VideoMMMU 视频会记录为 `video_decode_error` 并跳过。随后在指定模型层上分别对完整有效序列的文本 token 和视觉 token 求 hidden-state centroid。所有文本和视觉 centroid 共同经过标准化、PCA 和同一个 t-SNE，样本坐标定义为：

```text
p_i = (z_i_text + z_i_visual) / 2
```

最后在每个数据源配额约束下做 greedy farthest-point sampling（FPS），兼顾四源样本数量和特征空间覆盖。入选样本按可用 token 容量做 deterministic water-filling：默认目标总量为 `512 * 512 = 262144`，短样本可以少于 512，缺口由长样本补齐，不设置单样本 2048-token 上限。

```bash
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct \
OUTPUT_MANIFEST=storage/calibration_manifests/qwen3-mixed-512.json \
CANDIDATE_POOL_SIZE=4096 \
NUM_SAMPLES=512 \
TOTAL_SCORE_TOKENS=262144 \
MIN_SAMPLE_TOKENS=64 \
FEATURE_BATCH_SIZE=2 \
bash scripts/prepare_mixed_calibration.sh
```

For model sharding across four visible GPUs, explicitly request Accelerate's
automatic device map. Merely exposing four GPUs is insufficient because the
default loader target is `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DEVICE_MAP=balanced \
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct \
OUTPUT_MANIFEST=storage/calibration_manifests/qwen3-mixed-512.json \
CANDIDATE_POOL_SIZE=4096 NUM_SAMPLES=512 \
TOTAL_SCORE_TOKENS=262144 MIN_SAMPLE_TOKENS=64 \
bash scripts/prepare_mixed_calibration.sh
```

This is model parallelism: model weights are balanced across the four GPUs.
Candidate auditing remains sequential, and feature extraction still uses one
process rather than four-way data parallelism. With the default
`FEATURE_LAYER=0`, forward extraction stops after the first decoder block, so
later-layer shards may hold memory without doing useful compute.

## 第二阶段：收集分数

```bash
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct \
SELECTION_MANIFEST=storage/calibration_manifests/qwen3-mixed-512.json \
BATCH_SIZE=16 \
SECOND_ORDER_CHUNK_SIZE=auto \
AGGREGATION=mean \
bash scripts/run_collect_scores.sh
```

四卡按层并行收集（每张卡加载一份完整模型）使用：

```bash
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct \
SELECTION_MANIFEST=storage/calibration_manifests/qwen3-mixed-512.json \
OUTPUT_DIR=storage/scores/qwen3-mixed-512 \
GPUS=0,1,2,3 \
BATCH_SIZE=16 \
SECOND_ORDER_CHUNK_SIZE=auto \
AGGREGATION=mean \
HESSIAN_PROBE_LAYER=0 \
bash scripts/run_collect_scores_4gpu.sh
```

manifest 模式会严格按 `selection_rank` 重建相同样本；总 token 预算和每条样本的 `score_token_count` 以 JSON 为准。`scores.pt` metadata 会保存 manifest 路径、SHA256、来源统计和计分配置。第 0 层完成后，`hessian_probe_L0.pt` 会立即写到 `OUTPUT_DIR`，其余层可继续运行。

## 为什么需要两个 token mask

固定总 token 预算不能直接理解为“取每条序列的前若干 token”。如果视觉 token 在前、文本 token 在后，前缀截断会人为增加视觉路由事件，进而把专家误判为视觉偏好。

实现中使用两个不同用途的 mask：

- `affinity_mask` 是 processor 输出中的完整有效序列，用于累计每个专家的文本/视觉路由次数。只要序列没有超过模型本身的最大上下文，这里不做 2048-token 截断。
- `score_mask` 对每个样本选择 manifest 中记录的 `score_token_count`，所有样本合计恰好 262144 个 token，用于 layer loss、activation、gradient、saliency、channel score 和 expert Hessian。视觉/文本配额按该样本原始模态 token 数成比例分配，再分别在各模态的完整位置范围内等距抽取。

因此，固定 token 预算只控制不同实验的计分计算量，不改变专家模态偏好的曝光范围。模型前向仍使用完整输入，以免破坏视觉 placeholder 与视觉特征的对齐关系。

## 模态偏好归一化

每层先分别计算专家在两种模态中的路由率：

```text
r_text[e]   = routed_text[e]   / sum_e routed_text[e]
r_visual[e] = routed_visual[e] / sum_e routed_visual[e]
affinity[e] = (r_visual[e] - r_text[e]) / (r_visual[e] + r_text[e] + eps)
```

这样即使校准集中视觉 token 总数远多于文本 token，只要某个专家在两种模态中接收的路由份额相同，其 affinity 仍为 0。`+1` 表示偏视觉，`-1` 表示偏文本。
