
```
在线版 synthetic hidden distillation, 不依赖预先生成的 teacher cache 文件.

整体分两阶段, 但都在一次进程里顺序完成:

1) 初始化阶段 (filling synthetic pool)
   - 从多数据集 round-robin 流中反复取 raw batch, 前向教师模型, 在指定 decoder 层取 block 输出,
     再按 ``compression_mode`` 压到 ``compressed_length``, 得到与离线 cache 同构的
     ``(B, L, D)`` teacher hidden, 以及 ``modality_labels`` / ``position_ids``.
   - 连续累积直到凑满 ``synthetic_size`` 条序列, 作为合成集在参数空间里的初值模板.
   - 初始化阶段不再额外计算或缓存 ``teacher_layer+1`` 的 next-block 输出.

2) 蒸馏阶段 (optimization loop)
   - 仍从同一数据流在线抽 ``teacher_batch_size`` 条 teacher 序列, 与当前可学习的 synthetic
     表示算分布损失; 教师侧每步都是新样本, 不再读固定 ``.pt`` shard.
   - 可学习参数是 ``_build_synthetic_banks`` 返回的 ``ParameterDict`` (按模态分 bank),
     通过 ``_assemble_synthetic_hidden`` 拼出与模板同形状的 ``synthetic_hidden``;
     ``position_ids``, ``modality_labels`` 在在线脚本里冻结为初始化时的值.
   - 若 ``lambda_block > 0``, 训练时对 teacher / synthetic 两边的 compressed hidden
     现场前向 ``teacher_layer+1`` 并在 next-block 空间上做分布式对齐.
   - 优化器为 Adam, 对 ``bank_params`` 做 ``backward`` + ``step``.

与 ``distill_synthetic_hidden.py`` 的关系: 损失定义, 加权, 消融, diversity warmup 等逻辑复用该模块;
本文件只负责数据管线 (流式 teacher) 与在线训练期 next-block 监督.

Loss 项 (详见 ``_compute_losses``): 按模态分组的 MMD / cov / mean / var, 以及全局 synthetic token
上的 diversity (div); 可选 ``block_rel_l2`` 在 next-block 输出空间上约束 teacher / synthetic 分布.
```


# Online Distillation 说明

本文说明仓库中 online synthetic hidden distillation 的实际做法，主要参考以下两个入口文件：

- `scripts/data_distill/run_distill_compact_hidden_online.sh`
- `src/calibration/representation_distill/distill_synthetic_hidden_online.py`

同时补充它们直接依赖的几个核心模块：

- `src/calibration/representation_distill/distill_synthetic_hidden.py`
- `src/calibration/representation_distill/common.py`
- `src/calibration/representation_distill/runtime/forward_from_hidden.py`
- `src/calibration/representation_distill/build_teacher_hidden_cache.py`

---

## 1. 这个 online distillation 在做什么

目标是：从一个多模态教师模型中，蒸馏出一小份可学习的、压缩后的 hidden representation 集合 `synthetic_hidden`，让它在统计分布上逼近真实教师 hidden。

和普通“蒸馏 logits”不同，这里蒸馏的是某一层 decoder block 的 hidden states。

更具体地说：

1. 从教师模型指定层 `teacher_layer` 抽取真实样本的 hidden。
2. 把每条样本的长序列 hidden 压缩到固定长度 `compressed_length`。
3. 用这批压缩后的教师 hidden 初始化一个可训练的 synthetic hidden 集合。
4. 训练时不断重新从教师数据流在线抽样，再和 synthetic hidden 做分布匹配。
5. 最后把训练好的 synthetic hidden 保存成一个 `.pt` 文件，后续可被校准/剪枝/评测流程复用。

这里的 “online” 指的是：

- 训练时教师 hidden 不是提前全量缓存到磁盘再读。
- 每一步训练都在线走教师模型前向，实时得到新的 teacher hidden batch。
- 因而 teacher 侧监督是“流式”的，而不是固定缓存的。

---

## 2. 与 offline distillation 的核心区别

### Offline 版本

离线版本 `distill_synthetic_hidden.py` 的流程大致是：

1. 先用 `build_teacher_hidden_cache.py` 把教师 hidden 全部抽出来落盘。
2. 训练时从 `teacher_cache` 中采样。
3. 如果需要 next-block 监督，可以提前把 `next_block_cache` 一并缓存好。

优点：

- 训练期间不需要反复跑教师模型。
- 吞吐更稳定。
- 可复现实验更方便。

代价：

- 需要预生成缓存。
- 磁盘占用大。
- 数据和监督分布固定。

### Online 版本

在线版本 `distill_synthetic_hidden_online.py`：

1. 不依赖预先生成的 `teacher_cache.pt`。
2. 直接从原始数据样本开始构造 teacher batch。
3. 每次训练 step 都在线前向教师模型并压缩 hidden。
4. 若启用 `lambda_block > 0`，还会在训练期现场计算 next-block 输出。

优点：

- 不需要提前准备 teacher cache。
- 每步监督来自新抽取的 teacher batch，分布覆盖更“活”。
- 对 next-block 约束不需要额外缓存文件。

代价：

- 训练更贵，因为每步都要走教师模型前向。
- 对显存、算力、数据处理吞吐要求更高。
- 训练过程更依赖运行时环境稳定性。

---

## 3. 启动脚本 `run_distill_compact_hidden_online.sh` 做了什么

这个 shell 脚本本质上是一个参数装配器。它负责：

1. 选择 GPU。
2. 设置 `PYTHONPATH`、`HF_HOME`、输出路径等环境变量。
3. 组装在线蒸馏脚本的命令行参数。
4. 调用：

```bash
python -m src.calibration.representation_distill.distill_synthetic_hidden_online
```

5. 训练完成后把输出目录软链接到 `online-distilled-latest`。

### 3.1 入口脚本暴露的关键参数

#### 教师 hidden 抽取相关

- `TEACHER_LAYER`
  - 从教师模型第几层 decoder block 抽取 hidden。
- `COMPRESSED_LENGTH`
  - 每条样本压缩后保留多少 token。
- `COMPRESSION_MODE`
  - 压缩方式，支持 `pool`、`sample`、`attention_weighted`。
- `MODALITY_AWARE_COMPRESSION`
  - 是否按 text/image/video 分开压缩再拼接。
- `ATTN_TEMPERATURE`
  - `attention_weighted` 模式下的采样温度。

#### 教师数据流相关

- `TEACHER_DATASETS`
  - 教师样本来自哪些数据集。
- `SAMPLES_PER_DATASET`
  - 每个数据集采样多少条原始样本进入流。
- `TEACHER_BATCH_SIZE`
  - 每个训练 step 取多少条压缩后的 teacher 序列做监督。
- `TOKEN_PER_SAMPLE`
  - 期望的原始样本 token 长度筛选阈值。

#### synthetic 集合与优化相关

- `SYNTHETIC_SIZE`
  - synthetic 集合总共包含多少条压缩序列。
- `SYNTHETIC_BATCH_SIZE`
  - 每步从 synthetic 集合中抽多少条参与训练。
- `TRAIN_STEPS`
  - 总训练步数。
- `LR`
  - Adam 学习率。
- `INIT_STD`
  - 初始化时给 synthetic bank 加的高斯噪声。

#### loss 权重相关

- `LAMBDA_MMD`
- `LAMBDA_COV`
- `LAMBDA_DIV`
- `LAMBDA_MEAN`
- `LAMBDA_VAR`
- `LAMBDA_BLOCK`

#### 训练稳定性与监控

- `USE_EMA_NORMALIZED_LOSSES`
  - 是否用 EMA 对各损失做归一化。
- `LOSS_EMA_DECAY`
  - EMA 衰减系数。
- `DIV_WARMUP_STEPS`
  - diversity loss 的 warmup 步数。
- `MMD_SUBSAMPLE`
  - 计算 MMD 时最多抽多少 token。
- `LOG_INTERVAL`
  - 诊断日志的记录频率。
- `CHECKPOINT_INTERVAL`
  - 中间 checkpoint 的保存频率。
- `WANDB_*`
  - W&B 配置。

#### 消融参数

- `DIVERSITY_ABLATION`
  - `full` 或 `no_div`
- `DISTRIBUTION_ABLATION`
  - `full`、`moment_only`、`mmd_only`

---

## 4. 整体流程概览

online distillation 在一个进程里顺序执行两阶段：

1. 初始化阶段：先填满 synthetic pool。
2. 优化阶段：在线教师流 + synthetic 参数优化。

可画成下面的逻辑：

```text
raw multimodal samples
    -> prepare_raw_batch_inputs
    -> teacher model forward
    -> extract block output at teacher_layer
    -> compress hidden states to fixed length
    -> fill synthetic initialization pool
    -> build modality-specific trainable banks
    -> optimization loop:
         streamed teacher batch
         vs
         sampled synthetic batch
         -> distribution losses
         -> optional next-block loss
         -> Adam update
    -> save distilled synthetic_hidden payload
```

---

## 5. 阶段一：如何构造 online teacher 数据流

### 5.1 原始样本来源

脚本会调用 `dump_original_data(...)` 把原始多模态样本组织出来。样本来自 `teacher_datasets` 指定的数据集，例如：

- `gqa`
- `coco`
- `m4_instruct`

这些样本不是直接 tensor，而是原始 sample dict，里面可能包含：

- 文本
- 图片
- 视频帧
- 数据集名字
- 样本索引

### 5.2 按数据集分组与筛选

`_filter_and_group_samples(...)` 会把样本按数据集名分组。

然后对每个数据集调用 `_select_teacher_samples(...)`，这个逻辑和离线版保持一致，主要作用是：

1. 统计每个样本在 processor/tokenizer 后的 token 数。
2. 优先保留 token 数不少于 `token_per_sample` 的样本。
3. 如果不够，再从较短样本里补齐。

这一步的目的，是尽量保证 teacher 侧输入长度足够，避免压缩前的序列过短导致信息量不足。

### 5.3 Round-robin 流

`_RoundRobinDatasetStream` 是 online 版本很关键的结构。

它的行为是：

1. 在多个 teacher 数据集之间轮流取 batch。
2. 每个数据集内部维护一个 pointer。
3. 指针走到末尾后，对该数据集样本重新 shuffle，再继续取。

因此训练过程中 teacher 样本不是只来自单一数据集，也不是一次性固定顺序，而是：

- 跨数据集 round-robin
- 数据集内部循环重排

这样可以让 teacher 监督分布更平衡地覆盖多个来源。

---

## 6. 阶段一：如何从 teacher 在线抽 hidden

每次需要一批 teacher hidden 时，脚本会调用 `_extract_teacher_chunk(...)`。

这条链路基本是：

```text
raw_batch
  -> prepare_raw_batch_inputs
  -> move_inputs_to_model_device
  -> build_compression_token_masks (optional)
  -> extract_block_output
  -> compress_hidden_states
  -> return hidden / modality_labels / position_ids
```

### 6.1 `prepare_raw_batch_inputs`

这一步会：

1. 把原始样本转成 chat message 格式。
2. 调用多模态 processor 生成模型输入。
3. 得到 `input_ids`、`attention_mask`，以及图像相关输入。

### 6.2 `extract_block_output`

这是 teacher hidden 抽取的核心。

它不是拿最终输出，而是：

1. 在目标 decoder block 上注册 forward hook。
2. 运行模型前向。
3. 一旦目标层输出被捕获，就通过抛 `EarlyStopForward` 提前终止后续前向。

好处是：

- 只跑到所需层，节省开销。
- 直接得到 `teacher_layer` 的 block 输出 hidden。

如果 `compression_mode == attention_weighted`，它还会在 self-attention 子层上注册 hook，提取每个 token 的 attention importance，供后续压缩采样使用。

### 6.3 `compress_hidden_states`

teacher block 输出通常序列很长，不能直接拿来蒸馏，所以必须压缩到固定长度 `compressed_length`。

支持三种模式：

#### `pool`

- 旧的 mean pooling 逻辑。
- 把长序列压成固定长度。

#### `sample`

- 均匀随机采样真实 token。
- 保留离散 token 级 hidden，而不是做平均。

#### `attention_weighted`

- 根据 attention importance 对 token 采样。
- 更倾向选择信息量高的 token。

### 6.4 模态感知压缩

如果开启 `modality_aware_compression`：

1. 先根据 `input_ids` 和媒体 token 识别 text/image/video token。
2. 为各模态分配压缩长度预算。
3. 对 text/image/video 各自压缩。
4. 再把结果拼起来，并恢复位置顺序。

输出除了压缩后的 hidden，还有：

- `modality_labels`
  - 每个压缩 token 属于 text/image/video 中的哪一类。
- `position_ids`
  - 压缩 token 在原始序列中的位置编号。

这两个东西后面训练会继续用到。

---

## 7. 阶段一：如何填满 synthetic 初始化池

online 版本不会随机初始化一个完全无结构的 synthetic 集合，而是先从真实 teacher hidden 流里拿一批样本作为初始模板。

这一步由 `_collect_stream_examples(...)` 完成：

1. 不断调用 `stream.next_batch()` 取原始 teacher batch。
2. 对每个 batch 调用 `_extract_teacher_chunk(...)`。
3. 把压缩后的 `hidden`、`labels`、`position_ids` 累积起来。
4. 直到样本条数达到 `synthetic_size`。

最终得到：

- `init_hidden_cpu`
- `synth_labels_cpu`
- `init_position_ids_cpu`

它们代表 synthetic 集合的初始版本，但此时还不是参数化形式。

这样做的含义是：

- synthetic 不是从纯噪声开始学。
- 它一开始就是“真实 teacher hidden 的一份压缩子集”。
- 训练只是继续把这份子集变成更能代表整体分布的 synthetic representation。

---

## 8. synthetic hidden 是怎么参数化的

这部分复用 `distill_synthetic_hidden.py` 中的逻辑，核心函数是：

- `_build_synthetic_banks(...)`
- `_assemble_synthetic_hidden(...)`

### 8.1 为什么要做 bank 参数化

synthetic hidden 的形状是：

```text
(synthetic_size, compressed_length, hidden_size)
```

如果直接把整个张量当作一个大参数也能训，但这里做得更细：

1. 先把所有 token 展平成 `(N_tokens, hidden_size)`。
2. 按模态 label 分组。
3. 为每种模态单独建立一个 `nn.Parameter` bank。

比如会有：

- text bank
- image bank
- video bank

这样每个压缩 token 不是自己独立持有一份 metadata，而是通过：

- `template_labels`
- `template_bank_indices`

映射到某个 bank 的某一行。

### 8.2 `_build_synthetic_banks`

输入：

- 初始化 hidden
- 初始化 modality labels
- `init_std`

输出：

- `bank_params`
  - 一个 `ParameterDict`，每个模态一个参数矩阵。
- `flat_labels`
  - 每个 token 的模态标签。
- `flat_bank_indices`
  - 每个 token 在所属模态 bank 中对应哪一行。

如果 `init_std > 0`，会在初始 hidden 上加少量高斯噪声，打破完全复制教师样本的状态。

### 8.3 `_assemble_synthetic_hidden`

训练每一步真正参与 loss 计算的是一个当前版本的 synthetic batch。

这个 batch 不是单独存的，而是通过：

1. 查看 token 的模态标签。
2. 根据 `bank_indices` 去对应 bank 里取行。
3. 再 reshape 回 `(batch, compressed_length, hidden_size)`。

因此优化本质上是在更新 bank 中的向量。

### 8.4 哪些东西可训练，哪些是冻结的

在线版本里：

- 可训练：
  - `bank_params`

- 冻结：
  - `position_ids`
  - 初始化时得到的 `modality_labels`
  - synthetic 序列长度结构

也就是说，online distillation 学的是 token 表征值本身，不改 token 的模态布局和相对位置模板。

---

## 9. 阶段二：在线优化循环怎么跑

初始化池填满后，脚本进入真正训练阶段。

每个 step 的逻辑如下。

### 9.1 重新抽一个 teacher batch

每一步都重新调用 `_collect_stream_examples(...)`，收集 `teacher_batch_size` 条压缩 teacher hidden。

得到：

- `teacher_batch`
- `teacher_batch_labels`
- `teacher_position_ids`
- `teacher_attention_mask`

这里的关键点是：

- teacher batch 每一步都来自新的在线前向。
- 它不是固定缓存里的同一批样本。
- 所以优化目标是在追踪流式 teacher 分布。

### 9.2 从 synthetic 集合采样一个子 batch

如果 `synthetic_batch_size < synthetic_size`，则每步只抽 synthetic 集合的一部分，以降低显存和计算。

流程是：

1. 随机采样 synthetic 序列索引。
2. 取出对应的 label 模板和 bank 索引模板。
3. 用 `_assemble_synthetic_hidden(...)` 拼出当前 synthetic batch。

### 9.3 计算分布匹配损失

核心函数是 `_compute_losses(...)`。

它把 teacher 和 synthetic 展平成 token 集合，再按模态分组做统计匹配。

如果 modality labels 可用，则会分别对：

- text
- image
- video

求损失，然后对存在的模态组做平均。

如果某一步 labels 不可用或组内 token 太少，就退化成 pooled 全局计算。

---

## 10. loss 由哪些部分组成

### 10.1 MMD loss

`mmd`

使用多带宽 RBF kernel 的 MMD，衡量 teacher token 分布和 synthetic token 分布的整体差异。

特点：

- 不只比较均值或方差。
- 更关注整体分布形状。
- 计算较贵，所以会先做 token subsample。

### 10.2 Covariance loss

`cov`

比较 teacher 和 synthetic 的通道协方差矩阵：

- 先对 token 做中心化。
- 计算特征维协方差。
- 对两个协方差矩阵做 MSE。

它约束的是二阶结构。

### 10.3 Mean loss

`mean`

比较每个 hidden 维度上的均值。

### 10.4 Variance loss

`var`

比较每个 hidden 维度上的方差。

### 10.5 Diversity loss

`div`

只作用在 synthetic token 上。

做法是：

1. 对 synthetic token 做归一化。
2. 计算 token 间余弦相似度矩阵。
3. 取非对角元素平均值。

这个值越低，说明 synthetic token 之间越分散、越不塌缩。

它的作用是防止 synthetic hidden 全部收敛到少数模式。

### 10.6 Block-level loss

`block_rel_l2`

这是 online 版本里非常值得注意的一项可选监督。

当 `lambda_block > 0` 时：

1. 把当前 teacher compressed hidden 从 `teacher_layer + 1` 再前向一个 block。
2. 把当前 synthetic hidden 也从 `teacher_layer + 1` 再前向一个 block。
3. 比较二者 next-block 输出的 pooled mean/var 相对误差。

这个 loss 不是逐 token 硬对齐，而是对 next-block 空间中的整体统计做约束。

这样做的意义是：

- 不仅让 synthetic hidden 看起来像 teacher_layer 的分布。
- 还要求它经过下一层后，诱导出的表示统计也接近真实 teacher。

可以把它理解成一种“局部动力学一致性”约束。

---

## 11. loss 是怎么加权和稳定化的

总损失通过 `_compute_weighted_total_loss(...)` 得到。

### 11.1 基本形式

总损失近似为：

```text
lambda_mmd  * mmd
+ lambda_cov * cov
+ lambda_mean * mean
+ lambda_var * var
+ lambda_div * div_scale * div
+ lambda_block * block_rel_l2   # optional
```

### 11.2 Diversity warmup

如果 `div_warmup_steps > 0`，则：

- 前几步不会立刻满强度使用 diversity loss。
- `div_scale` 从 0 线性增到 1。

原因是训练初期别的 loss 可能很小，而 `div` 容易主导优化方向。

### 11.3 EMA-normalized losses

如果开启 `use_ema_normalized_losses`：

1. 每种 loss 都维护一个 EMA。
2. 当前 raw loss 会除以自己的 EMA。
3. 再乘对应权重。

这相当于一种自适应尺度归一化，缓解不同 loss 数值量级相差太大时的训练不平衡。

---

## 12. next-block 监督为什么 online 版本更自然

offline 版本如果想做 block-level 监督，通常要提前缓存 `next_block_cache`。

online 版本不需要提前缓存，原因是：

1. 教师模型本来就在训练过程中常驻可用。
2. 当前 step 已经拿到了 teacher compressed hidden。
3. 直接调用 `forward_from_hidden(...)` 即可从 `teacher_layer + 1` 再跑一个 block。

`forward_from_hidden(...)` 的作用是：

- 跳过 embedding 和前面层。
- 直接从给定 hidden 继续向后跑。
- 对 Kimi 和 Qwen3 分别实现。

因此 `lambda_block` 在 online 版本中的实现路径非常直接：

```text
teacher compressed hidden
    -> next block
synthetic hidden
    -> next block
compare pooled distribution statistics
```

它本质上是一种“在线局部 rollout 监督”。

---

## 13. 训练时记录哪些诊断信息

在 `log_interval` 或最后一步，脚本会计算 `_compute_diagnostics(...)`，包括：

- teacher token norm 的均值和方差
- synthetic token norm 的均值和方差
- 两者均值差
- centroid L2 距离
- synthetic token 两两余弦相似度的统计量
  - mean
  - p50
  - p90
  - max
- 按模态拆分的 centroid gap 和 norm gap

这些指标不是直接参与优化，而是用于判断：

- synthetic 是否塌缩
- teacher/synthetic 的尺度是否漂移
- 模态间是否对齐失衡

---

## 14. 消融开关怎么影响训练

`_apply_ablation_presets(...)` 会根据命名式消融参数改写实际 loss 权重。

### `diversity_ablation`

#### `full`

- 正常使用 diversity loss。

#### `no_div`

- 把 `lambda_div` 设为 0。
- 把 `div_warmup_steps` 设为 0。

### `distribution_ablation`

#### `full`

- 全部使用：MMD + cov + mean + var。

#### `moment_only`

- 把 `lambda_mmd = 0`。
- 只保留 moment 类损失。

#### `mmd_only`

- 把 `lambda_cov = 0`
- 把 `lambda_mean = 0`
- 把 `lambda_var = 0`
- 只保留 MMD。

---

## 15. checkpoint 和最终产物

### 15.1 中间 checkpoint

如果设置了 `checkpoint_interval`：

1. 脚本会定期把当前 `bank_params` 重新 assemble 成完整 synthetic hidden。
2. 保存为 `xxx-stepN.pt`。
3. 只保留最近少量 checkpoint，旧的会被删除。

### 15.2 最终输出 `.pt`

训练结束后，会把完整 synthetic hidden 保存到 `output_path`。

payload 主要包含：

- `synthetic_hidden`
- `attention_mask`
- `position_ids`
- `modality_labels`（若存在）
- `metadata`

metadata 中会记录：

- 方法名
- teacher 侧元信息
- synthetic 大小
- batch 配置
- 训练步数
- 学习率
- 各 loss 权重
- 消融设置
- final losses
- history
- 创建时间

这意味着最终文件不只是 tensor，还携带了足够多的训练上下文，方便后续追溯。

---

## 16. online 版本的几个关键设计判断

### 16.1 为什么先用真实 teacher hidden 初始化，而不是纯随机初始化

因为 synthetic 目标不是生成任意 hidden，而是近似教师 hidden 分布。

用真实 teacher hidden 做初始化有几个好处：

- 收敛更快
- 训练更稳定
- 更不容易落在完全错误的尺度区域

### 16.2 为什么把 position_ids 和 modality_labels 冻结

当前方法的目标是学习“表征值”，不是学习序列结构。

冻结后：

- 优化空间更小
- 训练更稳定
- 避免 synthetic 通过篡改位置结构取巧拟合统计量

### 16.3 为什么既要 MMD 又要 mean/var/cov

这三类约束关注的层面不同：

- `mean/var`
  - 一阶、二阶的边缘统计
- `cov`
  - 通道间相关结构
- `mmd`
  - 更整体的分布差异

把它们组合起来，相当于同时约束“局部矩信息”和“全局分布形状”。

### 16.4 为什么要加 diversity

如果没有 diversity，synthetic set 很容易学成很多彼此相似的 token 或序列，虽然某些矩统计能对上，但代表性很差。

diversity loss 的作用就是防止这种 collapse。

### 16.5 为什么 block loss 只比较 pooled 统计，而不是逐 token 精确对齐

因为 teacher batch 和 synthetic batch 并不是逐样本配对关系。

online 版本更关注：

- synthetic 整体是否能代表 teacher 分布

而不是：

- synthetic 第 i 个 token 必须对应 teacher 第 j 个 token

所以在 next-block 空间里比较 pooled mean/var 是更合理的分布级监督。

---

## 17. 一个 step 的最简伪代码

```python
# 1. 在线取 teacher batch
teacher_batch = stream_teacher_hidden(batch_size=teacher_batch_size)

# 2. 从 synthetic 集合中采样一个子 batch
synth_idx = sample_indices(synthetic_size, synthetic_batch_size)
synthetic_hidden = assemble_from_banks(bank_params, synth_idx)

# 3. 算分布匹配损失
losses = compute_losses(
    teacher_batch,
    synthetic_hidden,
    teacher_labels,
    synth_labels,
)

# 4. 可选：再过下一层 block，算 block-level 统计约束
if lambda_block > 0:
    teacher_next = forward_from_hidden(teacher_batch, start_layer=teacher_layer+1, end_layer=teacher_layer+1)
    synth_next = forward_from_hidden(synthetic_hidden, start_layer=teacher_layer+1, end_layer=teacher_layer+1)
    losses["block_rel_l2"] = pooled_rel_l2(teacher_next, synth_next)

# 5. 加权求和
total_loss = weighted_sum(losses)

# 6. 更新 synthetic banks
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

---

## 18. 一句话总结

这个 online distillation 方法，本质上是在做：

“从流式多模态教师数据中，持续在线提取并压缩某层 hidden，用一组按模态参数化的 synthetic hidden token bank 去拟合其分布统计，并可选地通过 next-block 约束保持局部层间动力学一致性。”

如果换成更工程化的表述：

- 输入：原始多模态样本流
- 教师监督：指定层压缩 hidden，必要时再加下一层输出统计
- 学习对象：一小份可训练 synthetic hidden 集合
- 训练目标：在尽量小的 synthetic 集合上逼近真实 teacher hidden 分布
- 输出：可复用的 distilled hidden `.pt`

---

## 19. 与这份说明最相关的代码入口

- `scripts/data_distill/run_distill_compact_hidden_online.sh`
- `src/calibration/representation_distill/distill_synthetic_hidden_online.py`
- `src/calibration/representation_distill/distill_synthetic_hidden.py`
- `src/calibration/representation_distill/common.py`
- `src/calibration/representation_distill/runtime/forward_from_hidden.py`
- `src/calibration/representation_distill/build_teacher_hidden_cache.py`

