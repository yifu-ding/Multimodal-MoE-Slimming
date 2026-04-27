# 表征蒸馏训练曲线绘制建议

这份说明专门回答一个问题：在论文里，如何通过训练曲线清楚体现本文方法同时优化了 `diversity` 和 `distribution matching`，并且让读者从图上直接看懂本文方法的设计动机、优化行为和方法细节。

本文不建议把训练曲线当成“例行展示 loss 收敛”的附属图，而是把它作为方法解释图的一部分。原因是本文方法的核心不是单一目标最小化，而是：

1. 用分布匹配项让 synthetic hidden 靠近 teacher hidden 的统计结构。
2. 用 diversity 项防止 synthetic set 退化成少数重复模板。
3. 用按模态分组的 loss，避免 visual token 数量多时主导整体目标。
4. 用 teacher-teacher baseline 和 ratio，把“loss 下降”解释成“逼近 teacher 自身波动尺度”，而不是孤立数值变化。

因此，最有价值的图不是只画 `total loss`，而是要把“分布是否匹配得更好”和“多样性是否没有塌缩”同时可视化。

---

## 一、先明确每类指标在本文里代表什么

当前蒸馏脚本里，和论文画图最相关的指标可以分成四组。

### 1. Distribution matching 指标

这些指标对应 synthetic hidden 与 teacher hidden 在统计分布上的接近程度。

- `loss/raw/mmd`
  - 衡量两组 token 特征分布的核距离。
  - 更偏全局分布形状是否一致。
- `loss/raw/cov`
  - 衡量通道协方差矩阵是否接近。
  - 更偏二阶结构、通道相关性是否对齐。
- `loss/raw/mean`
  - 衡量一阶矩是否对齐。
- `loss/raw/var`
  - 衡量二阶矩中的逐通道方差是否对齐。

这四项共同组成本文的 distribution matching 部分。论文里最好不要只报其中一项，因为本文并不是只依赖 MMD，而是同时做：

- 全局分布形状对齐
- 通道相关结构对齐
- 低阶矩对齐

这正是本文方法比“只做某一种分布距离”更稳定的地方。

### 2. Diversity 指标

当前训练里的 diversity 主损失是：

- `loss/raw/div`
  - synthetic token 两两余弦相似度的非对角均值
  - 越低表示越分散，样本之间越不重复

另外还有更适合论文画图的诊断项：

- `diag/div_cosine_mean`
- `diag/div_cosine_p50`
- `diag/div_cosine_p90`
- `diag/div_cosine_max`

这组诊断比单独画 `div loss` 更有解释力。原因是：

- `mean` 说明整体平均多样性
- `p90` 说明高相似对是否仍然很多
- `max` 说明是否存在极端接近的重复样本

如果一条方法曲线 `mean` 降了，但 `p90/max` 仍然很高，说明只是平均上更分散，但尾部仍有 collapse 现象。这个分析很适合写进论文。

### 3. Baseline 和 Ratio 指标

这是本文里非常值得强调、但很容易被忽略的一组。

- `baseline/mmd_teacher_teacher`
- `baseline/cov_teacher_teacher`
- `baseline/mean_teacher_teacher`
- `baseline/var_teacher_teacher`

这些表示 teacher cache 内部随机两批样本之间本来就存在的分布差异。

进一步脚本还定义了：

- `ratio/mmd_vs_teacher_teacher`
- `ratio/cov_vs_teacher_teacher`
- `ratio/mean_vs_teacher_teacher`
- `ratio/var_vs_teacher_teacher`

这组 ratio 的意义非常强：

- `ratio > 1`：synthetic-teacher 差距大于 teacher 自身批间波动
- `ratio ≈ 1`：synthetic 已经接近 teacher 的“自然波动尺度”
- `ratio < 1`：synthetic 与 teacher 的差异已经不大于 teacher-teacher 的内部差异

论文里如果只画 raw loss，读者不知道数值大还是小、是否已经足够接近。  
但 ratio 图能直接回答：

“我们的 synthetic set 是否已经逼近到 teacher 自身的统计噪声范围内？”

这是本文很重要的方法解释点，建议一定画。

### 4. 几个辅助诊断项

还可以作为补充图或 appendix 图：

- `diag/token_norm_mean_gap`
- `diag/token_norm_std_teacher`
- `diag/token_norm_std_synth`
- `diag/centroid_l2`
- `diag/centroid_l2/mod0/mod1/mod2`
- `diag/token_norm_mean_gap/mod0/mod1/mod2`

它们不一定是主图，但很适合支持以下论点：

- synthetic hidden 不是仅在某个 loss 上取巧
- synthetic hidden 的 token norm 统计没有明显漂移
- 各模态都在靠近 teacher，而不是只优化文本或只优化视觉 token

---

## 二、论文主图应该画什么

如果只给一页图，我建议画成三联图或四联图。

---

### 图 A：Distribution matching 主曲线

建议画：

- x 轴：training step
- y 轴：`ratio/*_vs_teacher_teacher`

可选画四条：

- `ratio/mmd_vs_teacher_teacher`
- `ratio/cov_vs_teacher_teacher`
- `ratio/mean_vs_teacher_teacher`
- `ratio/var_vs_teacher_teacher`

#### Motivation

本文想说明的不是“loss 在下降”，而是 synthetic distribution 逐渐逼近 teacher distribution，并最终接近 teacher 自身 batch fluctuation 的尺度。

#### 想体现的方法细节

- 本文不是单一 MMD 方法，而是多种 distribution constraint 的组合。
- 本文用 teacher-teacher baseline 重新标定 loss 尺度，避免 raw value 难解释。

#### 图上最好怎么标

- 在 `y=1` 处画一条虚线，标记 “teacher-teacher level”
- 如果某些 ratio 在训练后期接近 1，可以在 caption 里明确写：
  - “The synthetic set approaches the intrinsic batch-to-batch variation of the teacher cache.”

#### 如何分析

可以按下面逻辑写：

1. 训练初期 ratio 明显大于 1，说明 synthetic initialization 与 teacher 分布存在显著偏差。
2. 随着训练进行，MMD、cov、mean、var ratio 同时下降，说明本文方法不是只修正均值，而是同时匹配全局核距离、协方差结构和低阶矩。
3. 若后期 ratio 接近 1，说明 synthetic set 已经接近 teacher 自身的统计波动水平，而不是仅仅取得某个较小但不可解释的 raw loss。

---

### 图 B：Diversity 主曲线

建议画：

- `diag/div_cosine_mean`
- `diag/div_cosine_p90`
- `diag/div_cosine_max`

如果图面太挤，可以主图画 `mean + p90`，appendix 再补 `max`。

#### Motivation

distribution matching 容易和 mode collapse 同时发生。也就是说，synthetic set 可能通过重复少数模式来靠近 teacher 的某些统计量。  
所以论文必须证明：本文在 improving distribution matching 的同时，没有把 synthetic samples 压缩成高度相似的集合。

#### 想体现的方法细节

- diversity regularization 不是可有可无的附加项，而是用来防止 synthetic hidden collapse 的关键设计。
- 本文不是只看 average similarity，而是还看 high-similarity tail。

#### 为什么推荐 `mean/p90/max`

- `mean`：反映总体趋势
- `p90`：反映“最相似的那一批 pair 是否还很多”
- `max`：反映极端 collapse 对

这能帮助读者理解本文为什么需要 `div`，以及它具体抑制了什么失败模式。

#### 如何分析

可以按下面逻辑写：

1. 如果不加 diversity，distribution loss 可能继续下降，但 `p90/max` 会升高或维持较高水平，说明 synthetic set 逐渐重复化。
2. 本文方法中，distribution ratio 持续下降的同时，`div_cosine_mean/p90` 保持较低，说明分布匹配不是靠重复样本实现的。
3. 如果早期有 warmup，caption 或正文可以解释：
   - 前期让 `div` 渐进介入，是为了避免在 distribution term 尚未稳定时，过强的 repulsion 破坏对齐。

---

### 图 C：Distribution vs Diversity 的 trade-off 图

这张图非常适合本文。

建议画二维相图：

- x 轴：某个 distribution summary，例如
  - `avg_ratio = mean(ratio/mmd_vs_teacher_teacher, ratio/cov_vs_teacher_teacher, ratio/mean_vs_teacher_teacher, ratio/var_vs_teacher_teacher)`
  - 或者只用 `ratio/mmd_vs_teacher_teacher`
- y 轴：`diag/div_cosine_mean` 或 `diag/div_cosine_p90`
- 每隔若干 step 取一个点，按时间顺序连线

#### Motivation

单独画两张 time series 还不够直观。  
本文核心主张其实是：优化轨迹在“更好 distribution match”和“更高 diversity”之间找到了更好的 Pareto-like balance。

#### 想体现的方法细节

- 本文不是盲目最小化 distribution loss
- 本文是在约束 collapse 的前提下逼近 teacher
- warmup 和多项 loss 组合让轨迹朝“左下角”移动，而不是只在一个维度上改善

#### 如何分析

最理想的现象是：

- full 方法轨迹逐步移动到左下角
- `no_div` ablation 虽然 x 轴变小，但 y 轴变坏
- `mmd_only` 或 `moment_only` 可能在某一维改善，但整体 trade-off 不如 full

如果你要让图很有论文味道，这张图比单纯画 loss 曲线更能体现“方法设计的必要性”。

---

### 图 D：Ablation 对照图

建议直接复用你现在脚本里的 ablation 设定：

- Diversity ablation
  - `full`
  - `no_div`
- Distribution ablation
  - `full`
  - `moment_only`
  - `mmd_only`

建议画法：

- 左图：`avg distribution ratio` over step
- 右图：`diag/div_cosine_p90` over step

#### Motivation

这组图直接服务于方法消融，不只是训练过程展示。

#### 想体现的方法细节

- 为什么 distribution 部分要同时保留 MMD、cov、mean、var，而不是只做一种
- 为什么 diversity regularization 必须存在
- 为什么本文是“组合设计”，不是任意堆 loss

#### 如何分析

你可以按下面逻辑组织：

- `no_div`：
  - distribution ratio 也许并不差，甚至局部更低
  - 但 diversity 诊断显著恶化
  - 说明 distribution-only 容易 collapse
- `mmd_only`：
  - 全局核距离可能下降较快
  - 但 mean/var/cov 的对齐不够稳
  - 说明单独 MMD 难以稳定约束低阶统计和通道结构
- `moment_only`：
  - 均值方差改善
  - 但 MMD 下降不足
  - 说明仅靠矩匹配无法完整约束分布形状
- `full`：
  - distribution ratio 全面下降
  - diversity 维持更健康
  - 说明本文各个组件互补

---

## 三、如果只能挑最重要的 3 张图

如果正文版面非常紧，我建议优先这三张：

1. `Distribution ratio vs step`
2. `Diversity diagnostics vs step`
3. `Distribution-diversity trade-off trajectory`

这三张合在一起，基本就能把本文方法逻辑讲清楚：

- synthetic set 确实越来越像 teacher
- 不是靠 collapse 得到的
- 本文方法在两者之间取得了更好的平衡

---

## 四、每张图的具体绘制步骤

下面给出一套论文写作时最容易复现的流程。

### Step 1：收集日志字段

从 W&B 或 history 中导出以下字段：

- `step`
- `ratio/mmd_vs_teacher_teacher`
- `ratio/cov_vs_teacher_teacher`
- `ratio/mean_vs_teacher_teacher`
- `ratio/var_vs_teacher_teacher`
- `diag/div_cosine_mean`
- `diag/div_cosine_p90`
- `diag/div_cosine_max`
- 可选：
  - `diag/centroid_l2`
  - `diag/token_norm_mean_gap`
  - `mmd/mod0, mod1, mod2`
  - `cov/mod0, mod1, mod2`

### Step 2：构造一个 distribution summary

为了让图更简洁，建议再构造一个汇总量：

- `distribution_ratio_avg`
  - 四个 ratio 的均值

或者：

- `distribution_ratio_max`
  - 四个 ratio 的最大值

二者含义不同：

- `avg` 更适合主图，表示整体分布匹配水平
- `max` 更适合补充图，表示最差那一项是否仍然落后

### Step 3：对 diversity 选择合适指标

建议主文：

- `diag/div_cosine_mean`
- `diag/div_cosine_p90`

补充材料：

- `diag/div_cosine_max`

原因是 `max` 容易有噪声，主文中可能不够平滑；`p90` 更稳，且能反映尾部。

### Step 4：做平滑，但不要过度

论文图可以做轻微 smoothing，比如 moving average。  
但要控制在不改变趋势判断的程度。

建议：

- 短窗口平滑
- 同时保留原始采样点或在 appendix 提供 unsmoothed 版本

原因是：

- distribution 和 diversity 本身可能存在拉扯
- 过度平滑会把这种优化动态掩盖掉

### Step 5：给图加解释性标记

建议统一做这些视觉标记：

- 在 ratio 图上加 `y=1` 的虚线
- 在 trade-off 图上标箭头或用颜色映射 step
- 在 warmup 存在时，标出 warmup 结束 step

这些标记能帮助读者把训练机制和曲线行为对应起来。

---

## 五、如何把“按模态分组”的方法细节也体现在图里

本文一个很重要的实现细节是：

- `mmd/cov/mean/var` 是按模态分组后求平均
- `div` 是在 pooled synthetic tokens 上计算

这点建议不要只放在方法段落里，最好在图上也体现一次。

### 推荐做法

单独加一张 appendix 图，画：

- `loss/mmd/mod0`
- `loss/mmd/mod1`
- `loss/mmd/mod2`

以及对应的：

- `loss/cov/mod0`
- `loss/cov/mod1`
- `loss/cov/mod2`

这里的 `mod0/mod1/mod2` 可以在图例中明确写成：

- `text`
- `image`
- `video`

#### 想体现什么

这张图的目的不是展示哪一模态最低，而是说明：

- 本文不是被数量更多的 visual token 单方面驱动
- 各模态都在被显式约束
- 分组计算确实让多模态对齐更均衡

#### 可以怎么分析

你可以这样写：

“Without modality-grouped matching, optimization tends to be dominated by token-rich modalities.  
Our grouped losses reduce all three modality-specific gaps concurrently, indicating a more balanced multimodal calibration process.”

---

## 六、建议的正文叙述结构

正文可以按下面顺序展开，这样最顺：

### 段落 1：先讲 distribution

先说：

- synthetic set 与 teacher set 的统计距离持续下降
- 且下降到接近 teacher-teacher baseline 的量级

这个段落主要对应 ratio 图。

### 段落 2：再讲 diversity

接着说：

- distribution 改善并不是通过 collapse 获得
- diversity 诊断项保持较低，尤其尾部相似性没有恶化

这个段落主要对应 diversity 图。

### 段落 3：最后讲 trade-off

最后说：

- 本文 full 方法在 distribution-diversity 平衡上优于消融版本
- 说明本文的组合设计是必要的

这个段落主要对应 trade-off 图和 ablation 图。

这样叙述时，读者会自然理解：

- 为什么要同时画 distribution 和 diversity
- 为什么单一 loss 不够
- 为什么本文方法设计有效

---

## 七、几种常见错误画法，建议避免

### 错误 1：只画 total loss

问题：

- total loss 混合了多种权重和 warmup
- 很难解释到底是 distribution 变好，还是 diversity 在主导

### 错误 2：只画 raw MMD

问题：

- 它不能代表全部 distribution matching
- 也不能说明是否接近 teacher 自身的波动尺度

### 错误 3：只画平均 diversity

问题：

- `mean` 可能改善，但尾部仍有 collapse
- 至少要补一个 `p90` 或 `max`

### 错误 4：不画 ablation 对照

问题：

- 读者无法判断这些 loss 项是否只是“都加一点更好”
- 也看不出 diversity 与 distribution 之间的功能分工

---

## 八、我最推荐的最终成图方案

如果你想做一套最稳、最像论文主结果的图，我建议最终采用下面这个组合。

### 主文图 1：Distribution Matching

- 四条 ratio 曲线
- 或者 `distribution_ratio_avg + distribution_ratio_max`
- 带 `y=1` 虚线

### 主文图 2：Diversity Preservation

- `diag/div_cosine_mean`
- `diag/div_cosine_p90`

### 主文图 3：Trade-off / Optimization Trajectory

- x: `distribution_ratio_avg`
- y: `diag/div_cosine_mean` 或 `p90`
- full / no_div / mmd_only / moment_only 四条轨迹

### 附录图 1：Modality-wise Matching

- `mmd/mod{text,image,video}`
- `cov/mod{text,image,video}`

### 附录图 2：Stability Diagnostics

- `diag/centroid_l2`
- `diag/token_norm_mean_gap`
- `diag/token_norm_mean_gap/mod{text,image,video}`

---

## 九、一句总结，论文里可以怎么概括这些图

可以概括成下面这句逻辑：

“训练曲线表明，本文方法在降低 synthetic-teacher 分布差异的同时，保持了 synthetic set 的内部多样性；进一步地，相对于 teacher-teacher baseline 的 ratio 分析说明，最终 synthetic distribution 已经逼近 teacher 表征的自然波动尺度，而非通过模式塌缩获得表面上的 loss 降低。”

这句话基本就是整套图想传达的中心结论。

