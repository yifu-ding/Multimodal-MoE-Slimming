# 观察实验 B：单 expert 二阶信号验证

本目录包含论文用图，以及复现该图所需的精简真实数据。观察实验 A 和此前的
双 expert 三维 landscape 均标记为 **LEGACY / NOT USED**，不属于本图。

## 这张图说明什么

- 图 (a)-(b)：在 3,968 个 layer-expert 方向上，autograd HVP 分数
  `H_ee/2` 与 routed expert 输出能量、单 expert 删除误差一致。
- 图 (c)：量化计算收益。逐 expert 暴力消融需要每层进行 128 次 forward；
  Gram/energy 形式从一次正常 routed forward 同时取得所有 expert 的精确分数。
- 图 (d)-(f)：展示第 0 层低、中、高敏感度 expert 的真实 beta forward。
  在恒等重构点，一阶预测为零；而对角 Hessian 给出的抛物线与所有实测点
  重合，其中也包括 `beta_e=0` 的单 expert 删除点。图中同时明确报告：一阶
  排序的 Spearman 未定义，因为所有分数都是零；二阶排序的 Spearman 为 1。

本图不包含任何双 expert 联合删除。

冻结数据覆盖已经完成采集的 31 个 MoE 层，每层均包含 128 个 routed
experts，并使用校准 manifest 的前 32 个样本。这里没有必要运行模型的所有
层，因为本实验验证的是逐点成立的代数等价关系，而不是对总体均值进行统计
估计。

## 为什么图 (a)-(b) 是正确性检查，而不是精度优势

固定 router 后，block 输出对单个 expert scale `beta_e` 是线性的。记该
expert 的 routed contribution 为 `f_e(x)`，则相对恒等输出的 MSE 为

```text
L(beta_e) = (beta_e - 1)^2 * ||f_e(x)||^2.
```

因此

```text
H_ee / 2 = ||f_e(x)||^2 = expert_output_energy
           = single_expert_ablation at beta_e = 0.
```

所以图 (a)-(b) 中的点贴合 `y=x` 是由线性缩放和 MSE 共同保证的恒等式。
Pearson/Spearman 接近 1、相对误差约为 `1e-7`，证明的是 HVP 数值实现、
FP32 累积和 fixed-router 处理正确，并不表示它击败了一个较弱的预测方法。

该恒等式的实际收益体现在图 (c)：暴力得到全部单 expert 删除代价，需要对
每层 128 个 experts 分别做扰动 forward；Gram/energy 形式可以复用一次正常
routed forward 中的 contribution，同时得到全部精确分数，即 128 次对 1 次。
这里的“1 次”指 Gram/energy 数据获取，不应表述成“一次 HVP 得到全部 Hessian
对角元”。本仓库的精确 HVP 验证实现使用分块 batched VJP，可能包含多次
backward 调用。

## 为什么一阶为零不是 strawman

在 `beta_e=1` 的恒等重构点，`L=0` 且取得 MSE 最小值，因此根据极值点的一阶
必要条件，所有 experts 都有 `g_e=dL/d beta_e=0`。一阶分数由此退化成一个
常数，无法产生 expert 排序；其相对于真实删除代价的 Spearman 相关系数未定义。
二阶分数随 expert 改变，对真实删除代价的 Spearman 为 `1.000000`。

图 (d)-(f) 将这一点放在 P10、P50、P90 三种敏感度上直接展示：一阶预测对
三者都给出相同的零，二阶曲率则给出不同抛物线并穿过真实 forward 点。这与
Optimal Brain Damage 和 Optimal Brain Surgeon 中使用曲率区分待删除单元的
经典论证一致。

## 计划中的扩展：两个更公平的一阶基线（尚未采集数据，本节只是方案）

> [!NOTE]
> 本节记录的两个基线**目前没有对应的真实数据**，`method_validation_B.png/pdf`
> 里也没有画出来。这里只是把"如果要做，应该采集什么、怎么画"写清楚，供
> 后续实现时对照，不要误当成已经完成的实验。

现有图里的"一阶预测"（灰色虚线）用的是 `identity_gradient`：只在
`beta_e=1` 测一次导数，然后外推到别的 `beta`。这确实是唯一一种"不需要真的
做扰动就能打分"的一阶方法，但严格来说只回答了"在正常工作点上，纯梯度打分
行不行"这一个问题。下面两个基线要回答的是另外两个更公平的问题：如果真的
在每个扰动点上重新算，一阶和"直接看 loss"分别会是什么样、准到什么程度。

### 方案①：逐点局部一阶（不外推，在每个 beta 上重新求导）

对已经在扫描的每一个 `beta_0`（比如 -0.5、-0.25、0、0.25、...、1.5），
**真的把该 expert 的缩放系数设成 `beta_0`**（不是 1），重新做一次真实的
forward + backward，取该点自己的导数

```text
g_e(beta_0) = dL / d beta_e 在 beta_e = beta_0 处的真实值。
```

因为 `L(beta_e) = (beta_e - 1)^2 * ||f_e(x)||^2` 是精确二次的，这个量有
闭式解 `g_e(beta_0) = H_ee * (beta_0 - 1)`，随 `beta_0` 线性变化、只在
`beta_0 = 1` 处为零——这正是你说的"这个一阶必然不准"里"不准"的来源：
用 `g_e(beta_0)` 只做一步局部线性外推，预测 `beta_0` 附近一个小偏移
`delta` 处的 loss，

```text
L1_local(beta_0, delta) = L(beta_0) + g_e(beta_0) * delta,
```

它和真实值的误差是**可以precise 算出来的**：

```text
L(beta_0 + delta) - L1_local(beta_0, delta) = (H_ee / 2) * delta^2.
```

也就是说，不管从哪个 `beta_0` 出发，局部一阶的偏差永远等于二阶项本身，
偏移越大偏差越大——这比"从 beta=1 外推到 beta=0 恰好退化成 0"更有说服力，
因为它展示的是"一阶在处处都系统性偏小，不是只在这一个特殊点上失效"。

**需要新采集的数据**：在 `collect_method_validation_b.py` 现有的 beta
sweep 循环里，每个 `beta_0` 目前只记录 `measured_delta_mse`；需要在同一次
forward 里加一次 backward，把 `dL/d beta_e` 在**当前** `beta_0`（不是恒定
在 `beta_e=1`）处的真实值也存下来，建议加一列
`local_gradient_at_beta`（区别于现有的、只在 `beta_e=1` 测一次的
`identity_gradient`）。三个已选定的 P10/P50/P90 expert 复用现有 sweep 网格
即可，不需要新增 layer/expert。

**画法建议**：横轴 `beta`，在已有的 measured 圆点、二阶抛物线基础上，为
每个采样到的 `beta_0` 画一小段局部切线（只在该点附近一个小窗口内画，不要
画满全程，否则会和"从 beta=1 外推"的旧灰线混淆），或者更干净的方式：另开
一个子图，横轴 `beta_0`，纵轴是"局部一阶预测误差" `(H_ee/2)*delta^2`
（对固定 `delta`，比如 0.25）与"二阶预测误差"（应恒为 0，只有数值精度
噪声）的对比，直接把"处处都不准 vs. 处处都精确"画成两条曲线。

### 方案②：小扰动下的真实 loss 直接当敏感度代理（不做任何 Taylor 拟合）

选一个固定的小扰动 `delta`（比如 0.1，即 `beta_e = 0.9`），对每个待比较
的 expert 只做**一次**真实 forward，直接把测到的

```text
proxy_e = measured_delta_mse at beta_e = 1 - delta
```

当作重要性分数，不拟合任何一阶或二阶公式。这是"不够 precise 但很直觉"的
基线：既不需要 autograd（比一阶/二阶都简单），也不需要做到 `beta_e=0` 的
完整删除（比暴力消融便宜），但它测的是"打 9 折时的 loss"，不是"删除时的
loss"。

**用一个不涉及公式的比方，先说清楚这里最容易踩的坑**：假设要比较两个
expert A、B 谁更重要，方法是"把它的输出调小一点，看 loss 涨多少，涨得
越多说明越重要"。这里的关键是，"调小一点"这个"一点"必须对 A、B **用
同一个幅度**（比如都调小 10%），测出来的涨幅谁大谁小才是可信的——这是
在用同一把尺子量。如果对 A 只轻轻调小 5%、对 B 使劲调小 50%，那测出来
"B 涨得更多"完全可能只是因为 B 被推得更狠，跟 B 是不是真的更重要没关系，
排序会被推力大小本身带偏。所以 `proxy_e` 这个方案要测出正确的排序，前提
是**所有 expert 都必须用同一个 `delta`**，不能有的 expert 调 5%、有的调
50%。

在"所有 expert 用同一个 `delta`"这个前提下，因为 `L` 对每个 expert 都
严格二次，可以进一步精确证明：

```text
proxy_e = (delta^2 / 2) * H_ee，对每个 expert 都成立。
```

也就是说 `proxy_e` 只是把每个 expert 真正的重要性分数 `H_ee` 乘上了
**同一个**正常数 `delta^2/2`——同一把尺子等比例缩放，不会改变谁大谁小，
所以 `proxy_e` 排出来的顺序在数学上必然和精确二阶完全一样（Spearman
必然为 1）；不精确的地方只出在**绝对数值**上：`proxy_e` 系统性地比真实
删除代价（`beta_e=0` 处的 `H_ee/2`）小 `delta^2` 倍，除非恰好
`delta=1`。这一点值得显式画出来（比如在 log-log 图上，`proxy_e` 的点会
整体贴着一条斜率为 1、但纵截距偏移了 `log(delta^2)` 的平行线，而不是
偏离 `y=x` 的散点）——它比只说"不够精确"更诚实：问题不是排序错了，是
数值口径系统性地偏了一个可以提前算出来的固定倍数，不是随机误差。

**需要新采集的数据**：现有 `method_validation_B_beta_curves.csv` 里，
P10/P50/P90 三个 expert 已经采过 `beta=0.75`、`0.5`、`0.25` 等中间点，
可以直接拿现成数据做一个 3 点规模的验证（选其中一个已有 `beta` 当作
`delta`，检查 `proxy_e / H_ee` 在三个 expert 上是不是同一个常数）。但要在
全部 3,968 个 `(layer, expert)` 规模上验证排序不变性（像 (a)(b)(c) 那样
报告 Pearson/Spearman/相对误差的分布），现有 `method_validation_B_scores.csv`
只有 `beta_e=1`（energy/HVP）和 `beta_e=0`（完整删除）两个点，没有中间
`beta` 的测量，需要新增一次全量小扰动 forward 扫描，建议在
`collect_method_validation_b.py` 里对当前已经在扫描的所有 active
`(layer, expert)`，额外跑一遍 `beta_e = 1 - delta` 的 routed forward，
新增导出字段 `partial_perturbation_delta_mse` 和用到的 `delta` 值（写进
`method_validation_B_metadata.json`，因为这是一个需要固定并公开的超参数，
不能每个 expert 用不同的 `delta`，否则上面的排序不变性证明不成立）。

### 两个方案放在一起说明什么

方案①说明"即使不再用退化的 `g_e=0`，只要还是一阶（线性），处处都会有一个
可以精确算出来、随扰动幅度增大而增大的误差"；方案②说明"用真实 loss 做
代理确实能在排序上蒙对，但这只是因为单 expert 移除下 loss 恰好严格二次这
个特例——它对绝对数值的估计依赖一个必须提前选定、对所有 expert 统一的
`delta`，本质上是在用一个更贵（需要真实 forward）但仍然不完整的信号，去
近似本来就有闭式解的二阶量"。两者都不需要重新采集主实验数据，只需要在
现有 beta sweep 基础上分别新增"每点重新求导"和"每点额外测一次小扰动
loss"这两类记录。

## 每个数据点是什么意思

图 (a) 和 (b) 中的每个散点对应一个 `(layer, expert)`，结果在同一批 32 个
样本上累计，并除以参与评分的 token 总数。横坐标都是 autograd HVP 的结果：

```text
hvp_hessian_half = (1 / 2) * d^2 L / d beta_e^2，在 beta_e = 1 处计算。
```

图 (a) 的纵坐标是 `expert_output_energy`；图 (b) 的纵坐标是只把该 expert
的缩放系数设为零后产生的输出 MSE，即 `single_expert_ablation`。计算期间
router 保持不变。因此，一个点落在红色 `y=x` 线上，表示该 expert 的对角
Hessian 分数能够精确预测其单独删除代价。

图 (d)-(f) 中的每个圆点，是选定的第 0 层 expert 在某个 `beta` 取值下进行
一次真实 block forward 得到的结果。三个 expert 分别对应第 0 层 `H_ee/2`
排名的 P10、P50 和 P90。红色菱形是 `beta=0` 的圆点，也就是单 expert 删除。
这些圆点是实测值，不是从图中的理论抛物线上采样得到的。

## 数据字段说明

`data/method_validation_B_scores.csv` 中，每一行对应一个活跃的
`(layer, expert)`：

- `layer`、`expert`：从零开始计数的 MoE 层编号和 expert 编号。
- `hvp_hessian_half`：对角二阶分数 `H_ee/2`。
- `expert_output_energy`：routed expert contribution 的平方能量，使用相同的
  hidden-dimension MSE 和 score-token 归一化方式。
- `single_expert_ablation`：只删除该 expert 后，根据其真实 routed
  contribution 精确构造的输出 MSE；该值没有使用二阶公式计算。
- `identity_gradient`：在 `beta_e=1` 处的 `dL/d beta_e`。由于恒等重构点是
  MSE 最小值，这里的梯度为零。
- `active_batch_count`：两个验证 batch 中，该 expert 至少接收到一个评分
  token 的 batch 数量。

`data/method_validation_B_beta_curves.csv` 中，每一行对应一个真实 beta
forward：

- `sensitivity`、`quantile`：low/P10、medium/P50 或 high/P90。
- `layer`、`expert`、`beta`：被缩放的 expert 方向及实际计算的缩放系数。
- `measured_delta_mse`：相对于 `beta=1` 基线的实测 MSE 增量。
- `first_order_prediction`：`g_e * (beta-1)`。
- `second_order_prediction`：
  `g_e * (beta-1) + (H_ee/2) * (beta-1)^2`。
- `identity_gradient`、`hvp_hessian_half`：一阶和二阶预测使用的系数。

`data/method_validation_B_metadata.json` 记录模型、准确的 manifest 哈希、
层列表、样本数、归一化方式、FP32 验证精度和 fixed-router 检查结果。

## 复现图片

在仓库根目录运行：

```bash
python draw/hessian-3d-landscape/plot_method_validation_b.py
```

绘图脚本只读取以下三个文件：

```text
draw/hessian-3d-landscape/data/method_validation_B_scores.csv
draw/hessian-3d-landscape/data/method_validation_B_beta_curves.csv
draw/hessian-3d-landscape/data/method_validation_B_metadata.json
```

脚本会在本目录生成 `method_validation_B.pdf`、`method_validation_B.png`，
以及指标摘要 `method_validation_B.json`。

## 重新采集或导出数据

四卡采集器使用 FP32 copied block，以保证严格的数值一致性：

```bash
LIVE_LOGS=0 FORCE=1 VALIDATION_SAMPLES=32 \
bash scripts/run_method_validation_b_4gpu.sh
```

采集器会以原子方式逐层保存 checkpoint。任务中断后，可使用
`RESUME=1 RUN_SMOKE_TEST=0` 继续；resume 模式会先验证全部采集元数据，再
跳过已经完成的层。

将合并后的采集结果转换成精简公开数据：

```bash
python scripts/export_method_validation_b_csv.py \
  --input <output-dir>/method_validation_b.pt \
  --output-dir draw/hessian-3d-landscape/data
```

运行数值验收：

```bash
python scripts/check_method_validation_b.py \
  --input <output-dir>/method_validation_b.pt
```
