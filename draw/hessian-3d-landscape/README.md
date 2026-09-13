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
