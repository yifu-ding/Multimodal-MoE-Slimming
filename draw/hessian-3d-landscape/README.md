# 观察实验 B：单 expert 二阶信号验证

本目录包含论文用图，以及复现该图所需的精简真实数据。观察实验 A 和此前的
双 expert 三维 landscape 均标记为 **LEGACY / NOT USED**，不属于本图。

新增真实实验：[beta=0.95 一阶与 beta=1 Hessian 的排序反例](README_beta095_experiment.md)。
已完成 L2/KL 第0层32样本采集，结果和新图见 [beta095/RESULTS.md](beta095/RESULTS.md)。

## 待补采：所有 expert 同步 beta sweep 的真实梯度（2026-09-13）

**状态：待采集；本节是服务器运行规格，不是已经完成的实验结果。**
目标是实测验证：同一层所有 expert 的缩放系数一起变化时，每个 expert 的
一阶梯度是否随 beta 线性变化，以及相对于全体 beta=0 的梯度绝对值，
不同 expert 的变化百分比是否一致。不要预设或筛选“存在非线性”的结果。

### 范围与采集点

- 仅采 **第0层全部128个 routed experts**，保留每个expert，包括梯度为0者。
  其他层保持原模型状态，不同时缩放其他层。
- 主采样点严格为用户指定的 **beta=[0, 0.25, 0.5, 0.75, 0.95]**。
- Loss 分别采 **L2** 与 **hidden-state softmax KL**，分目录保存。
  KL沿用现有采集器的方向 `KL(softmax(teacher_hidden) || softmax(student_hidden))`，
  不是最终词表KL或下游任务交叉熵。
- 沿用 `Qwen/Qwen3-VL-30B-A3B-Instruct`、原GQA校准manifest的前32个样本，
  顺序、token mask和loss归一化与 `beta095/{l2,kl_div}/metadata.json` 一致。
  原记录为2批、batch size=16、合计4096个score tokens；若环境需改batch size，
  保留相同样本/token集合，记录变化，并重新核验数值。
- 原manifest位置为 `/home/data/dyf/MARS-results/storage/calibration_manifests/gqa-256-seed42.json`，
  SHA256为 `24c3a5e250c33ca53df6addb82ee78de2efe94f1bff2487ee754e6f6b347d6d5`。
  路径可随服务器调整，内容哈希必须核对。

### 最关键的操作：同值赋给向量，但分别对每个 expert 求导

保留独立可求导的128维向量 `alpha`，每个分量控制对应expert的routed contribution。
在每个采样点，先同时设置 `alpha[:] = beta`，再做一次真实forward/backward，
用 `autograd.grad(loss, alpha)` 一次得到128个偏导数：

```
g_e(beta) = partial L(alpha) / partial alpha_e, evaluated at alpha=beta*ones
```

**不要把alpha替换成一个广播的可求导标量beta再求导。** 那样得到的是
`dL(beta*ones)/dbeta = sum_e g_e(beta)`，只有一个总梯度，无法画expert-ID横轴。
也不要逐个expert改beta、其余保持1或0.95；那是现有单expert扫描，并非本实验。

伪代码（复用现有loss、mask和expert hook；不是可直接运行的完整脚本）：

```python
alpha = torch.ones(num_experts, device=device, requires_grad=True)
# alpha是叶子向量；只冻结模型参数，不能冻结alpha或禁用student的autograd。
for batch in calibration_batches:
    inputs, kwargs, mask = original_teacher_block_inputs(batch)
    with torch.no_grad():
        alpha.fill_(1.0)
        teacher = forward_block(inputs, kwargs).detach()
    # teacher在整个batch的sweep中固定；beta=0也不能换teacher。
    for beta in [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]:
        with torch.no_grad():
            alpha.fill_(beta)  # 本层全部expert一起变，每次恢复完整向量
        prediction = forward_block(inputs, kwargs)
        loss = loss_sum(prediction, teacher, mask, loss_fn)
        gradient = torch.autograd.grad(loss, alpha)[0]
        save_batch_measurement(beta, loss.detach(), gradient.detach(), mask.sum())
```

- 只缩放router加权后的expert贡献；residual、router logits、路由权重、共享分支
  保持原实现，**不重新归一化路由权重**。比较各beta的router indices/weights，
  确认与同批全1参考一致。输入hidden states来自原模型，不串行传播上一个beta的输出。
- teacher始终是该批原模型全1输出。每个点独立执行模型，不累积删除或扰动。
- 与现有协议一致：copied block为FP32，loss算术为FP64；落盘梯度和loss累积用FP64。
  如果采用不同精度，必须单独标记，不能与既有结果冒充同一数值实验。
- **先跨batch/token累加signed gradient，再取绝对值**：
  `g_e(beta)=sum_batch gradient_sum_e(beta)/sum_batch num_score_tokens`。
  不先取绝对值再平均，不对不同token数的batch等权平均。
- 原采集器 `src/calibration/collect_beta095_counterexamples.py` 中
  `set_scale(alpha, beta)` 与 `measure_batch` 的全体求导方式可复用；
  但其入口限制beta_work=0.95，且后续beta_curves是逐expert扫描。
  **服务器端需要新增全体同步sweep入口/模式，不能直接把旧beta_curves当作本数据。**

### 需要保存什么

新增独立目录，不覆盖既有 `beta095/` 或 `data/`：

```
draw/hessian-3d-landscape/global_beta_sweep/
    l2/
        gradients.csv
        losses.csv
        metadata.json
        validation.json
        linearity.json
        raw.pt
    kl_div/
        gradients.csv
        losses.csv
        metadata.json
        validation.json
        linearity.json
        raw.pt
```

`gradients.csv`：每个 `(beta, expert)` 一行。主采样640行，加beta=1检查点后768行/每种loss。
必需字段：

| 字段 | 含义 |
| --- | --- |
| layer, expert, loss_fn | 层号、expert ID、loss类型 |
| beta_global | 本层全部128个alpha的共同值；不能用单expert beta含糊替代 |
| num_score_tokens | 同一份聚合分母 |
| gradient_signed | 聚合后的真实偏导g_e，不含额外beta因子 |
| gradient_abs | abs(gradient_signed) |
| gradient_at_global_zero | 对应expert在全体alpha=0时的signed gradient |
| ratio_to_global_zero_pct | 100*abs(g_e(beta))/abs(g_e(0))；不稳定分母时留空 |
| ratio_valid | 分母是否通过近零阈值检查 |
| first_order_signed | 可选诊断列：-beta_global*gradient_signed |
| first_order_abs | 可选诊断列：abs(beta_global*gradient_signed) |

最后两列仅用于区分梯度与删除方向的一阶打分，**主图百分比必须用gradient_abs**。
尤其beta=0时 `abs(beta*g)=0`，不能用这个打分作百分比分母。

`losses.csv`：每个beta一行，保存 `layer,loss_fn,beta_global,num_score_tokens,loss`。
这里是全体同步缩放产生的重构loss，不是独立删除某一个expert的loss。

`raw.pt`：保存每个batch、每个beta的原始signed gradient向量、loss_sum、
num_score_tokens、样本ID/顺序、重复测量和有限差分验证结果；保留梯度全精度，
不能只留下百分比或舍入后的打印值。不需要保存整个模型或完整autograd计算图。

`metadata.json`：至少包含模型、layer、num_experts、loss精确定义、
`sweep_mode="all_experts_synchronous"`、主beta列表/检查beta列表、
`teacher_background=1`、缩放位置、fixed_router、原manifest路径/hash、
实际样本IDs、batch size、总token数、参数化 `independent_alpha_vector_at_equal_values`、
block/loss/累积dtype、随机种子、代码commit、采集脚本hash及complete标记。

### 数值验收与线性检查

1. 核对每个beta恰有同一组128个expert、总token数一致，数值有限；路由完全一致。
2. 在全局beta=0、0.5、0.95各重复一次独立forward/backward，保存每个expert
   的重复梯度差异和loss差异，用于区分真实非线性与数值噪声。
3. 对beta=0.5处expert ID 0、64、127进行中心差分检查，建议步长0.005和0.01。
   这里为了验证每个偏导，临时仅将被验证expert改为beta±step，其他expert固定beta；
   这些是验证记录，不能混进global_beta_sweep主表。
4. 同轮beta=1检查loss和梯度接近0；保存实际值，不强制写0。
   同轮beta=0.95与旧 `gradient_at_work` 在相同样本/归一化下核对，报告差异；
   可参考历史数据，但本轮各点优先全部重新测量，避免跨run精度混杂。
5. 对近零分母标记无效，而不是加epsilon后强行画百分比。
   可预先采用 `tau=max(8*max_repeat_gradient_error, 32*eps_float32*max_e(abs(g_e(0))))`，
   记录tau实际值和被屏蔽expert IDs；原始梯度行不丢弃。
6. 逐expert用实际beta坐标拟合 `g_e(beta)=a_e*beta+b_e`（signed gradient），
   输出斜率、截距、R²、最大绝对残差、残差/梯度变化范围；常数曲线的R²标为null。
   beta间距不全相同，不能按点序号拟合，也不能直接比较未经步长归一化的增量。
7. 独立计算 `r_e(beta)=g_e(beta)-(1-beta)*g_e(0)`，输出每beta/每expert的
   signed residual、绝对残差和有效分母下的百分比偏差。与重复测量误差比较后，
   才讨论是否存在可分辨的非线性；不要仅因浮点数不完全相等就声称非线性。
8. 输出每个beta下有效expert百分比的min/max/mean/std，描述跨expert的变化幅度。
   beta=0的有效比例应为100%；其余点来自真实测量，不按理论补齐或平滑。

### 后续图怎么画，以及预期结论的边界

- x轴：expert ID（0至127，按ID排列）；y轴：`|g_e(beta)| / |g_e(0)| * 100%`。
- 每个主采样beta一条曲线/一种颜色，全部使用新测量；L2/KL分别成子图。
  beta=1只用于验收，默认不加到主图；近零分母留缺口并报告数量。
- 若不同beta的线相近，可另外画相对 `100*(1-beta)` 的偏差（单位percentage points），
  但必须标清放大尺度和重复测量噪声，不能把噪声渲染成显著现象。
- L2固定路由且输出对alpha仿射时，理论为
  `g_e(beta*ones)=(beta-1)*(H*ones)_e`，故有效比例为 `100*(1-beta)%`。
  对指定5点预期分别是100%、75%、50%、25%、5%。这是理论预期，不是已采到的结果。
  若实测一致，应如实报告各expert共同缩放，而不是寻求不存在的差异。
- KL不保证上述严格比例，由本轮实测决定是否有跨expert差异以及差异是否超过数值噪声。
- 本次只需真实梯度和loss，**无需补采Hessian、逐expert独立删除、联合删除或其他层**。
  若以后需要实测“梯度随全局beta的导数”，对应方向量是 `(H*ones)_e`，
  不是单独的H_ee；不要把已有Hessian对角元直接当作这条global曲线的斜率。

---

## 当前图：第 0 层 L2 / KL 对照

当前 `method_validation_B.{png,pdf,json}` 使用同一份 GQA manifest 的前 32 个
样本、第 0 层 128 个 experts。所有绘图输入、图片和统计结果统一保存在本目录：
L2 CSV 在 `data/`，KL CSV 在 `data_kl/`，不再使用 artifacts 下的旧数据目录。

- 第一行 (a)-(c)：L2 HVP vs. Gram energy、L2 HVP vs. 真实删除代价、
  KL HVP vs. 真实删除代价。KL 不具有 Gram/energy 等价性，因此不画该检查。
- 第二行 (d)-(f)：L2 的 P10/P50/P90 beta 扫描。
- 第三行 (g)-(i)：KL 的 P10/P50/P90 beta 扫描。

扫描图左轴为对应 loss 的实测变化及 Hessian 二次预测，右轴只画每个 beta
处真实测得的一阶梯度。同一行共用左右轴范围，L2/KL 分别设置尺度。
分位数按各自 loss 选择，因此两行的 expert ID 不同。

KL 删除代价与 HVP 的 Pearson 为 0.999925、Spearman 为 0.999348，
相对 HVP 的中位误差为 1.05%、最大误差为 6.16%。逐点梯度相对
`H_ee * (beta - 1)` 的最大非基线偏差为约 0.0913%。
KL loss 扫描在接近零处的相对误差可能较大；这些数值直接来自 CSV，未平滑。
原始 KL metadata 的 `normalization` 和 CSV 的 `measured_delta_mse` 名称
沿用了 L2 字样；绘图依据 `loss_fn=kl_div` 标注为 KL，原始数据保留不改。

重新生成：

```bash
python draw/hessian-3d-landscape/plot_method_validation_b.py
```

可用 `--data-dir` 和 `--kl-data-dir` 分别指定 L2、KL 输入；默认输出就在本目录。

## 历史说明（下文的 31 层覆盖、旧面板编号及未采集状态不代表当前图）

### 原始图说明

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

### 方案③：换成 KL 散度重新验证 HVP vs. 真实移除代价（尚未采集数据）

图 (b)"HVP vs. single removal"目前完全贴合 `y=x`，但这**不是一个需要真实
数据才能知道答案的问题**——只要 loss 用的是 L2/MSE，且模型输出对 `beta_e`
是线性的，`H_ee/2` 就必然精确等于真实移除代价，这是代数恒等式（见上面
"为什么图 (a)-(b) 是正确性检查"一节），不需要再采一遍数据去验证。真正
没有先验答案、需要真实实验才知道结果的，是**换一个不是精确二次的 loss**
之后，这个"贴合 `y=x`"的结论还成不成立。

**先说清楚为什么不是随便换个 loss 都算数**。仓库里 `compute_block_loss`
（`src/calibration/helpers/helpers.py:18`）已经支持四种 `loss_fn`：`l2`、
`rel_l2`、`cosine`、`kl_div`。这四种里，`rel_l2` **不值得采**，原因可以
直接从公式看出来：

```text
rel_l2 = diff2 / (base2 + eps)，
diff2 = (pred - target)^2 之和,     # 和 l2 一样，对 beta_e 严格二次
base2 = target^2 之和。              # target 是 beta_e=1 时的输出，不依赖 beta_e
```

`base2` 是一个和 `beta_e` 完全无关的常数，所以 `rel_l2` 只是把 `diff2`
（跟 l2 一样严格二次）除以一个常数——**整条曲线还是严格二次的，只是被
重新缩放了**，不需要跑实验就能预判：采出来的结果还是会精确落在 `y=x`
上，跟现在的图一样，看不出新东西。

`kl_div` 则真的不一样，原因是 `pred` 在算 KL 之前先过了一遍 `softmax`：

```text
pred_logprob = log_softmax(pred)
teacher_prob = softmax(target)
kl = kl_div(pred_logprob, teacher_prob)
```

`softmax` 是非线性的。正文第 348 行附近已经证明：L2 情形下"the model-
nonlinearity term in the Hessian vanishes identically"，前提正好是"输出
对 `beta_e` 是仿射的、loss 是二次的"这两条同时成立；一旦中间插入
`softmax` 这个非线性环节，这个论证就不再适用了——**`H_ee/2` 在 KL 散度
下只是真实移除代价的一个近似，不再是精确恒等式，近似得有多好、在哪些
expert 上近似得差，这些都是需要真实数据才能回答的问题**，这正是这一版
图的空白，值得单独采一份数据填上。

有两件事在换 loss 之后仍然成立，不需要重新证明：

1. `identity_gradient`（`beta_e=1` 处的一阶导数）在 KL 下应该仍然精确为
   零。理由和 loss 类型无关：只要 `beta_e=1` 时 `pred=target`，KL 散度和
   L2 一样，在两个分布完全相同的地方取到全局最小值 0，极值点的一阶必要
   条件同样适用。如果实际采出来发现不是 0，说明实现有 bug，而不是说明
   这个结论只对 L2 成立。
2. "Gram/energy 闭式解"（图 (a) 里的 `expert_output_energy`）**在 KL 下
   没有对应版本，不需要为它采数据**。现在的闭式解能成立，是因为 L2 loss
   下 Hessian 恰好等于 routed contribution 的 Gram 矩阵；KL 散度的 Hessian
   是 softmax 之后的 Fisher 信息量，不再是一个简单的向量内积，没有理由
   假设它还等于 `||f_e(x)||^2`。图 (a) 这一对比换 loss 之后没有意义，只
   需要重新验证图 (b) 这一对。

**需要采集的数据**：`collect_method_validation_b.py` 里目前有两处硬编码
了 `loss_fn="l2"`（`_identity_point_gradient_and_hessian_diag` 的调用点，
以及 beta sweep 内层循环），需要改成一个可配置参数（比如环境变量
`VALIDATION_LOSS_FN`，默认仍为 `l2` 保持向后兼容），跑一遍
`VALIDATION_LOSS_FN=kl_div` 采集完整流程（第 0 层全部 128 个 expert 的
`hvp_hessian_half`/`single_expert_ablation`/`identity_gradient`，加上
P10/P50/P90 三个 expert 的完整 beta sweep，包括 `local_gradient_at_beta`
——方案①的逐点局部一阶在 KL 下同样是一个开放问题，可以顺带一起验证）。

**输出目录和 metadata 要求**：

- KL 数据必须导出到一个新目录（比如
  `draw/hessian-3d-landscape/data_kl/`），不能和现有 L2
  数据混在同一批 CSV 里——两者的数值量纲完全不同（KL 散度和 per-token
  MSE 不是同一个单位），混在一起会被误读成同一批可比较的数字。
- `method_validation_B_metadata.json` 里目前没有记录用了哪种
  `loss_fn`，需要补一个 `loss_fn` 字段（现有 L2 数据可以事后补写
  `"loss_fn": "l2"`），否则以后区分不了哪份数据对应哪种 loss。

**画法建议**：复用现成的 `plot_method_validation_b.py`，用
`--data-dir` 指向新目录即可画出对应的 (a)-(f)；如果要把 L2 和 KL 的
"HVP vs. single removal"结果放到同一张图里直接对比，需要新写一个小脚本，
把两份 `method_validation_B_scores.csv` 的 `hvp_hessian_half` /
`single_expert_ablation` 放到同一个 log-log 散点图里，用颜色区分
loss 类型，而不是像现在这样跨两个独立的图分别看 Pearson/Spearman。

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
