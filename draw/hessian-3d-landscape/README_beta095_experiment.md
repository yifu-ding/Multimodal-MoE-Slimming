# 非恒等工作点实验：完整 student 的 expert 输出统一缩放至 0.95

状态：服务器端实现与采集说明，尚未实现新采集参数、尚无本实验实测结果。
本文不改变已有 beta=1 的 L2 / KL 实验及其数据。

## 实验问题

保持 student 完整、不删除任何 expert，把被测层所有 routed expert 的输出
统一乘以 `beta_work=0.95`。teacher 使用同一输入下完整、未缩放的输出。
在该工作点实测一阶梯度、Hessian 对角元以及逐 expert 删除代价，检验：

1. 正常工作点是否产生足够高于数值噪声的非零梯度？
2. 一阶分数与真实删除敏感度是否发生排序反转？
3. 加入二阶项后，整体排序和删除代价预测是否改善？
4. 是否存在真实删除代价较高、但一阶重要性分数较低的 expert 对？

不预设一定存在反例或二阶必胜。若一阶排序同样好，应如实报告。

## 工作点：所有 experts 同时缩放，而非逐个单独缩放

模型：`Qwen/Qwen3-VL-30B-A3B-Instruct`。
首轮使用第 0 层、与原实验相同 frozen GQA manifest 的前 32/256 个样本。
复制完整 block，验证精度为 FP32；其余模型状态、输入及 score mask 保持一致。

定义绝对缩放参数向量 `b`，作用在 routed expert contribution 上：

- teacher：`b = ones(num_experts)`，输出记为 `y_teacher`，固定并 detach。
- student 基线：`b = full(num_experts, 0.95)`，输出记为 `y_work`。
- 删除 expert e：其他坐标保持 `0.95`，只把 `b[e]` 改成 `0`。
- 扫描 expert e：其他坐标保持 `0.95`，只改变 `b[e]`。

缩放位置沿用 `patch_expert_output_alpha_vector`；不要缩放整个 block 输出、
residual stream、router logits 或 router weights。不重新归一化剩余路由权重。
每次扰动前恢复整条向量至 `0.95`，避免删除累积。

**不要只把被测 expert 单独缩放到 0.95、其他 experts 保持 1。**
那样每个 expert 都在不同工作点被测；L2 下梯度仍与该 expert 能量严格成比例，
无法检验这里关心的共同残差下的排序差异。

仅被测层做缩放；模型其他层保持原样。沿用原采集器捕获的 teacher block 输入，
让 teacher 和 student 使用同一输入。这是 block 重构实验，不是端到端任务 loss 实验。

## loss 与真实敏感度

分别跑 `l2` 和 `kl_div`，分开存储。沿用现有 `compute_block_loss` 的定义，
并在 metadata 记录实际归一化方式。当前 KL 是 block hidden 输出上的 softmax KL，
不是最终词表分布的 KL，也不是 GQA 答案交叉熵。

令 `L(b) = loss(student(b), y_teacher)`，所有量按同一 score-token 数归一化。

```text
L_work      = 真实 forward 得到的 L(0.95, ..., 0.95)
g_e         = 工作点真实 autograd 一阶导数 dL/db_e
H_ee        = 工作点真实二阶 autograd / HVP 对角元
L_remove_e  = 真实 forward 得到的 L(b_e=0, b_others=0.95)
D_e         = L_remove_e - L_work
```

`D_e` 是本实验敏感度的唯一 ground truth，不能由梯度、Hessian 或 Gram 公式填充。
L2 也必须逐 expert 真实删除 forward，不能沿用旧采集器的 energy 消融捷径。
保存绝对 loss 和差值，允许 `D_e` 为负；负值表示删除减少了当前重构误差，不要截断。

对每个活跃 expert 采集以上全部数据，不先筛选“好看的”9个。

## 公平比较的分数

导数相对于绝对参数 `b_e` 计算。删除步长是 `delta = -0.95`，不是 `-1`：

```text
first_order_signed = -0.95 * g_e
first_order_abs    = abs(0.95 * g_e)
second_order      = -0.95 * g_e + 0.5 * 0.95**2 * H_ee
hessian_only      = 0.5 * 0.95**2 * H_ee
ground_truth      = D_e
```

这些是用真实导数计算的打分，不能称为真实删除 loss；真实删除值必须来自独立 forward。
有非零残差时，完整二阶打分包含一阶项；单独报告 Hessian-only 的表现，
不要把完整二阶分数的优势归因成 Hessian-only 的优势。

`first_order_signed` 是一阶 Taylor 删除代价；`first_order_abs` 是幅度重要性基线。
二者分别评估排序，避免把负梯度数值较小误解成一阶重要性较低。

如果实现改用相对参数 `a`（`b=0.95*a`，工作点 `a=1`），导数会变换；
必须记录 parameterization 并同步修改公式。首选直接使用绝对 `b`，避免歧义。

## 为什么统一缩放可能产生排序差异

在 fixed-router L2 下，用 `f_e` 表示未缩放 routed contribution，
共同残差是 `r = -0.05 * sum_j f_j`。一阶导数取决于 `r` 与 `f_e` 的内积，
Hessian 对角元取决于 `f_e` 自身的能量。因此不同 expert 的梯度不必与自身能量
成固定比例；其他 expert contribution 的交叉项会影响梯度。

如果这些贡献恰好近似正交或交叉项结构相似，一阶与敏感度仍可能高度相关。
统一缩放不保证出现反例，也不保证每个 expert 的梯度都显著非零。
上述关系只用于解释和数值验收，不用于生成任何实测值。

## 采集流程

每个 batch：

1. 所有缩放参数为 1，真实 forward 保存 `y_teacher` 与参考 router indices/weights。
2. 所有参数设为 0.95，真实 forward 保存 `L_work`，真实求导得到全部活跃 experts
   的 `g_e`、`H_ee`。沿用现有分块二阶自动微分和 OOM 降块机制。
3. 对每个活跃 expert 独立设 `b_e=0`，真实 forward 得到 `L_remove_e`。
   每次结束都恢复到工作点。
4. 累积 loss、梯度和 Hessian 的 token-sum，最后除以总 score-token 数。
   不对不同 token 数的 batch 直接等权平均。
5. 验证 teacher、工作点、删除和扫描的 router indices/weights 一致。

第一遍完成全部 expert 分数后，按下节固定规则选择扫描对象；第二遍复用相同样本，
采集每个扫描点的真实 loss 和真实局部梯度。

建议绝对 beta 网格：

```text
0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.25, 1.5
```

每个扫描点记录 `L(b_e=t, others=0.95)`、相对 `L_work` 的差值和该点 autograd 梯度。
注意：单个 `b_e=1` 时其他 experts 仍为 0.95，loss 不应被强制为零。

## 总体统计、9个展示对象及反例

对全部活跃 experts，分别评估四种分数与 `D_e` 的 Spearman、Kendall、
pairwise inversion rate，以及预测值的 MAE / RMSE。
相对误差仅作为补充，接近零的删除代价会放大相对误差。
报告实际参与比较的 expert 数、负删除代价数和梯度数值范围。

Pairwise inversion 使用真实 `D_e` 的高低顺序；预测打平单独统计，不算正确排序。
真实差异在预先记录的数值容差内的 expert 对不参与反转率分母。
容差应通过重复 forward / FP64 loss 累积等数值检查确定并记录，不能按反例数量调节。

按 `D_e` 升序划分低、中、高三个等人数区间（余数用固定 array_split 规则分配），
每组取组内 25%、50%、75% 分位附近的三个不同 expert，平局按 expert ID 排序。
分别对 L2、KL 选择，保存 ID 和选择规则。这9个用于代表性曲线展示。

另外导出全部稳健反例对：

```text
D_i > D_j + truth_tolerance
first_order_i < first_order_j - score_tolerance
second_order_i > second_order_j + score_tolerance
```

分别为 signed / abs 一阶分数导出，不混用。额外标注 Hessian-only 是否也排对。
按真实删除代价差降序列出，明确是反例展示，不能替代全体统计。
没有反例时输出空表和计数0。若反例对象不在9个代表 expert 中，额外采集其扫描，
不要悄悄替换代表性对象。

## 输出目录与字段

所有本实验文件统一放到：

```text
draw/hessian-3d-landscape/beta095/
  l2/       # 原始 pt、CSV、metadata、统计 JSON、日志和后续图
  kl_div/   # 相同结构，与 L2 分开
```

`scores.csv` 至少包含：

```text
layer, expert, beta_work, loss_fn, num_score_tokens, active_batch_count,
base_loss, removal_loss, true_removal_delta, gradient_at_work,
hessian_diag_at_work, first_order_signed, first_order_abs,
second_order, hessian_only
```

`beta_curves.csv` 至少包含：

```text
layer, expert, selection_group, selection_reason, beta_work, beta,
measured_loss, measured_delta_loss, local_gradient_at_beta
```

metadata 保存模型路径、代码 commit、manifest 路径及 SHA256、样本选择规则、样本数、
batch size、层号、dtype、loss 定义及归一化、`beta_work=0.95`、
`parameterization=absolute_expert_scale`、`background=all_experts_scaled`、
teacher 未缩放、fixed-router 检查、扫描网格、数值容差及选择规则。
保存 batch 级汇总，便于定位数值不稳定，但两个 batch 不能当成充分的统计置信度。

## 服务器端需要修改的代码

这不是目前旧命令加 `--betas 0.95` 就能运行的实验：旧参数只控制展示扫描，
不会改变求导工作点，也不会改变其他 expert 的背景缩放。

建议新增独立采集器，复用 `src/calibration/collect_method_validation_b.py` 的模型加载、
manifest、alpha patch、HVP 和 router 检查逻辑，避免改变已有实验的语义。
新接口需支持 `--beta-work 0.95`、`--loss-fn`，以及上述全体评分与二次扫描流程。
该接口尚未实现，本文不提供伪装成可直接执行的新命令。

移植时重点检查：

- `_identity_point_gradient_and_hessian_diag`：改为在传入工作向量处求导。
- `_true_ablation_via_forward`：所有恢复操作改为 `fill_(beta_work)`；L2/KL 均使用它。
- `collect_layer`：分离完整 teacher target 和缩放 student 基线。
- beta sweep：背景保持工作向量，禁止重新定义 teacher 或恢复为全1。
- 导出器：使用新字段和独立 schema；不复用 `identity_gradient` 等旧字段名。
- 旧绘图器假设恒等基线为零，不能直接用于新数据，需要后续新增对应绘图逻辑。

## 验收与结论边界

先用少量样本 smoke test，再完整采集32个样本：

- teacher target 在所有探针间固定；工作点 loss、梯度由实测报告，不强制其非零。
- beta=0 扫描值与对应真实删除 forward 一致，beta=0.95 差值接近0。
- 抽查若干 expert，在0.95附近用多个小步长中心差分验证梯度/Hessian；
  KL 的小 loss 注意 FP32 消减误差，必要时采用 FP64 loss 运算核验。
- fixed-router L2 下，完整二阶删除打分应与实测删除代价在数值精度内一致。
  KL 不要求精确一致，记录真实误差。不得以理论值替换异常测量点。
- 所有比较使用同一工作点、teacher、样本和归一化；两种 loss 数据不混合。
- 若预期排序反例没有出现，保留结果，不通过换 beta 或挑 expert 隐瞒失败。

首轮结果回答的是“完整模型在统一0.95缩放工作点下的 block 重构敏感度排序”。
不能直接扩大为未缩放原模型的排序优势或最终任务性能优势。
如需确认反例稳定性，可在后续独立 GQA 样本上验证已选 expert 对，
不把用于筛选的同32个样本称为独立验证。
