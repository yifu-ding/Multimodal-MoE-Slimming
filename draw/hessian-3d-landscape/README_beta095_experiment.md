# beta=0.95 一阶信号与 beta=1 Hessian 的 expert 排序反例

状态：2026-09-13 已完成第0层 L2/KL 各32个真实 GQA 样本采集、曲线和数值验收。
结果、反例表和图片见 [beta095/RESULTS.md](beta095/RESULTS.md)。

## 核心发现：L2 已找到真实排序反例

**128 个 expert 中，一阶排序出现 637 对反转，占全部 8128 对的 7.84%；
已有 beta=1 Hessian 对新增真实删除 forward 的排序完全一致，最大相对误差
仅为 6.6e-7。** 一阶 Spearman 为0.964750，Hessian Spearman 为1.000000。

这些一阶值是在所有 student expert 同时缩放至0.95、teacher 保持原始输出时
通过真实 autograd 得到的非零梯度；Hessian 和排序 ground truth 对应原模型
beta=1。不是用 beta=0.95 的一阶加二阶混合分数冒充纯 Hessian。

一阶 signed 与 absolute 在本批数据中相同，因为128个梯度均为负。
637是不同 expert 对的数量，不是637个 expert，也不将 signed/abs 重复计数。

## 画图候选：真实更敏感，一阶反而更低

下面每对均为第0层，ID从0开始。写作 `A / B` 表示真实删除 loss 为 `A > B`，
但一阶幅度分数为 `A < B`。百分比都以B为分母：
`真实增加 = D_A/D_B - 1`，`一阶降低 = 1 - |g_A|/|g_B|`。

| expert 对 A / B | A真实删除 loss 更高 | A一阶分数反而更低 | 两者Hessian最大相对误差 | 选择理由 |
| --- | ---: | ---: | ---: | --- |
| **89 / 53** | **59.61%** | **33.43%** | 3.04e-7 | 两个方向都有明显差距，优先推荐 |
| **92 / 103** | **31.40%** | **40.62%** | 1.79e-7 | 一阶反向差距尤其明显 |
| **109 / 103** | **54.36%** | **31.05%** | 1.79e-7 | 两个方向都明显，可替换上一对 |
| **124 / 98** | **44.34%** | **30.34%** | 1.57e-8 | 两个方向都明显，Hessian非常贴合 |
| 4 / 53 | 83.51% | 19.04% | 3.04e-7 | 真实删除差距较大 |
| 19 / 98 | 70.65% | 11.28% | 2.93e-8 | 按绝对删除差选出的第一对，已有完整曲线 |
| 47 / 43 | 122.47% | 4.34% | 1.04e-7 | 真实敏感度相差超过2倍，但一阶反向较弱 |

若只画三对且希望不重复 expert，建议 **89/53、92/103、124/98**。
原先按绝对删除差挑出的60/98、13/57，一阶反向幅度分别只有1.52%、0.89%，
因此不是展示“明显反转”的首选。

上述推荐是**采集完成后的画图筛选**，不改变全体统计和原先固定规则选出的9个
代表。637对中有68对同时满足“真实删除至少高20%、一阶分数至少低20%”。
完整候选表 [beta095/l2/plot_candidates.csv](beta095/l2/plot_candidates.csv)
按 `min(真实增加比例, 一阶降低比例)` 降序排列，保留原始数值和Hessian误差。
全体128个原始分数见 [beta095/l2/scores.csv](beta095/l2/scores.csv)。

这些候选的真实梯度、Hessian、独立删除 loss 均已采齐，可直接画分数/删除代价图。
完整 beta 曲线只对先前选出的代表和展示对象采集；候选表的
`both_full_curves_available` 标明一对是否已有两条完整扫描曲线，不能把缺失曲线
用 Hessian 公式生成后当作真实测量。

## 目的与比较口径

本实验为画图寻找真实 expert 反例：原模型的 Hessian 能准确匹配单 expert
删除 loss，而在统一轻微缩放后实际测得的非零一阶信号给出相反的敏感度排序。
不预设一定存在反例；全体排序统计和全部符合规则的反例一起输出。

两个工作点承担不同角色，不能混为一次 Taylor 展开：

| 数据 | 求值位置 | 用途 |
| --- | --- | --- |
| 真实 Hessian `H_ee(1)/2` | 所有 expert 的 beta=1 | 复用已有真实 autograd/HVP，预测原模型删除 loss |
| 真实删除代价 `D_e(1)` | 背景全1，仅 expert e 从1变0 | 本图唯一 ground truth；新增逐个独立 forward |
| 真实梯度 `g_e(0.95)` | 所有 expert 同时缩放到0.95 | 获得非零一阶敏感度代理，比较排序 |
| `D_e(0.95)` | 背景全0.95，仅 expert e 变0 | 单独保存的诊断量，不是主图 ground truth |

teacher 始终保持完整原始输出并 detach。仅被测层 routed contribution 缩放，
residual、router logits 和权重不缩放，不重新归一化路由。
每次扰动都恢复完整缩放向量，禁止删除累积。

**在 beta=1，一阶梯度理论上为零；在 beta=0.95，一阶项通常非零。**
后者不是纯 Hessian 就能完整预测删除 loss 的工作点：

```text
delta = -0.95
T2_at_work = -0.95 * g_e(0.95) + 0.5 * 0.95**2 * H_ee(0.95)
```

`T2_at_work` 包含一阶项，不能称为纯二阶。fixed-router L2 下该完整表达式仍然
精确，但这不是本实验想展示的比较；KL 下它通常只是近似。
本实验不重算 beta=0.95 Hessian，不用它替换已有 beta=1 Hessian。

主图打分为：

```text
first_order_signed       = -0.95 * g_e(0.95)
first_order_abs          = abs(0.95 * g_e(0.95))
hessian_half_at_identity = H_ee(1) / 2
ground_truth             = L(b_e=0, others=1) - L(ones)
```

一阶分数在这里是跨工作点的排序代理，不是对 `D_e(1)` 的一阶 Taylor 预测。
保留0.95这个共同正因子不会影响排序；不以一阶值的 MAE 宣称预测精度优劣。
只有 beta=1 的 Hessian 分数报告对删除 loss 的 MAE/RMSE/相对误差。
L2 是固定路由与仿射 expert 缩放下的精确二次情形；KL 必须报告实测误差，
不能称为数学上的精确恒等式。

## 数据与采集

模型 `Qwen/Qwen3-VL-30B-A3B-Instruct`，第0层，原 frozen GQA manifest
前32个样本、4096个 score tokens、batch size 16、FP32 copied block。
manifest SHA256 为 `24c3a5e250c33ca53df6addb82ee78de2efe94f1bff2487ee754e6f6b347d6d5`。
旧 Hessian 对应 `data/` 与 `data_kl/` CSV，原始 PT 在
`artifacts/method_validation_b_gqa_layer0{,_kl}/method_validation_b.pt`。

新采集器 `src/calibration/collect_beta095_counterexamples.py`：

1. 加载旧 Hessian PT，验证模型、loss、FP32、manifest 哈希和评分 token 总数。
2. 每个 batch 捕获原模型 block 输入，在 copied block 全1处测得并固定 teacher。
3. 在全1处实测梯度；在全0.95处做两次真实 forward/backward，记录梯度及重复误差。
4. 对全部128个 expert，在背景1和0.95下分别独立删除 forward。
   背景1重复测量，用于估计数值重复误差；L2 不使用 energy/Gram 捷径。
5. 对梯度绝对值最大的3个 expert，使用0.005、0.01中心差分步长核验一阶梯度。
6. 按 score-token sum 累积后统一归一化，不等权平均不同 token 数的 batch。
7. 对全部活跃 expert 做统计，再复用相同样本采集选定对象的真实 loss/局部梯度曲线。

所有 teacher、工作点、删除、中心差分和扫描 forward 都检查 router
indices/weights 完全一致。loss 使用 FP64 算术，block 保持 FP32。
L2 为 hidden 维 MSE；KL 为 block hidden softmax 分布的 KL，
不是最终词表 KL 或 GQA 答案交叉熵。FP64 仅改善 loss 数值运算，不合成测量值。

## 全体统计、代表与反例

分别计算 signed、abs 一阶与 Hessian 对 `D_e(1)` 的 Spearman、Kendall、
pairwise inversion rate。真实并列排除，预测并列单列，不算正确排序。
记录梯度范围、负删除数与参与比较的 expert 数；常数分数相关性输出 null。

数值容差固定为 `max(8 * 重复测量最大差, 32 * FP32 epsilon * 全体最大绝对值)`。
梯度的重复差同步乘0.95。该规则在采集前固定，不根据反例数量调节。

低、中、高各3个代表：按真实删除 loss 升序、expert ID 打破平局，使用
`array_split` 划分三组，各取组内 P25/P50/P75 最近的三个不同对象。

分别为 signed、abs 导出所有满足以下条件的 expert 对：

```text
D_i(1) > D_j(1) + truth_tolerance
first_i < first_j - first_tolerance
H_ii(1)/2 > H_jj(1)/2 + hessian_tolerance
两个 expert 的 Hessian 删除预测相对误差均 <= hessian_match_rtol
```

L2 默认 `hessian_match_rtol=1e-4`；KL 使用0.05，并明确这是5%内的近似匹配。
按真实删除差降序排列，expert ID 打破平局。
展示额外取前三个不同 expert 对；采集其端点扫描，不替换9个代表。
其余反例全部保留在 CSV，但不强制为每一个候选对象补扫曲线。
没有反例时保留空表和计数0。

曲线分别使用背景1和背景0.95，teacher 均为原始输出。绝对 beta 网格：
`0, 0.25, 0.5, 0.75, 0.9, 0.95, 1, 1.05, 1.25, 1.5`。
背景0.95时单个 expert 的 beta=1不是恒等点，不强制 loss 为零。
扫描 beta=0 必须与独立删除 forward 一致，beta=背景值时 delta loss 必须接近0。

## 运行与输出

在仓库根目录，用已安装模型依赖的 Python 运行：

```bash
python -m src.calibration.collect_beta095_counterexamples \
  --scores /home/data/dyf/MARS-results/storage/scores/qwen3-vl-30b-a3b_mixed-gqa-256-seed42-rel_l2-0913-132047/scores.pt \
  --identity-input artifacts/method_validation_b_gqa_layer0/method_validation_b.pt \
  --output-dir draw/hessian-3d-landscape/beta095/l2 --loss-fn l2
```

KL 改用 `_kl` reference 目录、`--loss-fn kl_div --hessian-match-rtol 0.05`，
输出目录改为 `beta095/kl_div`。
可先指定 `--smoke` 和独立输出目录，仅采1个样本；smoke 不把单样本新数据
与32样本旧 Hessian 拼接，不输出排序结论。已有 `raw.pt` 的目录拒绝覆盖。

输出含 `raw.pt`（真实 batch 张量）、`scores.csv`、`counterexamples.csv`、
`statistics.json`、`beta_curves.csv`、`metadata.json` 和中途保存的 `progress.pt`。
metadata 记录两种背景、归一化、参考 PT 哈希、代码版本/采集器哈希、
模型、manifest、样本数、token 数、路由检查及扫描对象。

```bash
python draw/hessian-3d-landscape/plot_beta095_counterexamples.py \
  --data-dir draw/hessian-3d-landscape/beta095/l2
python -m scripts.check_beta095_counterexamples \
  --data-dir draw/hessian-3d-landscape/beta095/l2
python -m scripts.select_beta095_plot_candidates \
  --data-dir draw/hessian-3d-landscape/beta095/l2
```

本实验支持的结论限于这批样本、这个 block 重构 loss 下的跨工作点排序反例。
筛选使用的32个样本不属于独立验证集，反例展示不替代全体统计。
