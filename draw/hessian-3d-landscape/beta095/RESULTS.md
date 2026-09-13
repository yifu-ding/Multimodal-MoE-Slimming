# beta=0.95 非零一阶的真实排序反例

2026-09-13，第0层全部128个 active expert，frozen GQA 前32样本，4096个
score tokens。teacher 为原始输出；一阶在所有 expert 同时乘0.95处实测。
Hessian 复用已有 beta=1 autograd/HVP；ground truth 为背景全1下新采集的
逐 expert 独立删除 forward，不由 Hessian 或 energy 公式填充。

## 全体结果

| 指标 | L2 | KL |
| --- | ---: | ---: |
| 一阶 signed/abs Spearman | 0.964750 | 0.972080 |
| 一阶反转对数 / 8128 | 637 (7.8371%) | 550 (6.7667%) |
| Hessian Spearman | 1.000000 | 0.999994 |
| Hessian 反转对数 / 8128 | 0 | 1 |
| Hessian 删除预测中位相对误差 | 6.459e-8 | 7.498e-5 |
| Hessian 删除预测最大相对误差 | 6.603e-7 | 0.0141683 (1.4168%) |
| 满足 Hessian 精度条件的反例对 | 637 | 550 |

本批全体梯度均为负，因此 `-0.95*g` 与 `abs(0.95*g)` 完全相同。
CSV 仍分开记录两种 baseline，所以反例表行数分别是1274和1100，
不能把它们误报成1274和1100个不同的 expert 对。
L2 全体一阶梯度范围 `[-1.947772e-7, -2.251604e-9]`，
KL 为 `[-6.929103e-8, -9.536290e-10]`。

## 可直接画图的 L2 反例

若希望一阶反向差距也明显，优先考虑 **89/53、92/103、124/98**；它们的真实删除
差距分别为59.61%、31.40%、44.34%，一阶分数反而分别低33.43%、40.62%、30.34%。
这是采集后的额外画图筛选；详见 [实验README候选表](../README_beta095_experiment.md)
及 [按双向差距排序的全部候选](l2/plot_candidates.csv)。
下面保留最初按绝对删除差选定的三对，不事后替换已报告的固定规则展示对象。

每行按真实敏感度从高到低写 expert ID；一阶分数次序全部相反。
`first` 表示 `abs(0.95*g(0.95))`。

| expert 对（高 / 低） | 真实删除 loss（高 / 低） | 一阶分数（高 / 低） | Hessian 相对误差（高 / 低） |
| --- | --- | --- | --- |
| 19 / 98 | 4.230080e-7 / 2.478819e-7 | 6.438304e-8 / 7.257282e-8 | 2.928e-8 / 1.338e-8 |
| 60 / 98 | 4.179946e-7 / 2.478819e-7 | 7.147262e-8 / 7.257282e-8 | 1.770e-7 / 1.338e-8 |
| 13 / 57 | 4.049963e-7 / 2.365224e-7 | 5.939747e-8 / 5.993304e-8 | 1.841e-9 / 1.363e-8 |

例如 expert 19 的删除 loss 比98高70.65%，一阶分数却低11.28%。
这是一阶给出相反排序的真实反例，且两者的纯 Hessian 预测都与删除 loss
在数值精度内重合。其余两对一阶差距更小，不能描述为同等强烈的反转。

KL 的三个固定规则展示对为13/57、13/98、19/57；Hessian 相对误差均小于0.24%。
KL 用FP64 loss 算术重新执行删除 forward，旧 FP32 loss 数据保留不变；
不能把这里的误差统计与旧数据统计混为一次采集。

## 图片与原始数据

- L2：[全体排序及反例 PNG](l2/counterexample_comparison.png) / [PDF](l2/counterexample_comparison.pdf)
- L2：[反例独立曲线 PNG](l2/counterexample_curves.png) / [PDF](l2/counterexample_curves.pdf)
- L2：[低中高各3个代表 PNG](l2/representative_curves.png) / [PDF](l2/representative_curves.pdf)
- KL：[全体排序及反例 PNG](kl_div/counterexample_comparison.png) / [PDF](kl_div/counterexample_comparison.pdf)
- KL：[反例独立曲线 PNG](kl_div/counterexample_curves.png) / [PDF](kl_div/counterexample_curves.pdf)
- KL：[低中高各3个代表 PNG](kl_div/representative_curves.png) / [PDF](kl_div/representative_curves.pdf)

两种 loss 各有 `scores.csv`、`counterexamples.csv`、`statistics.json`、
`beta_curves.csv`、`metadata.json`、`validation.json`、`raw.pt` 和 `progress.pt`。
L2 扫描14个对象、280个点；KL 扫描12个对象、240个点。
每个对象都扫描背景1与背景0.95；主图只把背景1的 loss 与 beta=1 Hessian 对齐。
`raw.pt` 在全体评分完成时保存，`metadata.json` 的 `complete=true` 表示后续扫描也完成。

## 验收

- 5个 CPU 单元测试通过，覆盖非正交 contribution 反例、完整背景恢复、固定
  teacher、L2/KL 中心差分、signed/abs 区分、Hessian 数值匹配门槛、并列与空反例。
- 真模型单样本 L2 smoke 通过，不与32样本 Hessian 产生混合排序。
- 两种 loss 的全体数据重算统计、token 归一化、CSV 与 batch 原始张量一致。
- 全1处最大梯度：L2 为0；KL 为2.493e-22，仅有浮点残差。
- 全0.95处真实梯度的中心差分最大相对偏差：L2 为1.040e-5，KL 为7.853e-6。
- 重复删除/工作点 forward 和重复梯度差异均为0；所有扫描删除端点误差为0，
  所有扫描基线梯度与全体采集梯度差异为0；全部路由一致性检查通过。

这里证明的是这批数据存在跨工作点的一阶排序反例，不是 beta=0.95 处的纯二阶
删除预测，也不是独立样本验证或最终 GQA 任务性能结论。
