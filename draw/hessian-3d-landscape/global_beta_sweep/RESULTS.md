# 第0层全部expert同步beta扫描：真实一阶梯度

2026-09-13已完成。模型为Qwen3-VL-30B-A3B-Instruct，沿用原GQA manifest
前32样本、batch size 16、2批、4096个score tokens。L2和hidden-state softmax KL
分别真实采集；主点严格为 `[0, 0.25, 0.5, 0.75, 0.95]`，另有beta=1检查点。

## 实际操作

对同一层的独立128维叶子向量 `alpha` 同时赋值：

```python
with torch.no_grad():
    alpha.fill_(beta)
loss = loss_sum(block_output, fixed_original_teacher, mask, loss_fn)
gradient = torch.autograd.grad(loss, alpha)[0]  # shape [128]
```

每个主点只做一次forward/backward获得128个偏导，不逐个expert改变背景后求导，
也不用一个广播标量的总梯度替代128个偏导。teacher始终固定在全1输出。
只有有限差分验证临时单独扰动expert，这些点仅存于raw.pt的验证记录。
其他层、residual、router权重保持原始状态，所有路由一致性检查通过。
copied block为FP32，loss算术和落盘累积为FP64。

先跨batch/token累加signed gradient，再取绝对值；百分比为
`100 * abs(g_e(beta)) / abs(g_e(0))`，不额外乘beta。

## 百分比实测结果

| 全局beta | L2跨expert最小值至最大值（%） | KL跨expert最小值至最大值（%） | KL均值（%） |
| --- | --- | --- | ---: |
| 0 | 100.000000至100.000000 | 100.000000至100.000000 | 100.000000 |
| 0.25 | 74.999989至75.000009 | 73.499459至75.060387 | 74.825474 |
| 0.5 | 49.999983至50.000011 | 48.155157至50.080895 | 49.779136 |
| 0.75 | 24.999979至25.000014 | 23.724346至25.060956 | 24.842672 |
| 0.95 | 4.999984至5.000016 | 4.697263至5.015503 | 4.961739 |

L2各expert的变化百分比相同，最大理论偏差仅 **2.071e-5个百分点**，
不能将这些浮点偏差解释成非线性。KL最大偏差为 **1.844843个百分点**，
出现在expert 114、beta=0.5；其他较明显对象包括expert 106、16、57。
这些数值是测量结果，不是按100%、75%、50%、25%、5%生成的曲线。

L2这一结果与此前“梯度排序不等于Hessian对角元的删除排序”不矛盾：
全局beta方向的一阶变化对应 `(H * ones)_e`，不是 `H_ee`。
本次没有补采Hessian或删除loss。

## 线性与数值验收

每个expert使用实际坐标 `[0, .25, .5, .75, .95, 1]` 拟合signed gradient。
线性拟合与 `g_e(beta)-(1-beta)*g_e(0)` 两种残差均有独立记录。

| 检查项 | L2 | KL |
| --- | ---: | ---: |
| 原始梯度表行数 | 768 | 768 |
| beta=0有效分母数 | 128/128 | 128/128 |
| 最小线性拟合R² | 0.9999999999999422 | 0.9995905108273636 |
| 超过数值阈值的拟合残差expert数 | 0 | 108 |
| 最大端点缩放残差（梯度单位） | 1.421e-13 | 2.500e-8 |
| 预先规定规则得到的数值阈值tau | 1.486e-11 | 5.580e-12 |
| 重复梯度/重复loss差异 | 0 / 0 | 0 / 0 |
| 最大中心差分相对误差 | 3.020e-5 | 2.935e-5 |
| beta=1最大绝对梯度 | 0 | 2.493e-22 |
| beta=1 loss | 0 | 9.212e-19 |
| beta=0.95与旧gradient_at_work最大差异 | 0 | 0 |

tau使用规格中的 `max(8*max_repeat_error, 32*eps_float32*max(abs(g(0))))`。
KL偏差超过这一数值阈值；108的计数是数值残差检查，不是统计显著性检验。
中心差分检查固定expert 0、64、127，背景beta=0.5，步长0.005和0.01。

3个新增CPU测试通过，覆盖独立向量同步求导、先累加signed再取绝对值、
不等token数归一化、零分母保留/屏蔽和KL非线性实测。
另从raw.pt重新计算全部CSV、线性统计和数值验收，并检查采集源码hash；L2/KL均通过。

## 文件

- [主图PNG](gradient_ratios.png) / [PDF](gradient_ratios.pdf)：按expert ID排列，五条主beta曲线。
- [偏差图PNG](ratio_deviations.png) / [PDF](ratio_deviations.pdf)：相对100(1-beta)的偏差，
  灰带表示数值阈值换算的量级，不是统计置信区间；L2/KL纵轴尺度不同并明确标注。
- [L2梯度](l2/gradients.csv)、[KL梯度](kl_div/gradients.csv)：保留所有128个expert及beta=1检查点。
- 每种loss另有 `losses.csv`、`metadata.json`、`validation.json`、`linearity.json`、
  `linearity_residuals.csv`、`raw.pt`。raw.pt含batch原始梯度、loss、token数、
  样本ID和顺序、重复测量、有限差分；metadata含manifest/源码hash与complete标记。

## 复现

在仓库根目录，使用已安装模型依赖的Python，L2采集命令：

```bash
python -m src.calibration.collect_global_beta_sweep \
  --scores /home/data/dyf/MARS-results/storage/scores/qwen3-vl-30b-a3b_mixed-gqa-256-seed42-rel_l2-0913-132047/scores.pt \
  --reference-dir draw/hessian-3d-landscape/beta095/l2 \
  --output-dir draw/hessian-3d-landscape/global_beta_sweep/l2 \
  --loss-fn l2
```

KL将两个目录末尾的l2和loss-fn改成kl_div。若manifest换了服务器路径，可指定
`--selection-manifest`，内容hash必须相同。已有raw.pt的输出目录拒绝覆盖；
重复实验请指定新目录。实际本次使用maes环境，GPU 1采L2、GPU 2采KL。

```bash
python -m scripts.check_global_beta_sweep --data-dir draw/hessian-3d-landscape/global_beta_sweep/l2
python -m scripts.check_global_beta_sweep --data-dir draw/hessian-3d-landscape/global_beta_sweep/kl_div
python draw/hessian-3d-landscape/plot_global_beta_sweep.py
python -m unittest discover -s tests -p 'test_global_beta_sweep.py'
```
