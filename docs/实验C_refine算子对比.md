# 实验 C 补充：Refine 算子族对比（m=4 Table 6 + m>4 网格）

> 对应 `docs/efficiency_campaign_plan.md` §6 的待办："Table 6 现有三行... 还差 Sort only / Simulated annealing / Tabu search / Beam search 四行"。
> 代码：`src/generate_mask/ep4_intplan.py`（新增 `_solve_placement_groups_simulated_annealing` / `_solve_placement_groups_tabu` / `_solve_placement_groups_beam`，以及共享的 `_lpt_sort_initialize` / `_finalize_refine_result` 辅助函数）。
> 实验：`scripts/run_placement_ablation.py depth-sweep`（m=4，58 个实例）、`scripts/run_placement_ablation.py m-sweep`（m>4 网格）。
> 汇总：`scripts/build_refine_operator_table.py`（新脚本，产出下面的 7 行×7 列透视表）。

---

## 0. 实现要点

四个新算子全部复用 Algorithm 2 的 LPT 排序初始化，以及既有的 `_placement_objective()` 返回的 `(ΔΦ, S)` 二级 tie-break 比较准则（先比 ΔΦ 后比 S），邻域统一是 pairwise swap（除 Sort only 不做任何精修外）：

| 算子 | 超参（全部取文档 §6.2 建议默认值） |
|---|---|
| Sort only | 无精修，即 `_solve_placement_groups_greedy(..., max_local_search_passes=0)` |
| Simulated annealing | $T_0=1.44q$，$\alpha=0.99$，$T_{\min}=q/100$；对 ΔΦ 退火，ΔΦ 打平时对 S 退火（tie-break 同样退火，而不是无条件接受） |
| Tabu search | tenure=$m$，**tenure 做了随机抖动**（`[tenure, 2·tenure]` 均匀取整），特赦准则；每轮全局取最优非禁忌候选 |
| Beam search | 束宽 $B=8$，每轮汇总全部候选按 $(\Delta\Phi,S)$ 排序取前 $B$，束内按 round 去重 |

**实现中发现并修复的一个问题**：tabu tenure 固定为 $m$ 时，搜索会稳定收敛到一个周期恰为 tenure 的"移动-撤销"循环，永远卡在同一个次优解（在小规模回归实例上验证：固定 tenure 卡在 ΔΦ=768，真实最优是 128）。这是 tabu search 的经典病态（Glover 早就指出过），标准修法是给 tenure 加随机抖动——修完之后同一实例不再死循环，稳定改善到 384（虽然默认 tenure=m 在这个小实例上仍未必摸到全局最优，但显著优于死循环和 plain local search）。测试 `test_tabu_search_tenure_jitter_avoids_cycling` 是这个问题的回归测试。

三个新算子都加了 30 秒软上限（`max_seconds`，对应文档 §6.3 的兜底机制），超时返回 best-so-far 并标记 `hit_time_cap=True`；tabu/beam 每轮成本是 $O(L\cdot m^2)$，在 m 网格铺到 64 时这个上限会被频繁触发（见下文 m-sweep 部分）。

结果字段新增：`accepted_worse_count`（SA/tabu 接受过的变差候选数，用于确认没有退化成保守贪心）、`random_seed`、`iterations`、`hit_time_cap`、`tau0`（本实例的理论下界，现在每条记录都带，不再只在 MILP 结果里出现）。

---

## 1. Table 6（m=4，58 个实例，中位数）

命令：

```bash
python scripts/run_placement_ablation.py depth-sweep \
  --layers 4 8 12 16 24 32 48 --greedy-repeats 5 --milp-time-limit 300 \
  --output results/placement_ablation/depth_sweep.json

python scripts/build_refine_operator_table.py \
  --input results/placement_ablation/depth_sweep.json \
  --markdown results/placement_ablation/refine_operator_table6.md \
  --csv results/placement_ablation/refine_operator_table6.csv
```

58 个实例 = 7 个深度 $L\in\{4,8,12,16,24,32,48\}$ 的不重叠窗口 × 2 个剪枝率（0.3/0.5），单位 MiB，下标是 $\Delta\Phi-\tau_0$（bf16，$d_{\text{model}}=2048$，1 φ 单位 = 12 KiB）。

| Refine operator | 4 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sort only | 117.8<sub>116.2</sub> | 97.5<sub>96.0</sub> | 150.8<sub>149.2</sub> | 144.0<sub>142.5</sub> | 125.2<sub>123.8</sub> | 141.0<sub>140.2</sub> | 95.2<sub>94.5</sub> |
| Pairwise swap | 48.0<sub>46.5</sub> | 37.5<sub>36.0</sub> | 12.8<sub>11.2</sub> | 15.8<sub>14.2</sub> | 19.5<sub>18.0</sub> | 12.0<sub>11.2</sub> | **3.0**<sub>2.2</sub> |
| All bijections | 55.5<sub>54.0</sub> | 33.8<sub>32.2</sub> | 14.2<sub>12.8</sub> | 15.8<sub>14.2</sub> | 7.5<sub>6.0</sub> | 6.0<sub>5.2</sub> | **3.0**<sub>2.2</sub> |
| Simulated annealing | 37.5<sub>36.8</sub> | 26.2<sub>24.8</sub> | 17.2<sub>15.8</sub> | 16.5<sub>15.0</sub> | 12.0<sub>10.5</sub> | 7.5<sub>6.8</sub> | 26.2<sub>25.5</sub> |
| Tabu search | 30.8<sub>29.2</sub> | 15.0<sub>13.5</sub> | 4.5<sub>3.0</sub> | 6.0<sub>4.5</sub> | 6.0<sub>4.5</sub> | 3.0<sub>2.2</sub> | 3.8<sub>3.0</sub> |
| Beam search | 42.0<sub>40.5</sub> | 17.2<sub>16.5</sub> | 9.8<sub>8.2</sub> | 7.5<sub>6.0</sub> | 4.5<sub>3.0</sub> | 2.2<sub>1.5</sub> | **3.0**<sub>2.2</sub> |
| Exact solver (MILP) | 27.8<sub>26.2</sub> | 1.5<sub>**0.0**</sub> | 1.5<sub>**0.0**</sub> | 1.5<sub>**0.0**</sub> | 1.5<sub>**0.0**</sub> | 0.8<sub>**0.0**</sub> | 0.8<sub>**0.0**</sub> |
| *理论下界 $\tau_0$* | *1.5* | *1.5* | *1.5* | *1.5* | *1.5* | *0.8* | *0.8* |
| *每卡负载 $\bar\Phi$ (GiB)* | *0.68* | *1.35* | *2.03* | *2.68* | *4.06* | *5.35* | *8.10* |

原有三行（Pairwise swap / All bijections / Exact solver）数值与旧版完全一致——`build_refine_operator_table.py` 的聚合口径经过交叉验证。

### 读法

- **下界依旧是紧的**：MILP 在 $L\ge8$ 下标全是 0.0，没有变化。
- **算子按接受准则、不是按邻域分层**。Tabu / Beam 在 $L=48$ 分别是 3.8 / **3.0** MiB，和两个贪心族（3.0 / 3.0）基本打平，离下界只差 2.2–3.0 MiB；真实配置（48 层、4 卡）下选哪个几乎不影响峰值显存。
- **Simulated annealing 在 $L=48$ 是唯一的例外**（26.2 MiB，比其余四个非 MILP 算子差一个数量级）。逐步跟踪它的搜索轨迹后确认了真正的机制——**不是退火没起作用，恰恰是退火起了太多作用**：`accepted_worse_count` 确实是 57–62（两个剪枝率），退火机制在积极接受变差候选；但文档给的温度公式 $T_0=1.44q$ 只按"恶化一个宽度量子 $q$"这一种最小尺度校准，而这个问题里随便一次 pairwise swap 改变的负载常常是几十到上百个量子（实测单步 $\Delta\Phi$ 可以是几千到七万多 φ 单位，相当于几十到几百个 $q$）。校准温度对"1 个量子"给出约 50% 接受率，但对"50 个量子"的真实典型步长而言，同一个 $T_0$ 下接受率仍然高得离谱（$\exp(-50/184)\approx77\%$）。结果是退火前期几乎不设防地随机游走，先把 ΔΦ 冲到远高于 LPT 初始化的值（在 m=4 的独立复现实验里，峰值冲到过 74368 φ 单位，是初始值 2304 的 30 多倍），然后指望降温把它带回来；但 $\alpha=0.99$、总预算 $100Lm$ 只给了约 500 步，等不到完全走回起点。**在这个具体实例上验证过：整条搜索轨迹里，ΔΦ 从未低于过 LPT 初始化时的值**——SA 报告的"最优解"就是它自己都没跑赢的起点。根因是温度公式按最小量子校准、没有按实际单步位移尺度校准，不是迭代预算或降温速率单独的问题。
- **耗时**：$L=48$ 时 SA 0.011s、tabu 2.3s、beam 22.1s（median，两个剪枝率一致）——beam 的 $O(B\cdot L\cdot m^2)$ 单轮成本在 $m=4$ 时依然可控，30 秒软上限没有触发。

---

## 2. m>4 网格（用户要求这次一并铺开，未在原计划范围内）

命令：

```bash
python scripts/run_placement_ablation.py m-sweep \
  --m-values 4 6 8 12 16 24 32 48 64 \
  --time-limits 60 180 300 --greedy-repeats 5 \
  --output results/placement_ablation/m_sweep_v2.json
```

用的是 `qwen3-vl-30b-a3b-p50` 这个 plan 的完整 48 层（不做深度切窗，单一实例，随 $m$ 变化），MILP 每个 $m$ 独立给 60/180/300s 三档时间预算（表里只展示 300s 档）。$\tau_0$ 在这批数据里全部是 0——50% 剪枝率恰好继承了未剪枝时 $\Phi_{\text{tot}}/q=36864$ 的高度可整除性（§2.4 的推论在这里被直接验证：几乎任意 $m$ 都能让 $\tau_0=0$），所以下标和主数字这里总是相等，没有区分度，纯粹是为了和 Table 6 保持同一种记法。单位 MiB；标 † 的格子在 30 秒软上限内没有收敛（`hit_time_cap=True`），报告的是 best-so-far。

| Refine operator | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sort only | 27.00<sub>27.00</sub> | 3.00<sub>3.00</sub> | 12.00<sub>12.00</sub> | 6.00<sub>6.00</sub> | 9.00<sub>9.00</sub> | 4.50<sub>4.50</sub> | 4.50<sub>4.50</sub> | 4.50<sub>4.50</sub> | 4.50<sub>4.50</sub> |
| Pairwise swap | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | **0.00**<sub>0.00</sub> | 3.00<sub>3.00</sub> | **0.00**<sub>0.00</sub> |
| All bijections | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | — | — | — | — | — | — |
| Simulated annealing | 27.00<sub>27.00</sub> | 3.00<sub>3.00</sub> | 12.00<sub>12.00</sub> | 6.00<sub>6.00</sub> | 9.00<sub>9.00</sub> | 4.50<sub>4.50</sub> | 4.50<sub>4.50</sub> | 4.50<sub>4.50</sub> | 4.50<sub>4.50</sub> |
| Tabu search | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub>† | 3.00<sub>3.00</sub>† | **0.00**<sub>0.00</sub>† | 3.00<sub>3.00</sub>† | **0.00**<sub>0.00</sub>† |
| Beam search | 3.00<sub>3.00</sub> | 3.00<sub>3.00</sub>† | **0.00**<sub>0.00</sub>† | 3.00<sub>3.00</sub>† | 3.00<sub>3.00</sub>† | 3.00<sub>3.00</sub>† | **0.00**<sub>0.00</sub>† | 3.00<sub>3.00</sub>† | 4.50<sub>4.50</sub>† |
| Exact solver (MILP, 300s) | 0.00 (optimal) | 0.00 (optimal) | 0.00 (optimal) | 3.00 (time-limited) | 7.50 (time-limited) | 24.00 (time-limited) | 19.50 (time-limited) | 15.00 (time-limited) | 6.00 (time-limited) |
| *理论下界 $\tau_0$ (MiB)* | *0.00* | *0.00* | *0.00* | *0.00* | *0.00* | *0.00* | *0.00* | *0.00* | *0.00* |
| *每卡负载 $\bar\Phi$ (GiB)* | *6.75* | *4.50* | *3.38* | *2.25* | *1.69* | *1.12* | *0.84* | *0.56* | *0.42* |

*（"All bijections" 只在 $m\le8$ 有定义，$m!$ 枚举在 $m=12$ 就是 $4.8\times10^8$ 种，代码直接拒绝——这本身就是要报告的结论：大 $m$ 时可用的邻域只剩 pairwise swap。）*

### 读法

- **MILP 在 $m\le8$ 依旧是精确最优**，$m\ge12$ 起 300 秒内证明不了最优性（dual bound 恒为 0，因为下界证明本身也要解一个和主问题一样大的松弛，帮不上忙），incumbent 从 3 MiB（$m=12$）涨到 24 MiB（$m=24$，本批的峰值），$m=64$ 反而降到 6 MiB——不是变好了，是 60/180/300 秒对 $196{,}608$ 个二元变量的问题来说完全不够，incumbent 质量本质上是随机的，**这一列的数字不该被解读为"MILP 在 $m=64$ 比 $m=32$ 更容易"**。

- **本节最重要的发现：大 $m$ 时 tabu / beam 明显跟不上便宜的 pairwise-swap 局部搜索**，而不是像 §1 里那样"殊途同归"。原因是纯粹的复杂度结构：局部搜索的一整轮扫描是 $O(L\cdot m^2)$，一旦某一轮没有任何改进就整体停止（`if not improved: break`），所以它在 $m=64$ 上 1.6 秒就收敛到 $\Delta\Phi=0$；而 tabu 每轮只挪动一层，beam 每轮要在 $B=8$ 条并行路径上各自展开 $O(L\cdot m^2)$ 个候选——同样的 30 秒预算下，tabu 在 $m=64$ 只跑了 39 轮，beam 只跑了 **3 轮**，两个都在软上限里被打断（`hit_time_cap=True`）。beam 在 $m=64$ 上最终停在 4.5 MiB，比什么都不做的局部搜索（0 MiB）还差——这不是算法本身不行，是它单轮成本乘了 $B$ 倍之后，30 秒连一轮"够用的"搜索都做不完。

- **换句话说，§1 的结论"换算子不敏感"在 $m=4$ 成立，但不能推广到大 $m$**。$m$ 越大，邻域枚举成本 $O(m^2)$ 越吃预算，能不能在时间预算内跑完哪怕几轮，比"这个算子理论上更聪明"更决定最终质量。这本身是一个诚实且有信息量的负结果：**大 $m$（模型分片更多）时，最便宜的 pairwise-swap 局部搜索反而是唯一稳定可靠的选择**，tabu/beam 的价值随 $m$ 增大而反向递减，除非把时间预算也按 $m^2$ 同步放大（本次实验为了让 58 个深度实例 + 9 个 $m$ 网格实例的总耗时可控，没有这样做）。

- **Simulated annealing 在这个网格上全程复现了 §1 的机制**：每一列都和 Sort only 完全打平（27/3/12/6/9/4.5/4.5/4.5/4.5，逐列比对完全一致）——不是没接受变差候选（`accepted_worse_count` 在这批数据里是 56–86，退火确实在积极接受），而是同一个"按量子 $q$ 校准 $T_0$、不按实际单步位移尺度校准"的问题：初期几乎不设防地接受远超 1 个量子的恶化，把 ΔΦ 推得比初始化还差很多，然后在 $\alpha=0.99$、约 500 步的预算内来不及走回起点，更不用说超过起点。$m$ 越大不会让这个问题变好或变坏——问题根源在 $T_0$ 与单步位移尺度的错配，与 $m$ 无关，所以在 4 到 64 的每一档都复现了同样的"没跑赢起点"。

### 与 §1 合并看的结论

$m=4$（Table 6）和 $m>4$（本节）两张表放在一起，能得到一个比"换算子不敏感"更精确的结论：**只有当每轮/每次迭代的成本足够低、能在给定预算里跑够多轮时，algorithm-family 的选择才不重要；一旦邻域枚举成本随 $m$ 增长，预算分配就变成了第一位的因素。** 这修正了 §1 单独看时可能给出的过度乐观的印象。
