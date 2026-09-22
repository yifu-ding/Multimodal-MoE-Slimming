# 实验 C：E0 算术下界早停验证

> [!IMPORTANT]
> **2026-09-22 新规划。E0 必须最先执行。**
> 旧的 475-case 补点方案作废。在 E0 出结果之前不运行后续 A/B/C，也不修改论文方法、Algorithm 2、正文或表格结构。

## 研究问题

跨层 placement 最小化各 EP rank 累计权重负载的极差：

```text
DeltaPhi = max_u Phi_u - min_u Phi_u
```

当前宽度档 `384/512/640/768` 都是 128 的倍数，因此每个 group load、每个 `Phi_u` 和 `DeltaPhi` 都是 128 的整数倍。令：

```text
total_quanta = sum_u Phi_u / 128
floor = 0    if total_quanta % m == 0
floor = 128  otherwise
```

当 `total_quanta % m != 0` 时，所有 rank 负载不可能相等，而任何非零极差至少为 128，所以 `floor=128` 是严格下界。该下界严格强于当前 MILP 的 LP 松弛下界 0。

当 `total_quanta % m == 0` 时，`floor=0` 只表示完美均衡在算术上没有被整除条件排除；它不是组合可行性的充分证明。m=12/16 是否能达到 0 仍由实验判断。

在 bf16、gate/up/down 三矩阵的口径下，128 个 `width x expert` 单位对应 1.5 MiB。

## 立即重标

旧 depth sweep 中有 29 个 m=4、`floor=128` 的实例，MILP incumbent 全部等于 128：

- 11 个已由 HiGHS 在预算内证明最优。
- 18 个以 time limit 结束，但 incumbent 已命中严格算术下界。
- 这 18 个不再标为 `OOT`，改为“已证明最优（算术下界）”。

逐行重标表由 E0 manifest 自动生成：

```text
results/placement_e0/relabelled_depth_cases.md
```

真正需要观察的是达到 floor 的速度，以及 m=12/16 这两个 `floor=0` 的大 m case 在预算内是否能找到完美解。

## E0 实例集

总计 31 个 case：

| 组别 | 数量 | 来源 | floor | 作用 |
| --- | ---: | --- | ---: | --- |
| m=4 depth windows | 29 | 旧 `depth_sweep.json` 中 L>=8 且算术下界为 128 的全部 p=0.3/p=0.5 窗口 | 128 | 测不可完美均衡实例多久命中严格下界；包含旧 18 个 time-limit case |
| m=12 full depth | 1 | 旧 p=0.5 m-sweep 的同一 plan 和 group 构造 | 0 | 大 m、算术允许完美均衡的反例 |
| m=16 full depth | 1 | 旧 p=0.5 m-sweep 的同一 plan 和 group 构造 | 0 | 大 m、算术允许完美均衡的反例 |

manifest 必须从旧结果和原 plan SHA256 重建并校验，不能手工重新定义窗口。m=12/16 必须复用 `_build_layer_placement_groups(..., ep_size=m)`，不能把四列 width counts 直接传给一个不存在的 rank-count 参数。

## Arm A：递增预算

对每个 case 冷启动 MILP，并依次使用：

```text
time_limit in {0.5, 1, 2, 5, 10, 30, 60} seconds
```

每次记录 incumbent、HiGHS status、dual bound、node count 和实际耗时。第一次返回 `incumbent == floor` 的预算是 time-to-floor 的上界估计，后续更大预算不再运行。

SciPy `milp` 没有 incumbent callback，所以 Arm A 不能在一次调用内部命中 floor 的瞬间中止；它只能在每个预算调用返回后检查。因此 Arm A 测的是离散预算上的 time-to-floor 上界。

## Arm B：下界可行性

把以下约束加入 assignment MILP：

```text
max_load - min_load <= floor
```

目标函数改为常数 0。求解器找到任一可行点即可结束；若该点满足算术 floor，它自动具有全局最优性证明，不需要 HiGHS 再抬 LP dual bound。

- 若 HiGHS 明确返回 infeasible，将 target 提高一个 quantum 后重试一次。
- 若只是在 300 秒预算内没有 incumbent，记为 OOT，不能当作 infeasible，也不能擅自提高 floor。
- Arm B 总预算保持 300 秒。

Arm B 才实现“命中 floor 后立即返回”的真实早停。Arm A 用于测预算阶梯下的可观测上界，两者都保留。

## 停止规则

```text
floor = 0 if total_quanta % m == 0 else quantum

incumbent reaches floor -> return with arithmetic optimality proof
budget B is exhausted   -> return incumbent if any and report OOT/unproven
```

两个条件缺一不可。不能只等待 floor，因为 floor 可能组合上不可达，或在现实预算内找不到；也不能只等固定时限，因为命中严格下界后继续搜索没有意义。

## CPU 并行与计时

本机为 16 个物理核 / 32 个逻辑核、单 NUMA。E0 使用四个并行 shard，每个 case 固定四个互不重叠的物理核：

| Shard | CPU affinity | 物理核 |
| ---: | --- | --- |
| 0 | `0,2,4,6` | 0--3 |
| 1 | `8,10,12,14` | 4--7 |
| 2 | `16,18,20,22` | 8--11 |
| 3 | `24,26,28,30` | 12--15 |

每个 shard 内串行跑自己的 case；四个 shard 并行。每次算法计时都继承同一组四核 affinity，并设置：

```text
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
```

不使用同一物理核的两个 SMT sibling，避免 case 之间争用执行单元。输出记录实际 affinity 和线程环境。

## 产物与完成条件

```text
results/placement_e0/manifest.json
results/placement_e0/relabelled_depth_cases.md
results/placement_e0/cases/<case-id>.json
results/placement_e0/summary.json
results/placement_e0/summary.md
```

每个 case 原子落盘，可断点恢复。只有 31 个 case 都存在、manifest digest 一致且汇总可解析时，E0 才算完成。

核心报告字段：

- arithmetic quantum、total quanta、floor；
- 旧 greedy spread/耗时和旧 MILP spread/status；
- Arm A 每个预算的 incumbent 与首次命中 floor 的预算；
- Arm B target、incumbent、耗时、状态和最优性证明来源；
- CPU affinity、依赖版本和 Git commit。

## 决策门

| E0 结果 | 方法决策 |
| --- | --- |
| 大多数部署规模 case 在几秒内命中 floor | placement 改为算术下界 + MILP 可行性早停；greedy 只作为大 m 或预算耗尽 fallback |
| 需要几十秒且实例间不稳定 | greedy 保持默认；正文改为“精确求解需要实例相关预算” |
| Arm B 显著快于 Arm A | 将“先算下界，再解有界可行性”作为候选方法主体 |

若 E0 支持方法切换，后续需要重新审视：

| 位置 | 当前叙述 | 候选修改 |
| --- | --- | --- |
| Algorithm 2 | greedy sweep | floor + MILP feasibility early stop，预算耗尽时 fallback |
| Section 4.3 Motivation | exact solving infeasible | deployment-scale exact placement can be certified early |
| Analysis | greedy 为默认 | 算术下界让精确求解实用 |
| Table 6 | greedy vs fixed-budget MILP | greedy vs early-stop MILP，或只报告新方法 |
| Appendix neighborhood | 支撑 pairwise swap | 若 greedy 不再是主体则删除或降级为 fallback 分析 |

> [!WARNING]
> 在 E0 完成并审阅之前，不修改论文，不启动旧补点方案，也不运行后续 A/B/C。

## E0 实测结果（2026-09-22 15:42 CST）

E0 已完成 31/31，提交到仓库的完整结果见 `artifacts/placement-e0/summary.md`，运行时原始产物保留在 `results/placement_e0/`。

- m=4 的 29 个 case 在 Arm A 中全部于 5 秒档内命中 floor=128，中位 time-to-floor 上界为 1 秒。
- Arm B 在 31/31 case 命中 floor，全体中位 0.311 秒。
- m=12/16 的 floor=0 并非反例：Arm B 分别用 2.984/1.305 秒找到 spread=0；Arm A 在 60 秒档内均未命中。
- 18 个旧 OOT 行已可依据 incumbent=128 和算术下界重标为已证明最优。

结果对应决策门的第一和第三条：秒级命中下界，且 Arm B 显著快于标准优化模型。候选方法应切换为“先算算术下界，再解有界可行性 MILP”；论文尚未修改，待本结果审阅后再执行方法叙述变更。
