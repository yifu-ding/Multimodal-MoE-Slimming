# 实验 C：Greedy 与 MILP

> [!IMPORTANT]
> **E0c PARTIAL SNAPSHOT（2026-09-22，90/144）**
> EP 规模与层深网格仍在后台运行。本次先归档 90/144 个已完成格点，供论文内容预审；这不是最终结果。

- 仓库快照：`artifacts/placement-grid/partial-summary.md`
- 已完成：90/144（62.5%）；已证明最优 64；预算结束但未认证 26；复用 E0/E0b 产物 12。
- 当前 p=0.3、L=48：m=4/12/16 已证明最优，m=32/48/64 在预算内未认证，m=6/8/24 尚待完成。
- 初步热力图显示 p=0.3 的 m<=16 大多可认证，m>=32 已频繁出现预算结束；p=0.5 尚主要覆盖浅层，暂不下最终结论。
- `time_limit=300` 是 HiGHS 求解器预算，不包含 Python 建模和部分预处理；最大已完成格点总墙钟为 876.543 秒，不能表述为“每格总墙钟最多 300 秒”。
- 完整 worker 继续运行，不因本次 Git 快照中断；论文正文仍未修改。

> [!IMPORTANT]
> **E0b DONE（2026-09-22 16:30 CST）**
> 论文修改前的补充验证已完成 31/31：24 个 L=4 retry case、5 个 m=4/floor=0 case，以及 m=32/64 扩展 case，全部获得最优性证明。

- 仓库汇总：`artifacts/placement-e0-followup/summary.md`
- 仓库逐 case 产物：`artifacts/placement-e0-followup/cases/`
- 运行时原始产物：`results/placement_e0_followup/`
- 自动化状态：`.automation/placement_e0_followup/STATUS.md`
- 总墙钟约 118.2 s，4 个 shard 使用四组互不重叠的 4 物理核 affinity。

| 补充项 | 结果 |
| --- | --- |
| L=4 retry | 24/24 触发，且第一个可行 target 与旧精确 MILP 最优值逐例相等；中位 0.470 s，最慢 3.133 s |
| m=4 floor=0 | 5/5 找到 spread=0，中位 0.252 s，最慢 0.344 s |
| m=32 | spread=0，1.184 s，算术下界证明最优 |
| m=64 | spread=0，110.746 s，算术下界证明最优 |

retry 分支不再是未测路径。当 floor 不可达时，算法逐 quantum 提高 target；只有 HiGHS 明确返回 infeasible 才继续，首个可行 target 在离散 quantum 上构成精确最优性证明。论文尚未修改。

> [!IMPORTANT]
> **E0 DONE（2026-09-22 15:42 CST）**
> 新的算术下界早停验证已完成 31/31 case，旧 475-case supplement 已作废。本轮只修改求解与实验代码，论文尚未修改，等待 E0 结果审阅后再决定方法叙述。

- 仓库汇总：`artifacts/placement-e0/summary.md`
- 仓库逐 case 产物：`artifacts/placement-e0/cases/`
- 旧结果重标：`artifacts/placement-e0/relabelled_depth_cases.md`
- 运行时原始产物：`results/placement_e0/`
- 自动化状态：`.automation/placement_e0/STATUS.md`
- CPU：4 个并行 shard，每个 shard 固定 4 个互不重叠的物理核；完整墙钟约 130 秒。

| 组别 | Arm A | Arm B |
| --- | --- | --- |
| m=4，29 个 `floor=128` case | 29/29 命中；0.5 s: 7，1 s: 16，2 s: 5，5 s: 1；中位上界 1 s | 29/29 命中；中位 0.297 s，最慢 1.755 s |
| m=12，`floor=0` | 60 s 内未命中，最后 incumbent=256 | 2.984 s 找到 spread=0，由算术下界证明全局最优 |
| m=16，`floor=0` | 60 s 内未命中，60 s incumbent=2304 | 1.305 s 找到 spread=0，由算术下界证明全局最优 |

E0 否定了“m=12/16 的 floor=0 组合上不可达”这一猜测：两者都可达，只是原始优化模型的搜索/证明路径在固定预算内效率较低。Arm B 31/31 命中下界，因此“算术下界 + 有界可行性 MILP”应作为方法切换的主要候选；Greedy 是否保留为预算耗尽 fallback 等待审阅决定。

> [!IMPORTANT]
> **DONE（2026-09-19 CST）**
> 旧的 3600 秒方案已在 14/39 处停止并保留 checkpoint。确认 Greedy 与 MILP 都固定第 0 层为 identity；新方案使用 300 秒上限、多窗口和加密 L，并固定在 CPU `24,26,28,30` 上完成。assignment MILP 保持现有实现，未加入两阶段求解或 warm start。

- TODO：[实验C-ablation-greedy-MILP.md](实验C-ablation-greedy-MILP.md)
- 旧方案结果：`results/placement_ablation/table6.json`
- 新方案结果：`results/placement_ablation/depth_sweep.json`、`results/placement_ablation/m_sweep_v2.json`
- Git 归档：`artifacts/placement-ablation/`
- 导出结果：同目录 `.csv` 与 `.md`
- 日志：`results/placement_ablation/pipeline.log`
- Worker：`maes-placement-ablation`
- Supervisor：`maes-placement-supervisor`
- 2026-09-18 17:25 CST: RUNNING | stage=placement-ablation completed=0/39 (0.0%) table6=0/18 m-sweep=0/21 | session=maes-placement-ablation.
- 2026-09-18 19:09 CST：重新排队。Worker 仅等待当前 Qwen VideoMME 进程结束；后续 Ours 自动化任务排除 CPU `24,26,28,30`，实验 C 固定使用这四个逻辑核并以 `nice -n 10` 运行。
- 2026-09-19 CST：根据首批结果停止旧方案。`fix_first_layer=True` 在 Greedy/MILP 两侧一致；MILP 实现保持不变。新 depth sweep 使用 `L={4,8,12,16,24,32,48}` 的不重叠连续窗口、MILP 300 秒单次求解；m sweep 时限改为 `60/180/300` 秒。

> [!IMPORTANT]
> **DONE (2026-09-19 18:37 CST)**
> 多窗口 depth sweep 与 M 扩展性扫描均已完成并通过产物检查。

> [!NOTE]
> **HISTORICAL / SUPERSEDED（2026-09-22 CST）**
> 下述 475-case supplement 排队记录已被 E0 新规划取代，从未启动。

- 2026-09-22 14:08 CST：补充实验仍为 0/475（pilot 0/10、depth 0/396、m-sweep 0/69）。Qwen VideoMMMU 当前 244/900，近期约 44 条/小时且持续增长，未卡住；其 Judge 完成后本流水线自动解除等待，预计 2026-09-23 05:00 CST 左右启动。当前 worker/supervisor 均存活，0/475 是有条件排队状态，不是执行停滞。
- 2026-09-22 15:00 CST：Qwen VideoMMMU 已按用户要求暂停在 283/900，缓存保留且不自动恢复。实验 C 旧补充流水线未启动；用户将提供新的规划，在新规划确认前保持暂停，不运行旧的 475-case 方案。
