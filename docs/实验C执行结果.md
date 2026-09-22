# 实验 C：Greedy 与 MILP

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

> [!IMPORTANT]
> **SUPPLEMENT QUEUED（2026-09-22 CST）**
> 新版补充实验已排在当前 Qwen3-VL p=0.3 视频流水线之后。范围包括 5 个 MILP 时限 pilot、三模型六份 plan 的 stride=4 滑窗、p=0.3 多模型 m 补点，以及 m=5/6/7 greedy 邻域直接对照。只有当前视频流水线通过完成校验后才会启动。
