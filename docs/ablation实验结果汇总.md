# Ablation 实验结果汇总

## 实验口径

- 当前实验为 **Ablation Study**，模型为 `moonshotai/Kimi-VL-A3B-Instruct`。
- 本汇总包含 **方法1 + 方法2** 与 **方法1-only** 两组 Ablation，均**不包含方法3**；表中的 p = 0.3 与 p = 0.5 表示剪枝比例，不代表方法编号。
- 当前配置仅执行直接剪枝（Direct Mask），使用未调整的 26 层 channel masks。
- 剪枝后不做圆整，不做向上取整或向下取整规整，也不做 Rearrange。
- Mask 配置为 `align_inter = 0`、`min_per_expert = 0`，不进行额外宽度或 expert 数量调整。
- 已完成且有效的结果保持原全量口径，不重跑。自 2026-09-22 15:18 CST 起，所有尚未完成的测试改为确定性随机子集：`seed = 42`，目标数量为 `min(N, max(ceil(N/2), 500))`；总量不足 500 时使用全量。
- 组合 benchmark 按子任务分层抽样：MVBench 为 1900/3800（20 类各 100），VideoMMMU 为 500/900（167/167/166）。MME 保持 yes/no 配对；MMBench Judge 保持 circular 变体组完整，因此 p = 0.3/p = 0.5 实际分别为 2166/4329、2168/4329。

## 当前结果

更新时间：2026-09-23 09:52 CST。“方法1 + 方法2”与“方法1-only”两组流水线均已完成 28/28 个 benchmark 和 2/2 个 Judge 阶段；当前 Ablation 计划全部完成，没有运行中、超时或待重跑任务。

**Kimi-VL-A3B-Instruct**

| 剪枝比例 | 方法 | GQA | COCO CIDEr | TextVQA | ChartQA | MMStar | MMBench | MME-P | MME-C | RealWorldQA | MMVet Judge | Video-MME | LongVideoBench | EgoSchema | VideoMMMU（总体） | VideoMMMU Adaptation | VideoMMMU Comprehension | VideoMMMU Perception | MVBench |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p = 0.3 | Ablation（方法1+方法2，Direct Mask） | 62.0687 | 0.9002 | 86.3760 | 86.9600 | 61.8556 | 82.5601 | 1631.7338 | 513.9286 | 66.0131 | 62.3394（全量 218） | 63.4815 | 60.8080 | 73.20 | 49.78 | 36.00 | 43.33 | 70.00 | 60.4211 |
| p = 0.5 | Ablation（方法1+方法2，Direct Mask） | 60.3594 | 1.1260 | 81.4300 | 79.6400 | 56.2819 | 80.2405 | 1579.1719 | 506.0714 | 62.8758 | 50.5046（全量 218） | 60.0370（重跑） | 60.2090 | 70.60 | 48.00（重跑） | 37.00 | 39.00 | 68.00 | 59.5789（随机 1/2） |
| p = 0.3 | Ablation（方法1-only Router，Direct Mask，随机 1/2） | 61.7745 | 1.0056 | 84.0160 | 85.5200 | 62.0930 | 81.8636 | 1685.1061 | 539.3998 | 68.00 | 59.6789（全量 218） | 64.2222 | 62.6310 | 70.80 | 49.2423 | 35.9280 | 41.3170 | 70.4820 | 59.0000 |
| p = 0.5 | Ablation（方法1-only Router，Direct Mask，随机 1/2） | 60.0572 | 1.1224 | 79.8200 | 78.0000 | 58.5132 | 79.0349 | 1608.5569 | 529.1518 | 66.00 | 44.4037（全量 218） | 58.7407（重排后缓存恢复） | 60.3890 | 68.20 | 46.4407 | 33.5330 | 38.9220 | 66.8670 | 56.9474 |

### 本地 Judge 补充结果

下表是本地多模态 Judge 的独立结果。主表中的 MMBench 与 VideoMMMU 列是各 benchmark 原生指标，不能用这里的 Judge 分数覆盖。

| 剪枝比例 | 方法 | MMVet Judge | MMBench Judge | VideoMMMU Judge（总体） | Adaptation | Comprehension | Perception |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p = 0.3 | 方法1+方法2 | 62.3394（218/218） | 83.8210（随机 2166/4329） | 52.80（随机 500/900） | 42.1687（166） | 43.1138（167） | 73.0539（167） |
| p = 0.5 | 方法1+方法2 | 50.5046（218/218） | 81.7869（随机 2168/4329） | 51.40（随机 500/900） | 41.5663（166） | 44.3114（167） | 68.2635（167） |
| p = 0.3 | 方法1-only Router | 59.6789（218/218） | 82.5291（随机 2165） | 49.00（随机 500/900） | 35.3293（167） | 41.3174（167） | 70.4819（166） |
| p = 0.5 | 方法1-only Router | 44.4037（218/218） | 79.5341（随机 2165） | 46.40（随机 500/900） | 33.5329（167） | 38.3234（167） | 67.4699（166） |

## 状态说明

- `—（运行中）`：对应 benchmark 正在执行，尚未生成通过完整性校验的结果文件。
- `—（待执行）`：流水线尚未执行到该 benchmark。
- `—（待 Judge）`：推理结果已生成，但本地 Judge 尚未执行。
- `数值（重跑）`：首次执行发生异常，使用已持久化的响应缓存恢复后得到完整结果。
- `随机 1/2`：确定性随机子集，`seed=42`，并应用至少 500 条规则；与未标注的既有全量结果分开解释。
- 百分比类指标按百分制展示；COCO CIDEr 与 MME-P/MME-C 保持原始指标尺度。
- 方法1+方法2 的 VideoMMMU 总体为 Adaptation、Comprehension、Perception 三个各 300 样本子集的等权汇总；方法1-only 使用分层随机 500/900（167/167/166）。

## 执行异常记录

- p = 0.5 Video-MME：首次执行在 `1143/2700` 后触发 vLLM 多模态处理器缓存一致性断言：`Expected a cached item for mm_hash=...`，属于推理框架的 multimodal processor cache bug，不是模型指标异常。流水线保留已有响应并自动重跑，最终完整结果为 `60.0370`。
- p = 0.5 VideoMMMU：首次执行在 `134/900` 处因 `validation_Electronics_13.mp4` 的 decord EOF 解码问题卡住，主进程被终止并遗留 vLLM worker。清理残留进程后改用选择性 OpenCV 解码，从原缓存恢复并于 2026-09-22 10:18 CST 完成；最终总体结果为 `48.00`。这是视频解码/进程清理 bug，不是模型指标异常。
- p = 0.5 MVBench：全量执行到缓存 1111/3800 时，根据新的随机 1/2 实验口径主动暂停，不是报错或卡住。缓存 SQLite 完整性检查为 `ok`；随机子集恢复时复用其中重叠响应，只补缺失样本。
- 方法1-only p = 0.3 GQA：首次启动的两次 attempt 均在推理前因 `lmms_eval/caching/response_cache.py` 的补丁错位触发 `SyntaxError`，因此没有产生样本，也不计为实验失败。根因是 `patches/lmms_eval_response_cache_identity.patch` 首个 hunk 上下文不足，将常量插入了 `CACHE_RELEVANT_KEYS` 集合内部。已修复补丁及本地工作副本，通过 Python 编译、响应缓存单测和随机抽样单测，并于 2026-09-22 18:18 CST 恢复运行。
- 方法1-only p = 0.5 Video-MME：在 1063/1350 条响应后，为将 VideoMMMU 调整到队列最后而主动停止 dispatcher；这不是超时或推理 bug。新进程命中全部 1063 条缓存，只补剩余 287 条，并于 2026-09-23 00:54 CST 完成，结果为 58.7407。
- 方法1-only 运行中偶尔出现 vLLM usage telemetry 的 `cpuinfo JSONDecodeError`，仅终止遥测后台线程；模型推理、响应缓存和任务切换均持续正常，不影响结果。
- 方法1-only p = 0.5 VideoMMMU 在保存完整结果并正常返回退出码 0 后记录了一条 `Engine core proc ... died unexpectedly`。该信息发生在 vLLM worker teardown 阶段，结果文件、500 条样本和完成标记均已通过流水线校验，不是任务中断。

## 数据来源

- 实验 TODO：`.automation/kimi_direct/TODO.md`
- 实验流水线：`.automation/kimi_direct/pipeline.sh`
- Mask 构建配置：`scripts/build_kimi_mask_plan.py`
- p = 0.3 结果：`results/vllm_ours/kimi/direct-mask/p30-full/tasks/`
- p = 0.5 结果：`results/vllm_ours/kimi/direct-mask/p50-full/tasks/`
- 后续随机 1/2 结果：`results/vllm_ours/kimi/direct-mask/p30-random-half-seed42/`、`results/vllm_ours/kimi/direct-mask/p50-random-half-seed42/`
- 方法1-only 随机 1/2 结果：`results/vllm_ours/kimi/method1-router-direct/p30-random-half-seed42/`、`results/vllm_ours/kimi/method1-router-direct/p50-random-half-seed42/`

本表只填写已生成且通过当前流水线校验的结果；当前两组 Ablation 的所有单元格均已完成汇总。

## 方法1-only 配置（已完成）

- `方法1 + 方法2` 与 **方法1-only** 两组 campaign 均已完整结束。
- 方法1-only 配置保持 `modality_aware = 1`，将 `intra_layer_method` 从二阶 attribution 改为 Router 自身输出 `router`。
- 已检查 Kimi `scores.pt`：`router`、`usage`、`token_count_text`、`token_count_visual` 均覆盖全部 26 × 64 个 expert 且为非零；成品中没有 `usage_fillzero` 或 `router_fillzero` 键，因此采用 planner 原生支持的 `router`。
- 该配置仍使用独立的 text/visual channel scores 生成双模态 mask，Router 分数只替换 expert 内预算来源，因此 `modality_aware = 1` 仍然生效。
- 实验继续使用 p = 0.3、p = 0.5、相同 14 个 benchmark 与 Judge；保持 Direct Mask，不做圆整、上下取整规整或 Rearrange。
- 方法1-only 的所有 benchmark 使用随机 1/2、至少 500、`seed=42`；结果目录后缀为 `random-half-seed42`，不会与全量结果混用。
