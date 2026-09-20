# 五模型推理效率实验（2026-09-17 用户授权）

## 下载重试更新（2026-09-18）

2026-09-18 08:01：用户要求更换国内源，权重下载切换为 ModelScope 同名仓库。已通过 API 比对 Mistral 3 个、Qwen 24 个权重分片，SHA256 与大小均与原 HF 固定清单相同；两模型均完成 1 MiB Range 试读（HTTP 206）。仅重启 `maes-download-mistral` 与 `maes-download-qwen235`，保留全部完整/partial 文件，未触碰评测会话。脚本按 ModelScope 文件 revision 请求，下载后仍按原 HF SHA256 验证；小配置文件继续使用原固定 HF 版本，避免混用不同配置。3 项重试测试和语法检查通过。切源恢复时先重新校验已有完整分片，再继续下载；试读速度仅约 0.05–0.10 MB/s，不能承诺高速完成。

用户授权增加失败重试。`scripts/download_model_mirror.py` 对 SSL 断连、连接/传输超时、429 和部分 5xx 自动进行每文件最多 12 次尝试（`DOWNLOAD_MAX_ATTEMPTS` 可设 1–100）；间隔 30/60/120/240/480 秒后上限 600 秒。不重试证书校验失败、权限/不存在、磁盘写入、校验损坏等永久错误，耗尽次数写 ATTENTION 并保留文件。清单固定 commit、完整分片跳过、partial 断点继续、完整 partial 校验后直接提升；权重必须有 SHA256 元数据。状态持续追加至每个模型目录 `download_status.md`，命令输出仍为 `download.log`。独立 tmux 和已有 flock 防止重复下载；不占 GPU，不调用 Codex 推理。3 项 CPU 测试覆盖 SSL 恢复、错误分类/次数和退避上限，已通过。

## 执行顺序与边界

先结束当前 Qwen p50 MVBench 补跑，继续 Debug/补齐 InternVL p50 MMVet、Kimi MMVet Judge、Qwen VideoMME/VideoMMMU。失败不得原样重试；沿用每配置三轮无新增产物的上限，无法解决则保留缺项，转入效率实验，禁止无限等待。当前效率自动接力尚未实现，不能宣称已经启动。

## 模型与协议

- Kimi-VL-A3B-Instruct、Qwen3-VL-30B-A3B-Instruct、InternVL3_5-30B-A3B-HF、mistralai/Mistral-Small-4-119B-2603、Qwen/Qwen3-VL-235B-A22B-Instruct-FP8。
- 四张 GPU，30% 和 50% 两档剪枝。以固定随机种子生成均衡的异构宽度方案，不使用 accuracy 的重要性剪枝结果决定各方法的方案；四种方法共享同一专家/通道集合，仅改变布局和执行方式。
- 四个非零档位可按模型调整；Qwen 30B 参考 384/512/640/768。剪枝率按实际保留参数/通道预算核实，不等同于仅删除专家数；如四档均衡与目标预算不相容，需要调整档位或显式加入 width=0，记录实际比例，不默改定义。
- Padding：同层专家补零到最大宽度，使用 fused MoE。
- Multi-kernel：按连续 expert ID 分到四个 EP rank；每卡每宽度一组 fused MoE。
- Single-width：所有层固定宽度到固定 rank 的映射，一卡一档，不逐层轮换。
- Ours：EP4 逐层 rearrange，跨层平衡各 rank 累计负载。
- 先 GQA 真实图像；同模型同数据同精度下四方案使用相同 batch（256/512 候选），共同可行 batch 才作主比较；OOM 单独记录，不能各取不同 batch 假装同口径。
- warmup 不计时，关闭答案缓存复用；记录 requests/s、input tokens/s、output tokens/s，区分 prefill-only 与完整 decode。旧数据是 max_new_tokens=1 的 prefill-only，不得冒充完整生成吞吐。
- 内存为 non-KV peak：逐卡峰值减该卡实际 KV 分配，主表报告四卡最大值，同时保留各卡数值。non-KV 不等于纯权重内存。
- 新模型需验证量化与 EP4/fused MoE 兼容性，不能直接沿用 BF16 slicing 路径到 FP8 权重。

## 已有代码

- `docs/qwen3_ep4_efficiency_plotting.md`：已有 Qwen GQA 均衡档位和 non-KV 指标说明。
- `scripts/run_qwen_ep4_batch_sweep.sh`、`scripts/collect_qwen_ep4_batch_sweep.py`：旧三策略测速、汇总。
- `draw/ep4-compare-method3/README.md`：静态宽度绑卡、ID 分配、跨层重排的示意说明；不能据此声称四策略都已支持运行。
- 当前测速入口仅 padded/multi_kernel/cross_layer，需要补 single-width 并验证五模型支持。

## 下载

`df -h` 显示 `/home/data3` 空闲约 3.7 TB。目标目录 `/home/data3/dyf/models/`；下载脚本 `scripts/download_efficiency_models.sh`，仅下载 HF 格式，排除重复 consolidated 权重及图片。

2026-09-17 23:47：两个后台下载已尝试，但均因 HF 元数据检查 `FileMetadataError: Distant resource does not seem to be on huggingface.co` 退出；尚未下载完成，也没有活跃下载可报告。日志在各模型目标目录的 download.log，需先诊断 endpoint/proxy/响应头后恢复。

2026-09-18：按用户要求改用 `https://hf-mirror.com`。镜像存在重定向，HF CLI 仍遇元数据错误，改为 `scripts/download_model_mirror.py`：GET 读取文件清单并固定 commit，curl 断点续传，验证大小及 LFS SHA256，成功后重命名。后台会话 `maes-download-mistral`、`maes-download-qwen235` 已恢复，现场确认两个模型配置文件下载并验证成功；大权重仍待完成，不宣称模型已可用。无需 GPU，不改评测队列。
