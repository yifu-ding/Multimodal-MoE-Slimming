# EP4 部署效率主表：batch=512 正式测速

2026-09-22 用户授权，直接开始正式测速（先中文交流）。

## 范围

Qwen3-VL-30B-A3B-Instruct、InternVL3.5-30B-A3B-HF、Kimi-VL-A3B-Instruct 三个模型，
每个模型：p=0（default 基线）+ p=0.3、p=0.5 各 4 种方法（padded/multi_kernel/
single_width/cross_layer），batch=512，1 warmup + 4 measured batch，prefill-only
（`max_new_tokens=1`，input tokens/s 即目标表格的 Tput. 定义）。

Mistral-Small-4-119B-2603、Qwen3-VL-235B-A22B-Instruct-FP8（Phase C）不在本轮范围内，
两个模型此前从未验证过能否在 vLLM 下加载，留待本轮完成后单独处理。

## 前置条件（已满足）

- `single_width` 策略、p=0 default 基线路径、随机均衡 plan 生成器已实现并测试。
- 三个模型的 padded/multi_kernel/single_width/cross_layer 已在 batch=8 冒烟验证通过。
- 过程中修复了两个潜在 bug：`validate_ep4_plan` 的设备不匹配、`multi_kernel` 对
  plain per-expert-key checkpoint（InternVL/Kimi）缺失的 loader 重定向。
- 新写了 Kimi/DeepSeek-v2 的 multi_kernel adapter（shared-expert 只归属一个 group，
  避免重复计算）。

## 执行

Dispatcher: `.automation/efficiency_batch512_main/pipeline.sh`，日志见同目录
`campaign.log`。幂等，可安全重复执行（复用 `sweep_status.tsv` 断点续跑）。

## 产出位置

`artifacts/efficiency_figure/<qwen3_gqa_ep4|internvl3_5_gqa_ep4|kimi_vl_gqa_ep4>/batch512_main/prune_{0,30,50}/`

## 待办

- [ ] 三模型 27 个运行点全部完成。
- [ ] 汇总成目标 LaTeX 表格 + Markdown 表格。
- [ ] 视情况补 decode（完整生成，非 prefill-only）吞吐数据。
- [ ] Phase C（Mistral-119B、Qwen3-VL-235B）：先做最小加载冒烟测试。

## 2026-09-22 第一轮中止并重跑（TOKENS_PER_REQUEST_BUDGET 512→256）

第一轮（`campaign_v1_budget512_aborted.log`）用 `TOKENS_PER_REQUEST_BUDGET=512`
（即 `max_num_batched_tokens=262144`）。Qwen3-VL-30B 全部 9 点成功，但 InternVL3.5
的 default/padded/multi_kernel/single_width 四点全部因
`Available KV cache memory: -2.59 GiB`（差 2.59 GiB）失败，只有 cross_layer 成功
（印证了 cross_layer 的 non-KV 显存优势）。

排查：`GPU_MEMORY_UTILIZATION` 从 0.90 提到 0.93 只勉强够 default 用（富余仅
0.26 GiB）；提到 0.95 反而在 multi_kernel 上榨干显存 margin，触发真正的
CUDA OOM（远比不调时更差）。根因是 `TOKENS_PER_REQUEST_BUDGET=512` 对 GQA
这种图文问答任务的调度预算过度保守——降到 256（`max_num_batched_tokens=131072`）
后 InternVL multi_kernel 的 `Available KV cache memory` 从 -2.59 GiB 变为
42.84 GiB，且不改变实际计算量（只是调度器预留空间，只要不小于数据集里最长
请求的真实 token 数就不影响测量语义）。

为保证三模型评测口径完全一致（不能不同模型用不同的调度预算/显存利用率，
否则引入测量之外的混杂变量），已清空全部已产出数据（含 Qwen 已成功的 9 点），
统一用 `GPU_MEMORY_UTILIZATION=0.90`、`TOKENS_PER_REQUEST_BUDGET=256` 从头
重跑全部 27 点。同时修复了 `pipeline.sh` 自身的一个日志 bug：
`echo "... exit=$?"` 里的 `$?` 被同一条命令里的 `$(date -Is)` 覆盖，导致
日志一直显示 `exit=0`（不影响实际执行，只影响监控可信度，已修复为提前保存
退出码到变量）。
