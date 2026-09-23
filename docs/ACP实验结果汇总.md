# ACP 剪枝方案 —— vLLM EP4 精度评测结果汇总

- **状态**: 进行中（详见"数据完整性说明"）
- **快照时间**: 2026-09-23 21:31
- **分支**: `ep4_intplan`
- **实验目录**: `.automation/acp_eval/`（`pipeline.sh` / `common.sh` / `resource_check.sh` 等）
- **结果原始文件**: `results/vllm_acp/<model>/ep4-p{30,50}-padded/{run,ep4-20260923-001917}/tasks/<task>/**/*_results.json`

## 1. 实验设置

- **剪枝算法（ACP）**: `scripts/build_ep4_pruning_plan.py`
  - `intra_method=first_attr_coverage`
  - `modality_aware=0`
  - `intra_expert_metric=gateup_act`
  - 通道对齐（新增）：`align_inter=128`, `min_per_expert=128`, `adjust_method=largest_channel`
    （见 `src/generate_mask/adjusters/align.py::adjust_with_align`，剪枝后把每个专家保留的通道数取整到 128 的倍数）
- **部署策略**: `MAES_EP4_STRATEGY=padded`（原生 vLLM EP4 zero-pad 到统一宽度，不做 MAES 自己的跨层重排，即"naive"基线）
- **并行度**: 4×H20, EP4（4-way expert-parallel）
- **剪枝比例**: p=0.3 与 p=0.5（channel-level FLOPs 剪枝率）
- **模型**（3 个）:
  - `moonshotai/Kimi-VL-A3B-Instruct`
  - `Qwen/Qwen3-VL-30B-A3B-Instruct`
  - `OpenGVLab/InternVL3_5-30B-A3B-HF`
- **评测方式**: lmms-eval，vLLM 优化路径（非逐样本 HF generate），**每个 benchmark 随机取 50% 样本**
  （`RANDOM_SUBSET_FRACTION=0.5`, `RANDOM_SUBSET_SEED=42`，来自 `patches/lmms_eval_random_subset.patch`）
- **计划覆盖的 benchmark（12 个）**: gqa, textvqa_val, chartqa, mmstar, mmbench_en_dev_static_local,
  mmvet, mme, realworldqa, videomme, longvideobench_val_v, egoschema_subset(_local), mvbench_available_3800
  - `coco2017_cap_val_local` 与 `video_mmmu_local` 按用户要求明确推迟到之后单独跑，本轮不包含。

## 2. 数据完整性说明（请先读这部分再看表格）

1. **六个模型×剪枝率组合的非 VideoMME predict 任务均已完成**：每组已有 11/12 个完成标记。InternVL3.5 p=0.5 的 `longvideobench_val_v`、`egoschema_subset_local`、`mvbench_available_3800` 和 `mmvet` predict 已于 17:06 前完成。
2. **`videomme` 已完成 2/6**：Kimi p=0.3 与 InternVL p=0.3 已产出真实结果；当前 Kimi p=0.5 正在运行，约 `488/1350`。已有任务均按完成签名跳过，没有重跑。
3. **Qwen VideoMME 尚未开始推理**：旧 Qwen RUN_DIR 的实验身份是 `qwen3_native_video=1`，统一 dispatcher 曾强制传入 `0`，因此被身份校验正确拒绝。后续必须保持 Qwen 原生视频协议并识别 `videomme_qwen3_vllm.complete`，不能在原目录混用不同设置。
4. **`mmvet` 只完成了 `--predict_only` 预测阶段**（`status/mmvet.complete` 已生成），真实分数需要额外启动本地
   judge 模型对预测结果打分（`scripts/judge_vllm_predictions.py`），这一步还没有执行。
   下表中 mmvet 列显示占位值 `bypass,none=999`（lmms-eval 在缺少 judge 结果时的固定占位符），**不是真实分数**。
5. `coco2017_cap_val_local`、`video_mmmu_local` 按用户指示本轮不跑，未包含在表中。
6. 其余列出的每一项分数均来自 `results/vllm_acp/**/*_results.json` 的真实 lmms-eval 输出，非预估/推断。

## 3. 结果表（p = 0.3）

| Benchmark | 指标 | Kimi-VL-A3B | Qwen3-VL-30B-A3B | InternVL3.5-30B-A3B |
|---|---|---:|---:|---:|
| gqa | exact_match | 61.95% | 60.82% | 61.36% |
| textvqa_val | exact_match | 84.36% | 81.77% | 76.87% |
| chartqa | relaxed_overall | 85.84% | 82.56% | 87.04% |
| mmstar | average | 63.26% | 65.77% | 69.14% |
| mmbench_en_dev_static_local | gpt_eval_score | 81.86 | 84.86 | 82.86 |
| mme | perception / cognition | 1678.4 / 542.3 | 1690.2 / 517.0 | 1652.8 / 693.6 |
| realworldqa | exact_match | 66.06% | 70.23% | 67.10% |
| longvideobench_val_v | lvb_acc | 61.29% | 64.13% | 57.70% |
| egoschema_subset | score | 69.60% | 64.80% | 80.80% |
| mvbench_available_3800 | accuracy | 59.21% | 61.42% | 69.79% |
| videomme | perception_score | 63.78% | *待修复 native-video 配置后跑* | 58.44% |
| mmvet | gpt_eval_score | *predict_only 完成，judge 待跑* | *同左* | *同左* |

## 4. 结果表（p = 0.5）

| Benchmark | 指标 | Kimi-VL-A3B | Qwen3-VL-30B-A3B | InternVL3.5-30B-A3B |
|---|---|---:|---:|---:|
| gqa | exact_match | 59.09% | 58.61% | 59.29% |
| textvqa_val | exact_match | 80.12% | 79.76% | 74.11% |
| chartqa | relaxed_overall | 78.88% | 81.20% | 79.44% |
| mmstar | average | 60.79% | 60.19% | 63.55% |
| mmbench_en_dev_static_local | gpt_eval_score | 78.87 | 81.86 | 79.87 |
| mme | perception / cognition | 1548.3 / 524.0 | 1591.3 / 440.3 | 1541.8 / 593.0 |
| realworldqa | exact_match | 63.71% | 70.76% | 66.32% |
| longvideobench_val_v | lvb_acc | 57.40% | 62.48% | 56.80% |
| egoschema_subset | score | 65.20% | 68.00% | 72.00% |
| mvbench_available_3800 | accuracy | 56.89% | 58.58% | 64.26% |
| videomme | perception_score | *进行中（488/1350）* | *待修复 native-video 配置后跑* | *待跑* |
| mmvet | gpt_eval_score | *predict_only 完成，judge 待跑* | *同左* | *同左* |

## 5. 观察（初步，基于目前已完成的数据）

- 三个模型在 p=0.3→p=0.5 时，图文类 benchmark（gqa/textvqa/chartqa/mmstar/realworldqa/mmbench）普遍呈现
  小幅、单调的精度下降，符合预期（剪枝比例越高精度越低）。
- InternVL3.5-30B 在两个剪枝比例下的 egoschema_subset 分数（80.8% / 72.0%）和 mvbench（69.8% / 64.3%）高于
  另外两个模型；这些结果来自固定 seed 的 50% 子集，比较时需保留该口径。
- mme 的 perception 分数在三模型间比较接近（p=0.3 时约 1650–1690），cognition 分数上 InternVL3.5 明显更高
  （p=0.3: 693.6 vs Kimi 542.3 / Qwen3 517.0），这一差距在 p=0.5 时依然保持（593.0 vs 524.0 / 440.3）。

## 6. 待办（下一步）

1. 等当前 Kimi p=0.5 VideoMME 完成，再运行 InternVL p=0.5；两者继续复用原 RUN_DIR 和 response cache。
2. 修正 Qwen 的完成标记逻辑：保持 `qwen3_native_video=1`，使用 `videomme_qwen3_vllm` 任务和 marker，再补 Qwen p=0.3/p=0.5；不复用设置不同的实验产物。
3. 启动本地 judge 服务，跑 `scripts/judge_vllm_predictions.py` 为全部 6 个 模型×剪枝率 组合计算真实
   MMVet 分数。
4. 全部跑完后，按用户指示补跑 `coco2017_cap_val_local` 与 `video_mmmu_local`。
