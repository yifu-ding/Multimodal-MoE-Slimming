# ACP (first_attr_coverage, non-modality-aware) p=0.3 / p=0.5 真实 vLLM EP4 评测

2026-09-23 用户授权。范围：

- 3 个模型：Kimi-VL-A3B-Instruct、Qwen3-VL-30B-A3B-Instruct、InternVL3_5-30B-A3B-HF。
- 剪枝比例：p=0.3 和 p=0.5（先完成三个模型的 p=0.3，再跑 p=0.5）。
- 部署策略：`padded`（vLLM 原生 padding，naive，不做 cross-layer 重排）。
- 4 卡 EP4 并行（`enable_expert_parallel=True`、`tensor_parallel_size=4`）。
- EP4 plan 生成参数：`intra_method=first_attr_coverage`、`modality_aware=0`、
  `intra_expert_metric=gateup_act`、`align_inter=128`、`min_per_expert=128`、
  `adjust_method=largest_channel`（其余保持 `build_ep4_pruning_plan.py` 默认：
  `inter_method=loss_smooth_2`、`smooth_fn=sqrt`、`ema_source_key=ema_matrix`、
  `shared_protect=1`、`use_ema=1`）。Scores 复用现有 mixed-512 校准产物
  （`storage/scores/{kimi-mixed-512,qwen3-mixed-512,internvl3_5-30b-a3b-mixed-512}/scores.pt`）。
- 12 个 benchmark：gqa, textvqa_val, chartqa, mmstar,
  mmbench_en_dev_static_local, mmvet, mme, realworldqa, videomme,
  longvideobench_val_v, egoschema_subset, mvbench_available_3800。
  `videomme` 现在跑（放在 coco/video_mmmu 之前）；`coco2017_cap_val_local` 和
  `video_mmmu_local` 仍明确放到最后一轮单独处理。
  注：Qwen3-VL-30B 自己的 baseline 脚本默认 `ENABLE_QWEN3_NATIVE_VIDEO=1`
  会把 videomme 重命名成 `videomme_qwen3_vllm`（Kimi/InternVL 的 wrapper 强制
  为 0，用的是 `videomme`），pipeline.sh 统一显式传 0，确保三个模型的完成
  标记文件名一致，都是 `status/videomme.complete`。
- 每个 benchmark 随机抽样 50%（`RANDOM_SUBSET_FRACTION=0.5`，
  `RANDOM_SUBSET_MIN_SAMPLES=1` 避免被下限拉高，`RANDOM_SUBSET_SEED=42`）。

不在本次授权范围内：COCO、VideoMMMU、`cross_layer`/`multi_kernel`/
`single_width` 策略、Mistral/Qwen3-VL-235B。

## 2026-09-23 更新：align 粒度改为 256

用户指示：**后续所有实验都把 align 改成 256**（即 `--align-inter 256
--min-per-expert 256`，`--adjust-method` 仍为 `largest_channel`）。

- 本轮（p=0.3/p=0.5，`runtime/ep4_plans/acp/*.pt`）已经用 align=128 构建并跑完，
  **不回溯重建**，`docs/ACP实验结果汇总.md` 里的数据继续按 align=128 标注有效。
- 从现在起，任何新建的 ACP EP4 plan（新剪枝率、新模型，或 COCO/VideoMMMU
  补跑如果需要新 plan）一律使用 align=256，不再用 128。
