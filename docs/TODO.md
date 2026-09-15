1. qwen3 videomme egoschema_subset, mvbench_available_3800


```sh
DECORD_EOF_RETRY_MAX=20480 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PARALLEL_MODE=ep4 \
TASKS="videomme" \
RUN_DIR=/home/dyf/code/distill/MAES/results/vllm_baseline/qwen3-vl-30b-a3b/ep4-videomme-native-video \
bash scripts/run_qwen3_vl_vllm_baseline.sh
```



```sh
cd /home/dyf/code/distill/MAES

DECORD_EOF_RETRY_MAX=20480 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PARALLEL_MODE=ep4 \
TASKS="egoschema_subset,mvbench_available_3800" \
RUN_DIR=/home/dyf/code/distill/MAES/results/vllm_baseline/qwen3-vl-30b-a3b/ep4-full-no-mvbench \
bash scripts/run_qwen3_vl_vllm_baseline.sh
```

结果写入本地md中（/home/dyf/code/distill/MAES/docs/自动化执行结果.md），只需要评测结果，按照 | Model | Pruning | Task | n-shot | Metric | Value | Stderr | Evaluation time | total_gen_tokens (tokens) | avg_speed (tokens/s) | avg_tpot (seconds/token) | avg_ttft (seconds) | 给我 md table

---

> 🔴 当前执行位置

2. kimi 复现

- 模仿 scripts/run_qwen3_vl_vllm_baseline.sh 写一个 scripts/run_kimi_vl_vllm_baseline.sh
- 运行
```sh
cd /home/dyf/code/distill/MAES

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PARALLEL_MODE=ep4 \
RUN_DIR=/home/dyf/code/distill/MAES/results/vllm_baseline/kimi-vl-30b-a3b/ep4-full-all \
TASKS="gqa,coco2017_cap_val_local,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,videomme,longvideobench_val_v,video_mmmu_local,egoschema_subset,mvbench_available_3800,mmbench_en_dev_static_local,mmvet" \
bash scripts/run_kimi_vl_vllm_baseline.sh
```

- 4卡评测完全结束后，找一张空卡运行 ：
```sh
cd /home/dyf/code/distill/MAES
JUDGE_MODEL=/home/data/dyf/models/Qwen2.5-32B-Instruct \
  CUDA_VISIBLE_DEVICES=0 \
  TENSOR_PARALLEL_SIZE=1 \
  bash scripts/serve_vllm_mm_judge.sh
```

另开窗口运行：
```sh
cd /home/dyf/code/distill/MAES
conda run --no-capture-output -n vllm-maes \
	python scripts/judge_vllm_predictions.py \
		--predictions-dir \
		/home/dyf/code/distill/MAES/results/vllm_baseline/kimi-vl-30b-a3b/ep4-full-all  \
		--tasks mmvet,mmbench,videommmu \
		--workers 8
```


所有评测结果写入本地md中（/home/dyf/code/distill/MAES/docs/自动化执行结果.md），只需要评测结果，按照 | Model | Pruning | Task | n-shot | Metric | Value | Stderr | Evaluation time | total_gen_tokens (tokens) | avg_speed (tokens/s) | avg_tpot (seconds/token) | avg_ttft (seconds) | 给我 md table


3. InternVL-3.5-30B-A3B-HF 复现

同kimi复现，改internvl脚本，然后跑所有任务，把该用judge来评测的任务评测一下

---

3. ours scores 两阶段生成

使用说明在 docs/mixed_calibration.md

第一阶段使用：

```sh
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct \
OUTPUT_MANIFEST=storage/calibration_manifests/qwen3-mixed-512.json \
CANDIDATE_POOL_SIZE=4096 \
NUM_SAMPLES=512 \
TOTAL_SCORE_TOKENS=262144 \
MIN_SAMPLE_TOKENS=64 \
bash scripts/prepare_mixed_calibration.sh
```

不要再传 `SCORE_TOKENS_PER_SAMPLE=2048`。运行结束会打印四源样本数，以及 quota 的 `total/min/median/max/by_source`。

四卡收集 scores：

```sh
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct \
SELECTION_MANIFEST=storage/calibration_manifests/qwen3-mixed-512.json \
OUTPUT_DIR=storage/scores/qwen3-mixed-512 \
GPUS=0,1,2,3 \
BATCH_SIZE=16 \
SECOND_ORDER_CHUNK_SIZE=auto \
AGGREGATION=mean \
HESSIAN_PROBE_LAYER=0 \
bash scripts/run_collect_scores_4gpu.sh
```

第 0 层完成后会立即生成：

```sh
storage/scores/qwen3-mixed-512/hessian_probe_L0.pt
```

最终生成的scores在 `storage/scores/qwen3-mixed-512/scores.pt` 

---

4. 使用生成的scores 进行剪枝和跨层EP4分配

剪枝算法代码中已实现，只需调用。脚本中的配置为

```sh
PRUNE_RATIO="${PRUNE_RATIO:-0.3}" 
INTER_METHOD="${INTER_METHOD:-loss_smooth_2}"
SMOOTH_FN="${SMOOTH_FN:-sqrt}"
INTRA_METHOD="${INTRA_METHOD:-second_attr_coverage}"
MODALITY_AWARE="${MODALITY_AWARE:-1}"  # 是否开启双模态
SHARED_PROTECT="${SHARED_PROTECT:-1}"  # 双模态时是否保留 shared channels
TEXT_ONLY="${TEXT_ONLY:-0}"  # ablation: 仅使用 text tentative mask
VISUAL_ONLY="${VISUAL_ONLY:-0}"  # ablation: 仅使用 visual tentative mask
NORMALIZE="${NORMALIZE:-0}"  # 是否对 text / visual 分模态做层级归一化
EXPERTWISE_BUDGET_NORMALIZE="${EXPERTWISE_BUDGET_NORMALIZE:-1}"  # 是否先按层归一化 expert raw budget 再分配
USE_EMA="${USE_EMA:-1}"  # 是否使用 EMA affinity 分配 modality budget
EMA_SOURCE_KEY="${EMA_SOURCE_KEY:-ema_matrix}"
INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC:-gateup_act}"
LAYERWISE_LOSS_KEY="${LAYERWISE_LOSS_KEY:-layerwise_loss}"
```

剪枝后，用跨层EP4分配算法，也是代码中新实现的，代码在 `/home/dyf/code/distill/MAES/src/generate_mask/ep4_intplan.py`

```python
DEFAULT_WIDTHS = (0, 384, 512, 640, 768)  # 这是qwen3模型的setting
```

该算法会分配给 gpu0～3 各自放置的专家宽度档位，跨层档位配置可不同，每个GPU上差不多均匀分配参数量，也即EP4并行。

5. 评测Ours

用第四步跨层EP4分配算法把experts放到gpu0~3之后，就可以调用vllm进行测试。由于还存在width=0的expert，是直接remove的，对于这些直接被remove的expert，它的token-expert pair ID应该是-1，这样vllm能直接把-1认为无效token，不会进行后续dispatch和专家计算，会直接跳过。这部分代码尚未检查，可以检查一下确保如期执行。

检查完上一步之后，可以评测一下30%剪枝、50%剪枝后的3个模型，在各个数据集上的表现。结果依旧放到  /home/dyf/code/distill/MAES/docs/自动化执行结果.md 中，按照 | Model | Pruning | Task | n-shot | Metric | Value | Stderr | Evaluation time | total_gen_tokens (tokens) | avg_speed (tokens/s) | avg_tpot (seconds/token) | avg_ttft (seconds) | 给我单独的 md table
