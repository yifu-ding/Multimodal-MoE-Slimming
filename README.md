# MAES

Anonymous implementation of **MAES**, a training-free framework for structured
compression and efficient deployment of multimodal Mixture-of-Experts models.

This release contains the core research code needed to reproduce the method:

- **MACS**: modality-aware channel scoring and budget allocation.
- **SOES**: expert sensitivity estimation with configurable reconstruction losses.
- **CLER**: deployment-friendly width quantization and cross-layer expert placement.
- Structural pruning and an optional vLLM expert-parallel runtime.

The repository intentionally excludes checkpoints, datasets, generated results,
paper figures, cluster-specific launch scripts, and internal experiment logs.

## Repository layout

```text
src/calibration/       calibration and second-order score collection
src/generate_mask/     MACS budgets, masks, and CLER placement
src/prune.py           in-memory structured pruning
src/vllm_*             plan validation and vLLM runtime integration
tasks/                 dataset adapters used during calibration/evaluation
scripts/               minimal command-line entry points
runtime/vllm_ep4/      vLLM bootstrap hooks
```

## Installation

Python 3.11 and CUDA-capable PyTorch are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights and benchmark data are not redistributed. Configure local paths
with standard Hugging Face variables and the task-specific variables below:

```bash
export HF_HOME=./data/huggingface-cache
export MVBENCH_ROOT=./data/MVBench
export VIDEO_MME_ROOT=./data/Video-MME
export VIDEO_MMMU_ROOT=./data/VideoMMMU
export LONGVIDEOBENCH_ROOT=./data/LongVideoBench
export STAR_ROOT=./data/STAR
```

## 1. Collect calibration scores

The default reconstruction loss is relative L2. Use `--loss_fn l2` for the
quadratic loss used by the exact single-expert sensitivity criterion.

```bash
python -m src.calibration.collect_scores_main \
  --model_name_or_path ./models/Kimi-VL-A3B-Instruct \
  --output_dir outputs/calibration \
  --dataset gqa \
  --num_samples 128 \
  --token_per_sample 2048 \
  --loss_fn l2 \
  --modality_aware
```

`--num_samples` controls only the calibration subset. The evaluation entry point intentionally exposes no sample-limit or subset arguments and always evaluates the complete benchmark split.

Run `python -m src.calibration.prepare_mixed_calibration --help` to construct a
fixed mixed-modality calibration manifest.

## 2. Build a MAES pruning and EP plan

```bash
python scripts/build_ep4_pruning_plan.py \
  --scores outputs/calibration/scores.pt \
  --output outputs/maes-plan.pt \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --prune-ratio 0.5
```

The generated plan is validated by `src/vllm_ep4_plan.py` and contains channel
masks, quantized expert widths, and the cross-layer expert placement.

## 3. Prune and evaluate

```bash
python scripts/prune_and_eval_kimi_gqa.py \
  --model_path ./models/Kimi-VL-A3B-Instruct \
  --scores_path outputs/calibration/scores.pt \
  --task gqa \
  --prune_ratio 0.5 \
  --output_dir outputs/evaluation
```

The script supports the released model loaders and dataset adapters. Use
`--help` for model- and task-specific options.

## 4. Cross-layer placement analysis

```bash
python scripts/run_placement_ablation.py table6 \
  --plans outputs/maes-plan.pt \
  --output outputs/placement.json
```

This entry point exposes the greedy, pairwise-swap, exhaustive, beam, tabu,
simulated-annealing, and MILP refinement operators used in the paper analysis.

## vLLM integration

Set the repository root on `PYTHONPATH`, point the bootstrap at a validated
plan, and start vLLM in the target environment:

```bash
export PYTHONPATH="$PWD/runtime/vllm_ep4:$PWD:${PYTHONPATH:-}"
export MAES_EP4_PLAN=./outputs/maes-plan.pt
```

`runtime/vllm_ep4/sitecustomize.py` installs the runtime hooks when the Python
process starts. Exact vLLM compatibility depends on the selected model and
vLLM version.

## License

See `LICENCE`.
